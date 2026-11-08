#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { appendFile, mkdir, readFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { setTimeout as delay } from 'node:timers/promises';
import axios from 'axios';
import { load } from 'cheerio';
import robotsParser from 'robots-parser';

export const sha256 = value => createHash('sha256').update(value).digest('hex');

export function normalizeUrl(value) {
  const url = new URL(String(value).trim());
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password) {
    throw new Error('Expected an HTTP(S) URL without embedded credentials');
  }
  url.hash = '';
  return url.href;
}

export function parseRecords(text, filename) {
  const trimmed = text.trim();
  if (!trimmed) return [];
  if (filename.endsWith('.json')) {
    const parsed = JSON.parse(trimmed);
    const rows = Array.isArray(parsed) ? parsed : parsed.documents;
    if (!Array.isArray(rows)) throw new Error('JSON input must be an array or documents object');
    return rows;
  }
  return trimmed.split(/\r?\n/).filter(line => line.trim()).map((line, index) => {
    if (/^https?:\/\//i.test(line.trim())) return { url: line.trim() };
    try { return JSON.parse(line); }
    catch { throw new Error(`Invalid URL or JSON record at ${filename}:${index + 1}`); }
  });
}

export function deduplicate(rows) {
  const urls = new Map();
  for (const raw of rows) {
    const row = typeof raw === 'string' ? { url: raw } : raw;
    const url = normalizeUrl(row.url);
    const categories = [...(row.categories || []), row.category].filter(Boolean);
    if (!urls.has(url)) urls.set(url, { ...row, url, categories: [] });
    const item = urls.get(url);
    item.categories = [...new Set([...item.categories, ...categories])];
  }
  return [...urls.values()];
}

export function extractPage(html) {
  const $ = load(html);
  const title = $('title').first().text().trim() || $('h1').first().text().trim();
  const timestamp = $('meta[property="article:published_time"]').attr('content')
    || $('meta[name="date"]').attr('content') || '';
  $('script, style, noscript, iframe, svg, canvas, nav, footer, header, form').remove();
  $('br').replaceWith('\n');
  $('p, div, section, li, h1, h2, h3, h4, tr').append('\n');
  const article = $('article').first();
  const main = $('main').first();
  const body = article.length ? article : main.length ? main : $('body');
  const content = body.text().replace(/\r/g, '').split('\n')
    .map(line => line.replace(/[^\S\n]+/g, ' ').trim()).filter(Boolean).join('\n');
  return { title, content, timestamp };
}

function decodeHtml(data, contentType) {
  const bytes = Buffer.from(data);
  const headerCharset = /charset\s*=\s*["']?([^\s;"']+)/i.exec(contentType)?.[1];
  const metaCharset = /charset\s*=\s*["']?([\w-]+)/i.exec(bytes.subarray(0, 8192).toString('latin1'))?.[1];
  const charset = headerCharset || metaCharset || 'utf-8';
  try { return new TextDecoder(charset).decode(bytes); }
  catch { return new TextDecoder('utf-8').decode(bytes); }
}

async function existingRecords(path) {
  try { return parseRecords(await readFile(path, 'utf8'), path); }
  catch (error) { if (error.code === 'ENOENT') return []; throw error; }
}

async function append(path, record) {
  await appendFile(path, `${JSON.stringify(record)}\n`);
}

export async function collect(options) {
  const output = resolve(options.output);
  const manifest = resolve(options.manifest || `${output}.manifest.jsonl`);
  if ([output, manifest].includes(resolve(options.input)) || output === manifest) {
    throw new Error('Input, corpus, and manifest must use different paths');
  }
  await mkdir(dirname(output), { recursive: true });
  await mkdir(dirname(manifest), { recursive: true });
  const old = await existingRecords(output);
  const done = new Set(old.filter(row => row.content).map(row => normalizeUrl(row.url)));
  const attempted = new Set((await existingRecords(manifest)).map(row => normalizeUrl(row.url)));
  const records = deduplicate(parseRecords(await readFile(options.input, 'utf8'), options.input));
  const selected = records.filter(row => !done.has(row.url)
    && (options.retryFailed || !attempted.has(row.url))).slice(0, options.limit);
  const robots = new Map();
  const counters = { input_unique_urls: records.length, already_collected: done.size,
    attempted: 0, accepted: 0, rejected: 0 };
  const http = axios.create({ timeout: options.timeoutMs, responseType: 'arraybuffer',
    maxContentLength: 10 * 1024 * 1024, maxRedirects: 5,
    headers: { 'User-Agent': options.userAgent },
    validateStatus: () => true });
  function get(url, config = {}) {
    const local = ['localhost', '127.0.0.1', '[::1]'].includes(new URL(url).hostname);
    return http.get(url, { ...config, ...(local ? { proxy: false } : {}) });
  }
  async function checkRobots(url) {
    const origin = new URL(url).origin;
    if (!robots.has(origin)) {
      const location = `${origin}/robots.txt`;
      const response = await get(location);
      if (response.status === 401 || response.status === 403) {
        robots.set(origin, robotsParser(location, 'User-agent: *\nDisallow: /'));
      } else if (response.status === 404 || response.status === 410) {
        robots.set(origin, robotsParser(location, ''));
      } else if (response.status >= 200 && response.status < 300) {
        robots.set(origin, robotsParser(location, Buffer.from(response.data).toString('utf8')));
      } else {
        throw new Error(`Cannot read robots.txt: HTTP ${response.status}`);
      }
    }
    if (robots.get(origin).isAllowed(url, options.userAgent) === false) {
      throw new Error('robots.txt disallows this URL');
    }
  }
  for (const row of selected) {
    const result = { url: row.url, categories: row.categories,
      retrieved_at: new Date().toISOString(), status: 'error', content_sha256: null };
    counters.attempted += 1;
    try {
      let currentUrl = row.url;
      let response;
      for (let hop = 0; hop <= 5; hop += 1) {
        await checkRobots(currentUrl);
        response = await get(currentUrl, { maxRedirects: 0 });
        if (![301, 302, 303, 307, 308].includes(response.status)) break;
        if (!response.headers.location || hop === 5) throw new Error('Invalid or excessive redirects');
        currentUrl = normalizeUrl(new URL(response.headers.location, currentUrl).href);
      }
      result.http_status = response.status;
      result.final_url = currentUrl;
      if (response.status < 200 || response.status >= 300) {
        throw new Error(`HTTP ${response.status}`);
      }
      const type = String(response.headers['content-type'] || '');
      if (!/text\/html|application\/xhtml\+xml/i.test(type)) {
        throw new Error(`Unsupported content type: ${type || 'missing'}`);
      }
      const bytes = Buffer.from(response.data);
      const page = extractPage(decodeHtml(bytes, type));
      const content = page.content.slice(0, options.maxContentChars);
      if (content.length < options.minContentChars) throw new Error('Page text is too short');
      const record = { id: sha256(row.url), url: row.url,
        title: page.title || row.title || '', timestamp: page.timestamp,
        category: row.category || row.categories[0] || '', categories: row.categories,
        category_labels: row.category_labels || '', discovered_via: row.discovered_via || '',
        content, source_type: 'clean', level: 'clean',
        retrieved_at: result.retrieved_at, final_url: result.final_url,
        content_sha256: sha256(content), response_sha256: sha256(bytes),
        original_content_chars: page.content.length,
        truncated: page.content.length > content.length };
      await append(output, record);
      Object.assign(result, { status: 'accepted', title: record.title,
        content_sha256: record.content_sha256, response_sha256: record.response_sha256,
        content_chars: content.length, truncated: record.truncated });
      counters.accepted += 1;
    } catch (error) {
      result.error = error.message;
      counters.rejected += 1;
    }
    await append(manifest, result);
    if (options.onProgress) options.onProgress(result);
    if (options.delayMs) await delay(options.delayMs);
  }
  // An empty corpus is still a valid file for inspecting a failed/blocked run.
  await appendFile(output, '');
  return counters;
}

function parseArgs(argv) {
  if (argv.includes('--help')) {
    console.log('Usage: node collect_clean_corpus.mjs --input urls.jsonl --output output/clean.jsonl\n'
      + 'Options: --manifest PATH --limit 50 --timeout-ms 15000 --delay-ms 300\n'
      + '         --min-content-chars 100 --max-content-chars 200000 --user-agent NAME\n'
      + 'Input: JSON array, JSONL records with url, or one HTTP(S) URL per line.\n'
      + 'Use --retry-failed to retry previously rejected URLs.\n'
      + 'Runs otherwise skip all previously attempted URLs.');
    return null;
  }
  const options = { limit: 50, timeoutMs: 15000, delayMs: 300,
    minContentChars: 100, maxContentChars: 200000, userAgent: 'HAE-GEO-Research/1.0' };
  const keys = { '--input': 'input', '--output': 'output', '--manifest': 'manifest',
    '--limit': 'limit', '--timeout-ms': 'timeoutMs', '--delay-ms': 'delayMs',
    '--min-content-chars': 'minContentChars', '--max-content-chars': 'maxContentChars',
    '--user-agent': 'userAgent' };
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === '--retry-failed') { options.retryFailed = true; continue; }
    const key = keys[argv[i]];
    if (!key || argv[i + 1] === undefined) throw new Error(`Invalid option ${argv[i]}`);
    options[key] = typeof options[key] === 'number' ? Number(argv[i + 1]) : argv[i + 1];
    i += 1;
  }
  if (!options.input || !options.output) throw new Error('--input and --output are required');
  for (const key of ['limit', 'timeoutMs', 'delayMs', 'minContentChars', 'maxContentChars']) {
    if (!Number.isInteger(options[key]) || options[key] < 0) throw new Error(`Invalid ${key}`);
  }
  if (!options.timeoutMs || !options.maxContentChars
      || options.minContentChars > options.maxContentChars) throw new Error('Invalid size/timeout settings');
  return options;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  try {
    const options = parseArgs(process.argv.slice(2));
    if (options) console.log(JSON.stringify(await collect(options), null, 2));
  } catch (error) { console.error(error.message); process.exitCode = 1; }
}
