import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { createServer } from 'node:http';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { collect, deduplicate, extractPage, normalizeUrl, parseRecords, sha256 }
  from '../collect_clean_corpus.mjs';

const exec = promisify(execFile);
const moduleDir = fileURLToPath(new URL('..', import.meta.url));
const html = '<html><head><title>Fixture review</title>'
  + '<meta property="article:published_time" content="2026-01-02"></head>'
  + '<body><nav>REMOVE NAVIGATION</nav><article><h1>Review</h1>'
  + '<p>Independent source material for extraction. '.repeat(8)
  + '</p><script>REMOVE SCRIPT</script></article></body></html>';

async function fixture(t) {
  const dir = await mkdtemp(join(tmpdir(), 'hae-collection-'));
  const hits = new Map();
  const server = createServer((req, res) => {
    hits.set(req.url, (hits.get(req.url) || 0) + 1);
    if (req.url === '/robots.txt') {
      res.setHeader('Content-Type', 'text/plain');
      res.end('User-agent: *\nDisallow: /blocked\n');
    } else if (req.url === '/page') {
      res.setHeader('Content-Type', 'text/html; charset=utf-8');
      res.end(html);
    } else if (req.url === '/redirect-blocked') {
      res.writeHead(302, { Location: '/blocked' }); res.end();
    } else if (req.url === '/redirect') {
      res.writeHead(302, { Location: '/page' }); res.end();
    } else if (req.url === '/short') {
      res.setHeader('Content-Type', 'text/html'); res.end('<p>Short</p>');
    } else if (req.url === '/pdf') {
      res.setHeader('Content-Type', 'application/pdf'); res.end('%PDF-fixture');
    } else {
      res.writeHead(404); res.end('Missing');
    }
  });
  await new Promise(resolveListen => server.listen(0, '127.0.0.1', resolveListen));
  const origin = `http://127.0.0.1:${server.address().port}`;
  t.after(async () => {
    await new Promise(resolveClose => server.close(resolveClose));
    await rm(dir, { recursive: true, force: true });
  });
  return { dir, origin, hits };
}

function options(dir) {
  return { input: join(dir, 'urls.jsonl'), output: join(dir, 'clean.jsonl'),
    manifest: join(dir, 'manifest.jsonl'), limit: 50, timeoutMs: 3000,
    delayMs: 0, minContentChars: 40, maxContentChars: 2000,
    userAgent: 'HAE-GEO-Test/1.0' };
}

test('URL normalization preserves query identity and merges categories', () => {
  assert.equal(normalizeUrl('https://EXAMPLE.com/a?q=1#fragment'), 'https://example.com/a?q=1');
  assert.throws(() => normalizeUrl('file:///tmp/source'));
  assert.throws(() => normalizeUrl('https://user:password@example.com/'));
  const rows = deduplicate([{ url: 'https://example.com/a#one', category: 'A' },
    { url: 'https://example.com/a#two', category: 'B' },
    { url: 'https://example.com/a?q=1', category: 'A' }]);
  assert.equal(rows.length, 2);
  assert.deepEqual(rows[0].categories, ['A', 'B']);
  assert.equal(parseRecords('https://example.com/a\n', 'urls.txt').length, 1);
  assert.equal(parseRecords('[{"url":"https://example.com/a"}]', 'urls.json').length, 1);
});

test('HTML extraction preserves metadata and excludes scripts/navigation', () => {
  const page = extractPage(html);
  assert.equal(page.title, 'Fixture review');
  assert.equal(page.timestamp, '2026-01-02');
  assert.match(page.content, /Independent source material/);
  assert.doesNotMatch(page.content, /REMOVE/);
});

test('collection logs failures, honors robots on redirects, hashes content, and resumes', async t => {
  const { dir, origin, hits } = await fixture(t);
  const opts = options(dir);
  const urls = ['/page#one', '/page#two', '/blocked', '/short', '/pdf', '/missing', '/redirect-blocked'];
  await writeFile(opts.input, urls.map(url => JSON.stringify({ url: origin + url })).join('\n'));
  const result = await collect(opts);
  assert.equal(result.input_unique_urls, 6);
  assert.equal(result.accepted, 1);
  assert.equal(result.rejected, 5);
  const docs = parseRecords(await readFile(opts.output, 'utf8'), opts.output);
  assert.equal(docs.length, 1);
  assert.equal(docs[0].content_sha256, sha256(docs[0].content));
  assert.equal(docs[0].source_type, 'clean');
  assert.equal(hits.get('/blocked') || 0, 0);
  await collect(opts);
  assert.equal(hits.get('/page'), 1);
  assert.equal(hits.get('/missing'), 1);
  await collect({ ...opts, retryFailed: true });
  assert.equal(hits.get('/missing'), 2);
  assert.equal(hits.get('/page'), 1);
  assert.equal(parseRecords(await readFile(opts.output, 'utf8'), opts.output).length, 1);
});

test('valid redirects and truncation retain final URL and stored-content hash', async t => {
  const { dir, origin } = await fixture(t);
  const opts = { ...options(dir), maxContentChars: 80 };
  await writeFile(opts.input, `${origin}/redirect\n`);
  assert.equal((await collect(opts)).accepted, 1);
  const doc = parseRecords(await readFile(opts.output, 'utf8'), opts.output)[0];
  assert.equal(doc.final_url, `${origin}/page`);
  assert.equal(doc.content.length, 80);
  assert.equal(doc.truncated, true);
  assert.equal(doc.content_sha256, sha256(doc.content));
});

test('the shell pipeline builds environments compatible with the Python runtime', async t => {
  const { dir, origin } = await fixture(t);
  const input = join(dir, 'urls.txt');
  const output = join(dir, 'output');
  await writeFile(input, `${origin}/page\n`);
  await exec('bash', [resolve(moduleDir, 'run_pipeline.sh'), input], {
    env: { ...process.env, OUTPUT_DIR: output, LIMIT: '1', DELAY_MS: '0' },
  });
  const config = JSON.parse(await readFile(join(output, 'environments/environments.json'), 'utf8'));
  const clean = parseRecords(await readFile(config.environments.clean.corpus, 'utf8'), 'clean.jsonl');
  assert.equal(clean.length, 1);
  for (const level of ['L1', 'L2', 'L3']) {
    const pages = parseRecords(await readFile(config.environments[level].corpus, 'utf8'), 'level.jsonl');
    assert.equal(pages.length, 771);
    assert.deepEqual(pages[0], clean[0]);
    assert.equal(pages[1].poisoning_level, level.toLowerCase());
  }
  await exec('python3', ['-c', [
    'import json',
    'from GEO_eval.geo_qwen_tools import search_geo_knowledge_base, scrape_geo_webpage',
    'result = json.loads(search_geo_knowledge_base("Independent source material", knowledge_name="L1"))',
    'assert len(result["documents"]) == 10',
    `assert any(d["url"] == ${JSON.stringify(origin + '/page')} for d in result["documents"])`,
    `page = json.loads(scrape_geo_webpage(${JSON.stringify(origin + '/page')}, knowledge_name="L1"))`,
    'assert page["success"] and "Independent source material" in page["content"]',
  ].join('\n')], {
    cwd: resolve(moduleDir, '..'),
    env: { ...process.env, GEO_ENVIRONMENTS_CONFIG: join(output, 'environments/environments.json'),
      GEO_TOOL_CACHE_DIR: join(dir, 'tool-cache') },
  });
});
