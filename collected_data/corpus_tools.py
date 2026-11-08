#!/usr/bin/env python3
"""Export released URL metadata, build environments, and inventory evaluator rules."""

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
from urllib.parse import urldefrag, urlsplit, urlunsplit


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(path):
    if path.suffix == '.json':
        payload = json.loads(path.read_text(encoding='utf-8'))
        return payload if isinstance(payload, list) else payload['documents']
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def export_manifest(source, output):
    urls = {}
    source_rows = records(source)
    for row in source_rows:
        url = urldefrag(str(row.get('url', '')).strip())[0]
        parsed = urlsplit(url)
        if parsed.scheme not in ('http', 'https') or not parsed.netloc:
            raise ValueError(f'Invalid released URL: {url!r}')
        if url not in urls:
            urls[url] = dict(url=url, title=row.get('title', ''), category=row.get('category', ''),
                             categories=[], level='clean', source_type='clean')
        if row.get('category') and row['category'] not in urls[url]['categories']:
            urls[url]['categories'].append(row['category'])
    write_jsonl(output, urls.values())
    metadata = dict(
        coverage='released clean-page subset only; not the complete original clean corpus',
        source_file=source.name, source_sha256=digest(source), source_records=len(source_rows),
        unique_urls=len(urls), manifest_sha256=digest(output),
        historical_body_hashes_available=False,
        note='The source release has no full-page bodies or historical collection times. '
             'New crawl manifests record retrieval times and body/response hashes.',
    )
    write_json(output.with_suffix('.meta.json'), metadata)
    return metadata


def build_environments(clean_path, poison_path, output):
    clean = records(clean_path)
    if not clean:
        raise ValueError('The clean corpus is empty; inspect the crawl manifest before building')
    if any(not str(row.get('content', '')).strip() or not row.get('url') for row in clean):
        raise ValueError('Every clean record needs a URL and nonempty full-page content')
    if len({row['url'] for row in clean}) != len(clean):
        raise ValueError('The clean corpus contains duplicate URLs')
    clean = [{**row, 'source_type': 'clean', 'level': 'clean'} for row in clean]
    groups = {level: [] for level in ('l1', 'l2', 'l3')}
    for page in records(poison_path):
        level = str(page.get('poisoning_level', page.get('level', ''))).lower()
        if level not in groups:
            raise ValueError(f'Unknown poison level: {level!r}')
        if not page.get('url') or not str(page.get('content', '')).strip():
            raise ValueError('Poison records need a URL and nonempty content')
        if page.get('source_type') not in ('generated', 'modified', 'synthetic_attack'):
            raise ValueError('Poison records must retain explicit attack provenance')
        groups[level].append({**page, 'level': level})
    if any(not pages for pages in groups.values()):
        raise ValueError('The poison corpus must include all three levels')
    clean_urls = {p['url'] for p in clean}
    aliases = []
    for level, pages in groups.items():
        urls = [p['url'] for p in pages]
        if clean_urls.intersection(urls):
            raise ValueError(f'URL collision in {level}; resolve corpus identities explicitly')
        duplicates = Counter(urls)
        # Distinct archived snapshots can share an original URL. Give each a
        # local fragment identifier so URL-keyed Search/Scrape caches keep all.
        for index, page in enumerate(pages):
            if duplicates[page['url']] > 1:
                original = page['url']
                identity = json.dumps([level, index, page], sort_keys=True, ensure_ascii=False)
                token = hashlib.sha256(identity.encode('utf-8')).hexdigest()[:16]
                parts = urlsplit(original)
                page['original_url'] = original
                page['url'] = urlunsplit(parts._replace(fragment=f'snapshot={token}'))
                aliases.append(dict(level=level, original_url=original, local_url=page['url']))
        if len({p['url'] for p in pages}) != len(pages):
            raise ValueError(f'Unresolved URL collision in {level}')
    config, stats = {}, {}
    for level, pages in [('clean', clean), *((level, clean + groups[level]) for level in groups)]:
        path = (output / f'{level}.jsonl').resolve()
        write_jsonl(path, pages)
        config['clean' if level == 'clean' else level.upper()] = {'corpus': str(path)}
        stats[level] = dict(records=len(pages), sha256=digest(path))
    write_json(output / 'environments.json', {'environments': config})
    write_json(output / 'corpus_manifest.json', dict(
        source_clean_sha256=digest(clean_path), source_poison_sha256=digest(poison_path),
        clean_records=len(clean), poison_records={k: len(v) for k, v in groups.items()},
        environments=stats, local_snapshot_aliases=aliases,
        note='All environments share exactly the same collected clean background. '
             'The output configuration uses machine-local absolute paths.',
    ))
    return stats


def evaluator_inventory(source, output):
    tree = ast.parse(source.read_text(encoding='utf-8'))
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Add):
                    left, right = node.value.left, node.value.right
                    if isinstance(left, ast.Name) and isinstance(right, ast.Name) and left.id in values and right.id in values:
                        value = values[left.id] + values[right.id]
                    else:
                        continue
                else:
                    continue
            values[name] = value
    vocabulary = {k: sorted(v) if isinstance(v, set) else v for k, v in values.items()
                  if k.isupper() and isinstance(v, (list, tuple, set)) and all(isinstance(x, str) for x in v)}
    payload = dict(
        evaluator_path='rule&rubric/geo_evaluate.py', evaluator_sha256=digest(source),
        rule_metric_version=values.get('RULE_METRIC_VERSION'),
        judge_prompt_version=values.get('JUDGE_PROMPT_VERSION'),
        release_role='English language-adapted public evaluator',
        paper_frozen_equivalence_verified=False,
        named_rule_vocabulary=vocabulary,
        note='This inventory identifies the current public file. A version string or hash alone '
             'does not establish equivalence to the paper evaluator. The implementation also '
             'contains matching logic and regex rules outside these named vocabularies.',
    )
    write_json(output, payload)
    return {key: value for key, value in payload.items() if key != 'named_rule_vocabulary'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    manifest = commands.add_parser('manifest')
    manifest.add_argument('--input', type=Path, default=ROOT / 'benchmark_data/clean_web_pages_subset.json')
    manifest.add_argument('--output', type=Path, default=ROOT / 'collected_data/manifests/clean_urls.jsonl')
    build = commands.add_parser('build')
    build.add_argument('--clean', type=Path, required=True)
    build.add_argument('--poison', type=Path, default=ROOT / 'benchmark_data/geo_attack_pages.json')
    build.add_argument('--output-dir', type=Path, required=True)
    inventory = commands.add_parser('evaluator-inventory')
    inventory.add_argument('--output', type=Path, default=ROOT / 'collected_data/manifests/evaluator_inventory.json')
    args = parser.parse_args()
    if args.command == 'manifest':
        result = export_manifest(args.input, args.output)
    elif args.command == 'build':
        result = build_environments(args.clean, args.poison, args.output_dir)
    else:
        result = evaluator_inventory(ROOT / 'rule&rubric/geo_evaluate.py', args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
