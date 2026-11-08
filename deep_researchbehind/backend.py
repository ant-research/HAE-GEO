"""Build a corpus-backed Xunsearch index and hydrate ranked document IDs."""

import hashlib
import json
from contextlib import closing
from pathlib import Path
import sqlite3
import subprocess
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_records(path):
    if path.suffix == '.jsonl':
        with path.open(encoding='utf-8') as handle:
            return [json.loads(line) for line in handle if line.strip()]
    data = json.loads(path.read_text(encoding='utf-8'))
    return data if isinstance(data, list) else data['documents']


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def php_bridge(action, request, php='php', timeout=120):
    result = subprocess.run(
        [php, str(HERE / 'bridge.php'), action], input=json.dumps(request, ensure_ascii=False),
        text=True, capture_output=True, timeout=timeout, check=False,
    )
    try:
        payload = json.loads(result.stdout)
    except ValueError as error:
        raise RuntimeError(f'Invalid PHP bridge response: {result.stderr[-1000:]}') from error
    if result.returncode or not payload.get('success'):
        raise RuntimeError(payload.get('error') or result.stderr[-1000:])
    return payload


def prepare(config_path, output, root=ROOT, index_server='127.0.0.1:8383',
            search_server='127.0.0.1:8384', endpoint='http://127.0.0.1:8765', corpus_dir=None):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for address in (index_server, search_server):
        if not address or any(c in address for c in '\r\n'):
            raise ValueError('Invalid Xunsearch server address')
    ini = output / 'project.ini'
    ini.write_text(
        'project.name = hae_geo\nproject.default_charset = utf-8\n'
        f'server.index = {index_server}\nserver.search = {search_server}\n\n'
        '[doc_id]\ntype = id\n\n[title]\ntype = title\nweight = 5\n\n'
        '[content]\ntype = body\n', encoding='utf-8',
    )
    data = json.loads(config_path.read_text(encoding='utf-8'))
    environments = data.get('environments', data)
    if not isinstance(environments, dict) or not environments:
        raise ValueError('Environment configuration must contain named environments')
    revision_data = []
    sources = {}
    for name, entry in environments.items():
        value = entry.get('corpus') if isinstance(entry, dict) else entry
        path = Path(value)
        path = corpus_dir / path.name if corpus_dir else path if path.is_absolute() else root / path
        sources[name] = path.resolve()
        revision_data.append((name, sha256(path)))
    revision = hashlib.sha256(json.dumps(dict(
        corpora=sorted(revision_data), ini_sha256=sha256(ini),
        bridge_sha256=sha256(HERE / 'bridge.php'), backend_version=1,
    ), sort_keys=True).encode()).hexdigest()
    store = output / f'documents-{revision[:16]}.sqlite'
    store.unlink(missing_ok=True)
    manifest = dict(index_revision=revision, project_ini=str(ini), store=str(store),
                    backend='xunsearch-bm25', environments={})
    runtime_config = {'environments': {}}
    with closing(sqlite3.connect(store)) as db, db:
        db.execute('CREATE TABLE documents (environment TEXT, doc_id TEXT, url TEXT, payload TEXT, '
                   'PRIMARY KEY(environment, doc_id), UNIQUE(environment, url))')
        for name, path in sources.items():
            rows = read_records(path)
            if not rows:
                raise ValueError(f'Corpus {name} is empty')
            database = 'env_' + hashlib.sha256((revision + name).encode()).hexdigest()[:24]
            index_path = output / f'{database}.jsonl'
            with index_path.open('w', encoding='utf-8') as handle:
                for row in rows:
                    if not isinstance(row, dict) or not row.get('url') or not str(row.get('content', '')).strip():
                        raise ValueError(f'Corpus {name} needs full bodies and unique document URLs')
                    doc_id = hashlib.sha256(str(row['url']).encode()).hexdigest()
                    # Only text and an opaque ID enter the ranking engine.
                    indexed = dict(doc_id=doc_id, title=str(row.get('title', '')), content=str(row['content']))
                    handle.write(json.dumps(indexed, ensure_ascii=False) + '\n')
                    db.execute('INSERT INTO documents VALUES (?, ?, ?, ?)',
                               (name, doc_id, row['url'], json.dumps(row, ensure_ascii=False)))
            manifest['environments'][name] = dict(database=database, records=len(rows),
                corpus_sha256=sha256(path), index_file=str(index_path))
            runtime_config['environments'][name] = dict(
                corpus=str(path), backend='xunsearch', endpoint=endpoint.rstrip('/'),
                index_revision=revision, timeout_seconds=30,
            )
    write_json(output / 'prepared.json', manifest)
    write_json(output / 'environments.pending.json', runtime_config)
    return manifest, runtime_config


def build(config_path, output, root=ROOT, php='php', index_server='127.0.0.1:8383',
          search_server='127.0.0.1:8384', endpoint='http://127.0.0.1:8765', bridge=php_bridge,
          corpus_dir=None):
    output = output.resolve()
    if (output / 'manifest.json').exists():
        raise ValueError('Choose a new output directory for each index build; an active manifest exists')
    manifest, runtime = prepare(config_path, output, root, index_server, search_server, endpoint, corpus_dir)
    for name, entry in manifest['environments'].items():
        bridge('index', dict(project_ini=manifest['project_ini'], database=entry['database'],
                             input=entry['index_file'], expected_n=entry['records']), php=php, timeout=3600)
        # Xunsearch commits asynchronously. Publish only a complete searchable index.
        for attempt in range(60):
            status = bridge('count', dict(project_ini=manifest['project_ini'],
                                         database=entry['database']), php=php)
            if status['count'] == entry['records']:
                break
            time.sleep(1)
        else:
            raise RuntimeError(f'Index {name} did not become searchable with the expected count')
    write_json(output / 'manifest.json', manifest)
    write_json(output / 'environments.json', runtime)
    return manifest


class Backend:
    def __init__(self, manifest_path, php='php', bridge=php_bridge):
        self.manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
        self.php = php
        self.bridge = bridge
        if not Path(self.manifest['store']).is_file():
            raise FileNotFoundError('Indexed document store does not exist')

    def search(self, environment, query, top_k=10, dataset='all', index_revision=None):
        if environment not in self.manifest['environments']:
            raise KeyError('Unknown environment')
        if index_revision != self.manifest['index_revision']:
            raise ValueError('Index revision mismatch; use the matching generated environments.json')
        if not isinstance(query, str) or not query.strip() or len(query) > 8192:
            raise ValueError('query must be a nonempty string of at most 8192 characters')
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 1000:
            raise ValueError('top_k must be between 1 and 1000')
        if not isinstance(dataset, str):
            raise ValueError('dataset must be a string')
        entry = self.manifest['environments'][environment]
        result, seen = [], set()
        offset = 0
        with closing(sqlite3.connect(Path(self.manifest['store']).as_uri() + '?mode=ro', uri=True)) as db:
            while len(result) < top_k and offset < entry['records']:
                limit = min(max(top_k, 100), 1000)
                ranked = self.bridge('search', dict(
                    project_ini=self.manifest['project_ini'], database=entry['database'],
                    query=query, limit=limit, offset=offset,
                ), php=self.php)['hits']
                for hit in ranked:
                    if hit['doc_id'] in seen:
                        raise RuntimeError('Xunsearch returned a duplicate document ID')
                    seen.add(hit['doc_id'])
                    found = db.execute('SELECT payload FROM documents WHERE environment=? AND doc_id=?',
                                       (environment, hit['doc_id'])).fetchone()
                    if not found:
                        raise RuntimeError('Index and document store do not match')
                    document = json.loads(found[0])
                    if dataset not in ('', 'all') and str(document.get('dataset', 'all')) not in (dataset, 'all'):
                        continue
                    result.append({**document, 'score': float(hit['score'])})
                    if len(result) == top_k:
                        break
                offset += len(ranked)
                if len(ranked) < limit:
                    break
        return dict(success=True, environment=environment,
                    index_revision=self.manifest['index_revision'], documents=result)
