from contextlib import contextmanager
from http.server import ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from threading import Thread
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from deep_researchbehind.backend import Backend, build, prepare
from deep_researchbehind.client import search_documents
from deep_researchbehind.server import make_handler
from GEO_eval import geo_qwen_tools as tools
import requests


class RankingFixture:
    """Contract fixture only: this does not implement or test Xunsearch BM25."""

    def __init__(self):
        self.databases = {}

    def __call__(self, action, request, **kwargs):
        database = request['database']
        if action == 'index':
            self.databases[database] = [json.loads(line) for line in
                                        Path(request['input']).read_text().splitlines()]
            return dict(success=True, indexed_n=len(self.databases[database]))
        if action == 'count':
            return dict(success=True, count=len(self.databases[database]))
        rows = list(reversed(self.databases[database]))
        offset, limit = request['offset'], request['limit']
        return dict(success=True, hits=[dict(doc_id=row['doc_id'], score=float(len(rows) - i))
                    for i, row in enumerate(rows) if offset <= i < offset + limit])


@contextmanager
def http_backend(backend):
    with ThreadingHTTPServer(('127.0.0.1', 0), make_handler(backend)) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f'http://127.0.0.1:{server.server_port}'
        finally:
            server.shutdown()
            thread.join(timeout=5)


class XunsearchBackendTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='hae-xunsearch-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root / 'source.json'
        environments = {}
        for level in ('clean', 'L1', 'L2', 'L3'):
            rows = [dict(url='https://clean.example/a', title='Laundry guide', content='Clean body',
                         source_type='clean', dataset='clean')]
            if level != 'clean':
                rows.append(dict(url='https://attack.example/shared', title='Claim',
                                 content=f'{level} PRIVATE_BODY', source_type='generated',
                                 dataset='poison', brand='PRIVATE_BRAND'))
            path = self.root / f'{level}.json'
            path.write_text(json.dumps(rows))
            environments[level] = dict(corpus=str(path))
        self.config.write_text(json.dumps(dict(environments=environments)))
        self.ranker = RankingFixture()

    def create_backend(self):
        output = self.root / 'index'
        manifest = build(self.config, output, bridge=self.ranker)
        return Backend(output / 'manifest.json', bridge=self.ranker), manifest, output

    def test_index_only_contains_text_and_id_and_preserves_provenance_in_store(self):
        backend, manifest, output = self.create_backend()
        self.assertEqual(len({r['database'] for r in manifest['environments'].values()}), 4)
        for rows in self.ranker.databases.values():
            for row in rows:
                self.assertEqual(set(row), {'doc_id', 'title', 'content'})
                self.assertNotIn('PRIVATE_BRAND', json.dumps(row))
        result = backend.search('L1', 'query', index_revision=manifest['index_revision'])
        self.assertEqual(result['documents'][0]['source_type'], 'generated')
        self.assertEqual(result['documents'][0]['brand'], 'PRIVATE_BRAND')
        with self.assertRaisesRegex(ValueError, 'active manifest'):
            build(self.config, output, bridge=self.ranker)

    def test_revision_changes_with_corpus_and_duplicate_urls_fail(self):
        first, _ = prepare(self.config, self.root / 'first')
        path = self.root / 'L1.json'
        rows = json.loads(path.read_text())
        rows[0]['content'] = 'Changed body'
        path.write_text(json.dumps(rows))
        second, _ = prepare(self.config, self.root / 'second')
        self.assertNotEqual(first['index_revision'], second['index_revision'])
        path.write_text(json.dumps(rows + [rows[0]]))
        with self.assertRaises(sqlite3.IntegrityError):
            prepare(self.config, self.root / 'duplicate')

    def test_dataset_filter_and_environment_isolation(self):
        backend, manifest, _ = self.create_backend()
        revision = manifest['index_revision']
        clean = backend.search('clean', 'query', index_revision=revision)['documents']
        self.assertEqual(len(clean), 1)
        self.assertEqual(clean[0]['source_type'], 'clean')
        filtered = backend.search('L1', 'query', dataset='clean', index_revision=revision)['documents']
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]['source_type'], 'clean')
        for environment, query, top_k, rev in [('other', 'q', 10, revision),
                                               ('L1', '', 10, revision),
                                               ('L1', 'q', 0, revision),
                                               ('L1', 'q', 10, 'old')]:
            with self.assertRaises((KeyError, ValueError)):
                backend.search(environment, query, top_k, index_revision=rev)

    def test_failure_does_not_publish_ready_configuration(self):
        def fail(*args, **kwargs):
            raise RuntimeError('Engine unavailable')
        output = self.root / 'failed'
        with self.assertRaisesRegex(RuntimeError, 'Engine unavailable'):
            build(self.config, output, bridge=fail)
        self.assertFalse((output / 'manifest.json').exists())
        self.assertFalse((output / 'environments.json').exists())

    def test_explicit_corpus_directory_remaps_host_paths_for_container_builds(self):
        config = json.loads(self.config.read_text())
        for name, entry in config['environments'].items():
            entry['corpus'] = f'/original-host/corpora/{name}.json'
        self.config.write_text(json.dumps(config))
        manifest, runtime = prepare(self.config, self.root / 'remapped', corpus_dir=self.root)
        self.assertEqual(manifest['environments']['L1']['records'], 2)
        self.assertEqual(runtime['environments']['L1']['corpus'], str((self.root / 'L1.json').resolve()))

    def test_http_api_and_real_runtime_search_scrape_contract(self):
        backend, manifest, output = self.create_backend()
        with http_backend(backend) as endpoint:
            config = json.loads((output / 'environments.json').read_text())
            for entry in config['environments'].values():
                entry['endpoint'] = endpoint
                entry['corpus'] = '/container-only/corpus.jsonl'
            active = self.root / 'active.json'
            active.write_text(json.dumps(config))
            with patch.dict(os.environ, GEO_ENVIRONMENTS_CONFIG=str(active)), \
                    patch.object(tools, 'CACHE_ROOT', self.root / 'cache'):
                tools._environment_config.cache_clear()
                self.addCleanup(tools._environment_config.cache_clear)
                clean = json.loads(tools.search_geo_knowledge_base('query', knowledge_name='clean'))
                self.assertTrue(all(d['source_type'] == 'clean' for d in clean['documents']))
                result = json.loads(tools.search_geo_knowledge_base('query', knowledge_name='L1', top_k=1))
                self.assertEqual(len(result['documents']), 1)
                doc = result['documents'][0]
                self.assertNotIn('content', doc)
                self.assertEqual(doc['source_type'], 'generated')
                page = json.loads(tools.scrape_geo_webpage(doc['url'], knowledge_name='L1'))
                self.assertEqual(page['content'], 'L1 PRIVATE_BODY')
                other = json.loads(tools.scrape_geo_webpage(doc['url'], knowledge_name='L2'))
                self.assertFalse(other['success'])
                tools.search_geo_knowledge_base('query', knowledge_name='L2')
                other = json.loads(tools.scrape_geo_webpage(doc['url'], knowledge_name='L2'))
                self.assertEqual(other['content'], 'L2 PRIVATE_BODY')
                config['environments']['L1']['index_revision'] = 'different-index'
                active.write_text(json.dumps(config))
                tools._environment_config.cache_clear()
                self.assertFalse(json.loads(tools.scrape_geo_webpage(doc['url'], knowledge_name='L1'))['success'])
                with self.assertRaises(requests.HTTPError):
                    tools.search_geo_knowledge_base('query', knowledge_name='L1')

    def test_bad_backend_response_is_not_cached(self):
        entry = dict(endpoint='http://127.0.0.1:8765', index_revision='revision')
        class Response:
            def raise_for_status(self):
                pass

            def json(self):
                return dict(success=True, environment='L2', index_revision='revision', documents=[])
        with patch('requests.Session.post', return_value=Response()):
            with self.assertRaisesRegex(ValueError, 'mismatched'):
                search_documents(entry, 'L1', 'query', 10)

    def test_legacy_local_backend_and_model_projection_stay_compatible(self):
        with patch.dict(os.environ, GEO_ENVIRONMENTS_CONFIG=str(self.config)), \
                patch.object(tools, 'CACHE_ROOT', self.root / 'legacy-cache'):
            tools._environment_config.cache_clear()
            self.addCleanup(tools._environment_config.cache_clear)
            raw = tools.search_geo_knowledge_base('Laundry', knowledge_name='L1')
            payload = json.loads(raw)
            self.assertTrue(payload['success'])
            self.assertEqual(payload['documents'][0]['source_type'], 'clean')
            # Exercise the existing agent projection without modifying its file.
            module_path = ROOT / 'agent_infer_multi_tool.py'
            spec = importlib.util.spec_from_file_location('hae_agent_projection_test', module_path)
            agent = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(agent)
            visible = json.loads(agent.project_tool_result_for_model('search_geo_knowledge_base', raw))
            for doc in visible['documents']:
                self.assertEqual(set(doc), {'title', 'url', 'timestamp'})


if __name__ == '__main__':
    unittest.main()
