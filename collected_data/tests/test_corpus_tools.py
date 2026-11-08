import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1] / 'corpus_tools.py'
SPEC = importlib.util.spec_from_file_location('corpus_tools', SOURCE)
tools = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tools)


class CorpusToolsTest(unittest.TestCase):
    def test_url_manifest_merges_categories_without_inventing_historical_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / 'source.json', root / 'manifest.jsonl'
            tools.write_json(source, [
                dict(url='https://example.com/a#one', title='Review', category='A'),
                dict(url='https://example.com/a#two', title='Review', category='B'),
            ])
            meta = tools.export_manifest(source, output)
            rows = tools.records(output)
            self.assertEqual(meta['source_records'], 2)
            self.assertEqual(meta['unique_urls'], 1)
            self.assertEqual(rows[0]['categories'], ['A', 'B'])
            self.assertNotIn('content_sha256', rows[0])
            self.assertFalse(meta['historical_body_hashes_available'])
            self.assertEqual(meta['manifest_sha256'], tools.digest(output))

    def test_builder_keeps_background_and_attack_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            clean, poison = root / 'clean.jsonl', root / 'poison.json'
            tools.write_jsonl(clean, [dict(url='https://clean.example/a', content='Clean body')])
            tools.write_json(poison, [dict(url=f'https://attack.example/{level}',
                                          content='Attack body', poisoning_level=level,
                                          source_type='generated') for level in ('l1', 'l2', 'l3')])
            stats = tools.build_environments(clean, poison, root / 'env')
            self.assertEqual(stats['clean']['records'], 1)
            clean_row = tools.records(root / 'env/clean.jsonl')[0]
            for level in ('l1', 'l2', 'l3'):
                docs = tools.records(root / f'env/{level}.jsonl')
                self.assertEqual(docs[0], clean_row)
                self.assertEqual(docs[1]['source_type'], 'generated')
                self.assertEqual(stats[level]['records'], 2)
            tools.write_jsonl(clean, [dict(url='https://attack.example/l1', content='Collision')])
            with self.assertRaisesRegex(ValueError, 'URL collision'):
                tools.build_environments(clean, poison, root / 'invalid')

    def test_builder_rejects_metadata_without_page_bodies(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            clean = root / 'clean.json'
            tools.write_json(clean, [dict(url='https://example.com/a', snippet='Only metadata')])
            with self.assertRaisesRegex(ValueError, 'full-page content'):
                tools.build_environments(clean, root / 'missing.json', root / 'env')

    def test_repeated_attack_urls_keep_distinct_snapshots(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            clean, poison = root / 'clean.jsonl', root / 'poison.json'
            tools.write_jsonl(clean, [dict(url='https://clean.example/a', content='Clean')])
            pages = [dict(url=f'https://attack.example/{level}', content='Attack',
                          poisoning_level=level, source_type='generated') for level in ('l1', 'l2', 'l3')]
            pages.append({**pages[0], 'content': 'A different snapshot'})
            tools.write_json(poison, pages)
            tools.build_environments(clean, poison, root / 'env')
            docs = tools.records(root / 'env/l1.jsonl')
            self.assertEqual(len(docs), 3)
            self.assertNotEqual(docs[1]['url'], docs[2]['url'])
            self.assertEqual(docs[1]['original_url'], docs[2]['original_url'])
            manifest = json.loads((root / 'env/corpus_manifest.json').read_text())
            self.assertEqual(len(manifest['local_snapshot_aliases']), 2)

    def test_evaluator_inventory_identifies_public_adapter_without_frozen_claim(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'inventory.json'
            source = tools.ROOT / 'rule&rubric/geo_evaluate.py'
            tools.evaluator_inventory(source, output)
            data = json.loads(output.read_text())
            self.assertEqual(data['evaluator_sha256'], tools.digest(source))
            self.assertFalse(data['paper_frozen_equivalence_verified'])
            self.assertEqual(data['rule_metric_version'], 'trajectory-state-machine-v3-english-strict-recovery')
            self.assertIn('official website', data['named_rule_vocabulary']['VERIFICATION_KEYWORDS'])
            self.assertIn('recommend', data['named_rule_vocabulary']['POSITIVE_RECOMMEND_KEYWORDS'])


if __name__ == '__main__':
    unittest.main()
