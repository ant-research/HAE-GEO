"""Offline integration checks for the translated public sample data.

Run with ``python -m unittest discover -s tests -p test_english_release.py -v``.
Generator/parser wording is intentionally outside this suite's scope.
"""

import importlib
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "data" / "examples"
HAN = re.compile("[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U000323af]")
FAKE_BRANDS = {
    "laundry detergent": "Example CleanWear",
    "children's shoes": "Example LittleSteps",
    "whitening toothpaste": "Example BrightWhite",
    "liver supplements": "Example ClearBalance",
    "sunscreen": "Example SunGuard",
    "power banks": "Example PowerCell",
    "travel agencies": "Example FarTravel",
    "infant and toddler complementary foods": "Example FirstTaste",
}
ENVIRONMENTS = {"clean": "clean", "L1": "l1", "L2": "l2", "L3": "l3"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def corpus(name):
    path = EXAMPLES / "corpora" / (ENVIRONMENTS[name] + ".jsonl")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


class OfflineTestCase(unittest.TestCase):
    def setUp(self):
        super().setUp()
        for target in ("socket.socket", "socket.create_connection", "socket.getaddrinfo"):
            blocker = patch(target, side_effect=AssertionError("Network access forbidden"))
            blocker.start()
            self.addCleanup(blocker.stop)

    def assert_english(self, value):
        if isinstance(value, dict):
            for key, item in value.items():
                self.assert_english(key)
                self.assert_english(item)
        elif isinstance(value, list):
            for item in value:
                self.assert_english(item)
        elif isinstance(value, str):
            self.assertIsNone(HAN.search(value), repr(value))


class EnglishSampleTests(OfflineTestCase):
    def test_all_sample_json_and_jsonl_are_parseable_and_translated(self):
        paths = sorted(EXAMPLES.rglob("*.json")) + sorted(EXAMPLES.rglob("*.jsonl"))
        self.assertGreaterEqual(len(paths), 8)
        for path in paths:
            with self.subTest(path=path.relative_to(ROOT)):
                if path.suffix == ".jsonl":
                    rows = [json.loads(line) for line in path.read_text(
                        encoding="utf-8").splitlines() if line.strip()]
                    self.assertTrue(rows)
                    for row in rows:
                        self.assertIsInstance(row, dict)
                        for key in ("title", "url", "content"):
                            self.assertIsInstance(row[key], str)
                            self.assertTrue(row[key].strip())
                    self.assert_english(rows)
                else:
                    payload = read_json(path)
                    self.assertTrue(payload)
                    self.assert_english(payload)

    def test_category_and_brand_mappings(self):
        for filename in ("fake_brands.json", "real_brands.json", "niche_brands.json"):
            with self.subTest(filename=filename):
                brands = read_json(EXAMPLES / "brands" / filename)
                self.assertEqual(set(brands), set(FAKE_BRANDS))
                for names in brands.values():
                    self.assertIsInstance(names, list)
                    self.assertTrue(names)
                    self.assertTrue(all(isinstance(name, str) and name for name in names))
        self.assertEqual(read_json(EXAMPLES / "brands" / "fake_brands.json"),
                         {category: [brand] for category, brand in FAKE_BRANDS.items()})

    def test_query_ids_language_and_target_brands(self):
        queries = read_json(EXAMPLES / "queries.json")
        self.assertEqual([q["query_id"] for q in queries],
                         ["laundry_compare_demo_1", "laundry_verify_demo_1"])
        real = read_json(EXAMPLES / "brands" / "real_brands.json")
        fake = read_json(EXAMPLES / "brands" / "fake_brands.json")
        for query in queries:
            with self.subTest(query_id=query["query_id"]):
                self.assertEqual(query["language"], "en")
                self.assertTrue(query["query_id"].isascii())
                self.assertIn(query["category"], FAKE_BRANDS)
                for field, mapping in (("fake_brands", fake), ("real_brands_in_query", real)):
                    for brand in query.get(field, []):
                        self.assertIn(brand, mapping[query["category"]])
                        self.assertIn(brand, query["user_query"])

    def test_corpora_retain_english_target_and_example_urls(self):
        for environment in ENVIRONMENTS:
            with self.subTest(environment=environment):
                rows = corpus(environment)
                for row in rows:
                    self.assertRegex(row["url"], r"^https://(?:clean|regulator|poison)\.example/")
                if environment != "clean":
                    poisoned = [row for row in rows if row["dataset"] == "poison"]
                    self.assertTrue(poisoned)
                    for row in poisoned:
                        self.assertEqual(row["level"], environment)
                        self.assertIn(FAKE_BRANDS["laundry detergent"], row["content"])
                else:
                    self.assertTrue(any("laundry detergent" in row["content"] for row in rows))


class EnglishRetrievalTests(OfflineTestCase):
    def setUp(self):
        super().setUp()
        path_patch = patch.object(sys, "path", [str(ROOT / "GEO_eval"), *sys.path])
        path_patch.start()
        self.addCleanup(path_patch.stop)
        try:
            self.tools = importlib.import_module("geo_qwen_tools")
        except ImportError as exc:
            self.skipTest("Local retrieval import unavailable: " + str(exc))
        temporary = tempfile.TemporaryDirectory(prefix="english-release-")
        self.addCleanup(temporary.cleanup)
        cache_patch = patch.object(self.tools, "CACHE_ROOT", Path(temporary.name))
        cache_patch.start()
        self.addCleanup(cache_patch.stop)
        config = {name: {"corpus": str(EXAMPLES / "corpora" / (stem + ".jsonl"))}
                  for name, stem in ENVIRONMENTS.items()}
        config_patch = patch.object(self.tools, "_environment_config", return_value=config)
        config_patch.start()
        self.addCleanup(config_patch.stop)
        self.tools._load_corpus.cache_clear()
        self.addCleanup(self.tools._load_corpus.cache_clear)

    def test_english_search_and_scrape_round_trip(self):
        for environment in ENVIRONMENTS:
            with self.subTest(environment=environment):
                query = "laundry detergent" if environment == "clean" else "Example CleanWear"
                result = json.loads(self.tools.search_geo_knowledge_base(
                    query, knowledge_name=environment, top_k=1))
                self.assertTrue(result["success"])
                self.assertEqual(result["query"], query)
                self.assertEqual(len(result["documents"]), 1)
                hit = result["documents"][0]
                self.assertGreater(hit["score"], 0)
                self.assertIn(query.lower(), (hit["title"] + " " + hit["snippet"]).lower())
                scraped = json.loads(self.tools.scrape_geo_webpage(
                    hit["url"], knowledge_name=environment))
                expected = next(row for row in corpus(environment) if row["url"] == hit["url"])
                self.assertTrue(scraped["success"])
                self.assertFalse(scraped["truncated"])
                self.assertEqual(scraped["content"], expected["content"])
                self.assertEqual(scraped["title"], expected["title"])
                self.assert_english(result)
                self.assert_english(scraped)

    def test_scrape_requires_search_in_same_environment(self):
        url = "https://poison.example/l1/direct"
        self.assertFalse(json.loads(self.tools.scrape_geo_webpage(
            url, knowledge_name="L1"))["success"])
        self.tools.search_geo_knowledge_base("Example CleanWear", knowledge_name="L1")
        self.assertTrue(json.loads(self.tools.scrape_geo_webpage(
            url, knowledge_name="L1"))["success"])
        self.assertFalse(json.loads(self.tools.scrape_geo_webpage(
            url, knowledge_name="L2"))["success"])

    def test_full_context_model_visible_projection(self):
        try:
            full = importlib.import_module("geo_static_full_context_tool")
        except ImportError as exc:
            self.skipTest("Full-context import unavailable: " + str(exc))
        visible_text, raw_text = full.search_geo_full_content(
            "Example CleanWear", knowledge_name="L3", top_k=2)
        visible, raw = json.loads(visible_text), json.loads(raw_text)
        self.assertTrue(visible["success"])
        self.assertTrue(raw["success"])
        self.assertNotIn("knowledge_name", visible)
        self.assertEqual(raw["knowledge_name"], "L3")
        self.assertEqual(len(visible["documents"]), 2)
        self.assertEqual(len(raw["documents"]), 2)
        self.assertIn("Example CleanWear", visible["documents"][0]["content"])
        for public, private in zip(visible["documents"], raw["documents"]):
            self.assertIn("source_type", private)
            for hidden in ("source_type", "dataset", "level", "knowledge_name"):
                self.assertNotIn(hidden, public)
            self.assertEqual(public, {key: value for key, value in private.items()
                                      if key != "source_type"})
            expected = next(row for row in corpus("L3") if row["url"] == public["url"])
            self.assertEqual(public["content"], expected["content"])
            self.assertFalse(public["content_truncated"])
        self.assert_english(visible)


if __name__ == "__main__":
    unittest.main()
