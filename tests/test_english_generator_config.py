"""Offline regression tests for English generator configuration and queries."""

import random
import shlex
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


GENERATOR_ROOT = Path(__file__).resolve().parents[1] / "generate_attack_pages"
sys.path.insert(0, str(GENERATOR_ROOT))

from brandgen.cli import build_parser, config_from_args
from brandgen.llm import LLMError
from brandgen.models import LLMStats
from brandgen.query_types import QUERY_TYPES, llm_query, sample_example
from brandgen.utils import read_categories


CATEGORIES = [
    "laundry detergent",
    "sunscreen",
    "power banks",
    "children's shoes",
    "liver supplements",
    "whitening toothpaste",
    "infant and toddler complementary foods",
    "travel agencies",
]


class EnglishGeneratorConfigTests(unittest.TestCase):
    def make_config(self, *options):
        args = build_parser().parse_args(
            ["generate", "--categories", CATEGORIES[0], *options]
        )
        return config_from_args(args)

    def test_quoted_categories_preserve_spaces_and_apostrophes(self):
        for category in CATEGORIES:
            with self.subTest(category=category):
                argv = shlex.split("generate --categories " + shlex.quote(category))
                config = config_from_args(build_parser().parse_args(argv))
                self.assertEqual(config.categories, [category])

    def test_comma_separated_categories(self):
        config = self.make_config("--categories", " , " + ", ".join(CATEGORIES) + ", ")
        self.assertEqual(config.categories, CATEGORIES)

    def test_default_category_file_preserves_multiword_names(self):
        with patch.object(Path, "read_text", return_value="\n".join(CATEGORIES)) as read:
            config = config_from_args(build_parser().parse_args(["generate"]))
        self.assertEqual(config.categories, CATEGORIES)
        read.assert_called_once_with(encoding="utf-8")

    def test_empty_category_list_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no categories found"):
            read_categories(Path("unused"), " , \n ")

    def test_quoted_brand_preserves_spaces(self):
        argv = shlex.split(
            'generate --categories "laundry detergent" --brands "Example CleanWear"'
        )
        config = config_from_args(build_parser().parse_args(argv))
        self.assertEqual(config.brands, ["Example CleanWear"])
        self.assertFalse(config.brands_all)

    def test_comma_separated_brands_trim_only_surrounding_whitespace(self):
        config = self.make_config(
            "--brands", " Example CleanWear, , Children's Choice, Example  Double Space, "
        )
        self.assertEqual(
            config.brands,
            ["Example CleanWear", "Children's Choice", "Example  Double Space"],
        )

    def test_all_brands_default_and_sentinels(self):
        self.assertIsNone(self.make_config().brands)
        for value in ("all", " ALL ", "", "   "):
            with self.subTest(value=value):
                config = self.make_config("--brands", value)
                self.assertIsNone(config.brands)
                self.assertTrue(config.brands_all)

    def test_level_splitting_still_accepts_whitespace_and_commas(self):
        for value in ("L1 L3", "L1,L3", " l1, l3 "):
            with self.subTest(value=value):
                self.assertEqual(self.make_config("--level", value).levels, ("L1", "L3"))
        self.assertEqual(self.make_config().levels, ("L1", "L2", "L3"))


class EnglishQueryTests(unittest.TestCase):
    def test_every_template_generates_english_with_intact_names(self):
        brand = "Example CleanWear"
        for category in CATEGORIES:
            for query_type in QUERY_TYPES:
                for template in query_type.templates:
                    with self.subTest(category=category, type=query_type.id, template=template):
                        rng = Mock(spec=random.Random)
                        rng.choice.return_value = template
                        query = sample_example(rng, query_type, category, brand)
                        self.assertEqual(query, template.format(category=category, brand=brand))
                        self.assertTrue(query.isascii())
                        self.assertNotIn("{", query)
                        if "{category}" in template:
                            self.assertIn(category, query)
                        if "{brand}" in template:
                            self.assertIn(brand, query)
                        rng.choice.assert_called_once_with(query_type.templates)

    def test_llm_prompt_requests_english_without_live_client(self):
        expected = "Is Example CleanWear laundry detergent worth buying?"
        client = object()
        stats = LLMStats()
        with patch("brandgen.query_types.call_json", return_value={"query": " " + expected + " "}) as call:
            query = llm_query(QUERY_TYPES[1], CATEGORIES[0], "Example CleanWear", client, stats)
        self.assertEqual(query, expected)
        call.assert_called_once()
        self.assertIs(call.call_args.args[0], client)
        messages = call.call_args.args[1]
        self.assertIn("in English", messages[0]["content"])
        self.assertIn("laundry detergent", messages[1]["content"])
        self.assertIn("Example CleanWear", messages[1]["content"])
        self.assertTrue(all(message["content"].isascii() for message in messages))
        self.assertIs(call.call_args.kwargs["stats"], stats)

    def test_no_client_returns_none_without_llm_call(self):
        with patch("brandgen.query_types.call_json") as call:
            result = llm_query(QUERY_TYPES[0], CATEGORIES[0], "Example CleanWear", None, LLMStats())
        self.assertIsNone(result)
        call.assert_not_called()

    def test_llm_failure_allows_template_fallback(self):
        with patch("brandgen.query_types.call_json", side_effect=LLMError("offline")):
            result = llm_query(QUERY_TYPES[0], CATEGORIES[0], "Example CleanWear", object(), LLMStats())
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
