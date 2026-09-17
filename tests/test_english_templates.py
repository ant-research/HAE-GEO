"""Render every eligible category/level/page-role combination without an API."""

import contextlib
import io
import json
import os
from pathlib import Path
import re
import sys
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "generate_attack_pages"))

from brandgen.depth import LEVEL_SPECS
from brandgen.orchestrator import fallback_profile
from brandgen.page_type import PAGE_BUILDERS, PageContext, build_offline_signals


class EnglishTemplateTests(unittest.TestCase):
    def test_cli_generates_all_categories_and_levels_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, PYTHONPATH=str(ROOT / "generate_attack_pages"),
                       GEO_REFERENCE_CORPUS="")
            result = subprocess.run(
                [sys.executable, "-m", "brandgen", "generate", "--llm-provider", "none",
                 "--pages-per-brand", "1", "--level", "L1,L2,L3", "--output-dir", directory],
                cwd=ROOT, env=env, capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stderr[-2000:])
            records = []
            for path in Path(directory).rglob("*.json"):
                records.extend(json.loads(path.read_text()))
            self.assertEqual(len(records), 24)
            self.assertEqual(len({row["category"] for row in records}), 8)
            self.assertEqual({row["level"] for row in records}, {"L1", "L2", "L3"})
            han = re.compile("[" + chr(0x3400) + "-" + chr(0x4DBF)
                             + chr(0x4E00) + "-" + chr(0x9FFF) + "]")
            self.assertIsNone(han.search(json.dumps(records, ensure_ascii=False)))

    def test_all_eligible_templates_render_in_english(self):
        brands = json.loads((ROOT / "data/examples/brands/fake_brands.json").read_text())
        han = re.compile("[" + chr(0x3400) + "-" + chr(0x4DBF)
                         + chr(0x4E00) + "-" + chr(0x9FFF) + "]")
        for category in brands:
            for level, spec in LEVEL_SPECS.items():
                for role in spec.carriers:
                    for direction in ("positive", "negative"):
                        with self.subTest(category=category, level=level, role=role, direction=direction):
                            profile = fallback_profile(category, "Example TestBrand", "fake", direction, "test")
                            ctx = PageContext(
                                profile=profile, page_type=role, level=level,
                                manipulation_direction=direction,
                                profile_mode={"L1": "none", "L2": "base", "L3": "professional"}[level],
                                core_trait=spec.core_trait,
                                credibility_profile=spec.credibility_profile,
                                attacker_capability=spec.attacker_capability,
                                timestamp="2026-07-28", seed="test",
                            )
                            if level == "L3":
                                ctx.professional_signals = build_offline_signals(ctx)
                            builder = PAGE_BUILDERS[role]
                            title, body = builder.build_tmpl(ctx)
                            self.assertTrue(title.strip())
                            self.assertTrue(body.strip())
                            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                                prompt = builder.build_prompt(ctx, [])
                            rendered = json.dumps(
                                [title, body, prompt, profile.to_dict()],
                                ensure_ascii=False,
                            )
                            self.assertIsNone(han.search(rendered))


if __name__ == "__main__":
    unittest.main()
