"""Offline checks for the English runtime prompts and release text."""

import ast
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
HAN = re.compile("[" + chr(0x3400) + "-" + chr(0x4DBF)
                 + chr(0x4E00) + "-" + chr(0x9FFF) + "]")


class EnglishPromptTests(unittest.TestCase):
    def test_runtime_prompts_are_english(self):
        for filename in ("agent_infer_multi_tool.py", "agent_infer_static_full_context.py"):
            namespace = {}
            tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id.endswith("PROMPT")
                    for target in node.targets
                ):
                    exec(compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"), namespace)
            for name, value in namespace.items():
                if name.endswith("PROMPT"):
                    self.assertIsNone(HAN.search(value), (filename, name))
            self.assertIn("English", namespace["BASELINE_SYSTEM_PROMPT"])
            self.assertIn("Evidence safety requirements", namespace["DEFENSE_SYSTEM_PROMPT"])

    def test_source_and_examples_have_no_chinese(self):
        suffixes = {".py", ".json", ".jsonl", ".md", ".txt", ".sh", ".yaml", ".yml"}
        # Full benchmark corpora retain their original evidence and are audited separately.
        excluded = {".git", ".cache", "__pycache__", ".venv", "venv", "benchmark_data"}
        for path in ROOT.rglob("*"):
            name = str(path.relative_to(ROOT))
            if excluded.intersection(path.relative_to(ROOT).parts):
                continue
            if not path.is_file() or path.suffix not in suffixes:
                continue
            with self.subTest(path=name):
                text = path.read_text(encoding="utf-8")
                self.assertIsNone(HAN.search(text), name)


if __name__ == "__main__":
    unittest.main()
