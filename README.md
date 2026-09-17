# HEO-Bench

This is the sanitized research release accompanying **Evaluating Deep-Search
Agents under Hierarchical Web Evidence Poisoning**. It contains released benchmark
data, the controlled L1/L2/L3 page-construction pipeline, Search--Scrape ReAct
runtime, deterministic trajectory metrics, and the six-dimensional split Judge.

Production credentials, internal service endpoints, private corpora, raw
trajectories, and cloud-storage configuration are deliberately excluded.

## Language

This release uses English agent and Judge prompts, page-generation templates,
example data, and rule-matching vocabulary. Legacy ASCII schema keys and query
identifiers are retained for compatibility. The English prompts and rules are
a language adaptation, not a guarantee of equivalence to earlier Chinese
evaluations. Rerun and validate evaluations when changing language; do not
reuse archived scores as results of this English configuration.
Supply English queries and reference corpora for English-only runs. Imported
reference-page content is preserved by the template-only modified path; this
release does not automatically translate external datasets.
The benchmark datasets in `benchmark_data/` retain their original source language;
the English migration covers code, prompts, documentation, and `data/examples/`.

## Repository map

- `generate_attack_pages/brandgen/`: deterministic and LLM-assisted L1/L2/L3
  construction, including generated and modified paths.
- `GEO_eval/geo_qwen_tools.py`: runnable local JSONL Search/Scrape backend.
- `GEO_eval/backend_adapter_pseudocode.py`: interface sketch for an external
  hybrid retrieval system.
- `agent_infer_multi_tool.py`: Base/Defense prompts and multi-turn ReAct loop.
- `run_geo_eval.py`: resumable trajectory generation, one JSON file per query.
- `rule&rubric/geo_evaluate.py`: E-A-V-R-Y rules and split six-metric Judge.
- `data/examples/`: synthetic query, brand, and environment examples only.
- `benchmark_data/`: benchmark queries, poisoned pages, and a partial clean-page release.

## Benchmark data

The files in `benchmark_data/` cover eight consumer-product categories and
retain their original text and annotations. They are distinct from the small
English demonstration fixtures in `data/examples/`.

| File | Records | Description |
| --- | ---: | --- |
| `queries.json` | 1,011 | The query pool, including query IDs, categories, task types, user questions, and target fake-brand annotations. This is the full pool, not just the 240-query evaluation set. |
| `geo_attack_pages.json` | 2,310 | Poisoned pages for 154 fabricated target brands: 770 pages at each of L1, L2, and L3. Records include page text, construction provenance, attack-vector annotations, and `poisoning_level`. |
| `clean_web_pages_subset.json` | 54,358 | The publicly released subset of clean-page records, containing category, title, URL, and snippet metadata. These records do not include a separate full-page `content` field. |

### Clean-page release scope

Only part of the clean Web corpus is publicly released. Many source sites
prohibit automated crawling or otherwise restrict collection and reuse. Pages
subject to those restrictions are excluded from the public release to reduce
redistribution and compliance risks; the released subset should not be treated
as a complete copy of the clean corpus used in the experiments.

Public accessibility does not by itself grant permission to collect or
redistribute a page. The release does not grant additional rights to underlying
third-party content. Any extension or redistribution of the corpus should
exclude restricted pages unless the necessary permission has been obtained.
Results obtained using only the released subset may differ from those obtained
with the original full retrieval environment.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
cp configs/environments.example.json configs/environments.json
```

Export the values from `.env` in your shell. Never commit `.env`.

## Generate controlled L1/L2/L3 pages

The template-only path needs no model credential:

```bash
PYTHONPATH=generate_attack_pages python -m brandgen generate \
  --categories "laundry detergent" \
  --brand-source fake \
  --pages-per-brand 2 \
  --level L1,L2,L3 \
  --output-dir generate_attack_pages/output
```

To use an OpenAI-compatible generator, set `GENERATOR_BASE_URL`,
`GENERATOR_API_KEY`, and `GENERATOR_MODEL`, then pass
`--llm-provider openai-compatible`. The modified path reads only from the
licensed local file named by `GEO_REFERENCE_CORPUS`; no public page is modified
or published by this code.

## Run Search--Scrape trajectories

```bash
export MODEL_API_URL=https://your-provider.example/v1/chat/completions
export MODEL_API_KEY=...
export MODEL_NAME=your-tool-capable-model
bash scripts/run_local_demo.sh
```

Search exposes title/URL/timestamp metadata. Scrape exposes page content. Hidden
fields such as `source_type` remain only in `raw_result` for offline attribution
and are removed from the model-visible projection.

Completed trajectories with a non-empty final answer are reused on rerun. A
model that exhausts the tool budget receives the forced-finalization prompt.

## Judge existing trajectories

```bash
export JUDGE_API_URL=https://your-judge.example/v1/chat/completions
export JUDGE_API_KEY=...
export JUDGE_MODEL=your-fixed-judge-model
bash scripts/judge_local_demo.sh
```

The attack-aware Judge scores M1--M3; the attack-label-blind Judge scores M4--M6.
Scores are retained exactly as returned in `{0,1,2}` with no normalization.
Checkpoint compatibility includes trajectory hashes and Judge configuration.

## Release checklist

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
python3 -m unittest discover -s 'rule&rubric' -p 'test_*.py'
bash scripts/scan_secrets.sh
```

See `SECURITY.md` for the controlled-use and data-release boundary.
