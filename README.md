# HEO-Bench

This repository contains released benchmark
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
- `collected_data/`: direct Axios webpage collection, URL manifests, corpus
  hashing, and local Clean/L1/L2/L3 environment construction.
- `deep_researchbehind/`: optional Xunsearch BM25 indexing, local search
  service, and an adapter for the existing Search--Scrape tools.

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

### Collect clean webpage bodies

The released clean records contain metadata and snippets. To collect full-page
text from a supplied URL pool, use the standalone Axios pipeline:

```bash
npm ci --prefix collected_data
python3 collected_data/corpus_tools.py manifest
LIMIT=20 bash collected_data/run_pipeline.sh
export GEO_ENVIRONMENTS_CONFIG="$PWD/collected_data/output/environments/environments.json"
```

The pipeline deduplicates URLs, extracts visible text, records retrieval times
and SHA256 hashes, and builds four environments with a shared clean background.
It uses no internal search endpoint or cloud-storage account. It can also read
your own URL list. See [collected_data/README.md](collected_data/README.md) for
input formats, collection settings, resume behavior, and manifest coverage.

### Evaluator version and reproducibility artifacts

`rule&rubric/geo_evaluate.py` contains the public deterministic trajectory
parser and English rule vocabulary. It identifies itself as
`trajectory-state-machine-v3-english-strict-recovery`. This language-adapted
implementation is not a verified copy of the paper's frozen evaluator.
`collected_data/manifests/evaluator_inventory.json` identifies the current
file by SHA256 and exports its named vocabularies and prompt-version string.

`collected_data/manifests/clean_urls.jsonl` provides a deduplicated URL manifest
for the released clean subset. It does not cover all original clean pages or
contain historical full-page bodies. Newly collected pages have their own
content hashes and collection timestamps. These artifacts support inspection
and new runs; they do not by themselves reproduce the paper's historical
retrieval environment or certify identical scores.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
cp configs/environments.example.json configs/environments.json
```

Export the values from `.env` in your shell. Never commit `.env`.

## Optional Xunsearch retrieval backend

The default local backend uses a simple lexical score. To run indexed BM25
retrieval over the clean and poisoned pages, use
[deep_researchbehind/README.md](deep_researchbehind/README.md). It includes
Docker deployment, corpus import, separate Clean/L1/L2/L3 databases, an HTTP
adapter, and live smoke-test commands. Point `GEO_ENVIRONMENTS_CONFIG` at its
generated configuration to use the same agent/evaluation runners with
Xunsearch. Search returns ranked metadata; Scrape reads the environment-scoped
cached body. This backend does not include dense retrieval or a neural reranker.

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
