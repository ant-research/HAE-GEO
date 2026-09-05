# HEO-Bench

This is the sanitized research release accompanying **Evaluating Deep-Search
Agents under Hierarchical Web Evidence Poisoning**. It contains the paper
source, controlled L1/L2/L3 page-construction pipeline, Search--Scrape ReAct
runtime, deterministic trajectory metrics, and the six-dimensional split Judge.

Production credentials, internal service endpoints, private corpora, raw
trajectories, and cloud-storage configuration are deliberately excluded.

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
- `iclr2026/`: manuscript source and released figures.

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
  --categories 洗衣液 \
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

## Compile the paper

```bash
cd iclr2026
latexmk -pdf -interaction=nonstopmode -halt-on-error mian.tex
```

## Release checklist

```bash
python3 -m unittest discover -s 'rule&rubric' -p 'test_*.py'
bash scripts/scan_secrets.sh
```

See `SECURITY.md` for the controlled-use and data-release boundary.
