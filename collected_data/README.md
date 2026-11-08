# Clean webpage collection

This directory provides a portable URL-to-corpus pipeline. It accepts an
existing URL pool, deduplicates URLs, downloads pages with Axios, extracts
visible text, records retrieval metadata and SHA256 hashes, and builds local
Clean/L1/L2/L3 environments. It uses no internal search service, company
endpoint, private cache, or cloud upload.

URL discovery and benchmark retrieval are separate steps. This collector reads
URLs that you supply or that appear in the released clean-page metadata. Once
you build a corpus, `GEO_eval/geo_qwen_tools.py` searches within that corpus and
returns ranked results. The current local backend uses a simple lexical score;
the environment files can also support an external open-source retrieval
backend. This directory does not replace that score with BM25 or dense retrieval.

## Install

Use Node.js 20.19 or newer and Python 3.10 or newer:

```bash
cd collected_data
npm ci
cd ..
```

## Run the pipeline

From the repository root:

```bash
python3 collected_data/corpus_tools.py manifest
LIMIT=20 bash collected_data/run_pipeline.sh
```

The pipeline attempts at most `LIMIT` previously unattempted URLs per run;
the default is 50. Repeating the command resumes collection. It preserves
local outputs and advances past previously rejected URLs. Set
`RETRY_FAILED=1` to retry earlier failures, or pass `--retry-failed` to the
direct Axios collector. Previously accepted URLs always remain skipped.

To use your own JSONL URL pool:

```json
{"url":"https://example.com/product-review","title":"Product review","category":"laundry detergent","discovered_via":"independent product comparison"}
```

```bash
LIMIT=100 DELAY_MS=500 bash collected_data/run_pipeline.sh /path/to/urls.jsonl
```

Input can be a JSON array, a JSONL file with `url` records, or a text file with
one HTTP(S) URL per line. URL fragments do not create duplicate documents;
query parameters and trailing slashes remain significant. The collector
merges category metadata for repeated URLs.

## Direct Axios collection

```bash
node collected_data/collect_clean_corpus.mjs \
  --input collected_data/manifests/clean_urls.jsonl \
  --output collected_data/output/clean.jsonl \
  --manifest collected_data/output/crawl_manifest.jsonl \
  --limit 20 --delay-ms 500 --timeout-ms 15000 \
  --min-content-chars 100 --max-content-chars 200000
```

Axios sends HTTP GET requests to the supplied page URLs. Cheerio extracts the
title, publication metadata, and visible article/main/body text. The collector
does not run page JavaScript. It accepts HTML pages with sufficient text and
logs blocked URLs, HTTP errors, unsupported formats, and short pages in the
crawl manifest. It checks robots.txt, including redirect destinations, and
does not attempt to bypass access controls. Use sources for which collection
and reuse are permitted; robots.txt is not a redistribution license.

Accepted records contain `url`, `title`, `timestamp`, `content`, `category`,
`categories`, `discovered_via`, `source_type: "clean"`, `level: "clean"`,
`retrieved_at`, `final_url`, `content_sha256`, `response_sha256`, and truncation
metadata. `content_sha256` hashes the stored UTF-8 text; `response_sha256`
hashes the downloaded response bytes after HTTP decompression. The label
`clean` denotes the background-corpus role, not a factual-accuracy judgment.

## Build matched environments

```bash
python3 collected_data/corpus_tools.py build \
  --clean collected_data/output/clean.jsonl \
  --poison benchmark_data/geo_attack_pages.json \
  --output-dir collected_data/output/environments

export GEO_ENVIRONMENTS_CONFIG="$PWD/collected_data/output/environments/environments.json"
```

The builder writes `clean.jsonl`, `l1.jsonl`, `l2.jsonl`, and `l3.jsonl`, a
configuration compatible with the local Search/Scrape runtime, and
`corpus_manifest.json` with counts and file hashes. All four environments
share exactly the same collected clean background. Each attack environment
adds only its corresponding poison level and retains hidden attack provenance.
The released poison corpus contains distinct snapshots with repeated URLs.
For those snapshots, the builder adds a deterministic `#snapshot=...` fragment
to the local tool URL, keeps `original_url`, and records the mapping in the
corpus manifest. This preserves all 770 pages per level in URL-keyed caches;
the source dataset remains unchanged. These are local snapshot identifiers,
not new public webpage addresses. The builder rejects clean/poison URL
collisions and duplicate clean URLs. The generated configuration uses
absolute paths; regenerate it after moving the files to another machine.

## URL manifest coverage

`manifests/clean_urls.jsonl` lists the unique URLs in the released
`benchmark_data/clean_web_pages_subset.json`. The companion
`clean_urls.meta.json` records the source and manifest hashes and coverage.
This is a manifest of the released subset, not the full original clean corpus.
The released metadata has no full-page body hashes or original collection
times. The exporter does not invent them. Fresh collections add their own
retrieval times and hashes through `crawl_manifest.jsonl`.

The code supports collecting a new background corpus and rerunning the
workflow. Page edits, removals, extraction differences, corpus coverage, and
retrieval changes can alter exact benchmark scores. A URL list alone does not
recover historical page bodies or establish that a result trend persists.

## Evaluator and rule vocabulary

The repository includes the deterministic parser and named rule vocabularies
in `rule&rubric/geo_evaluate.py`. Its current identifiers are:

- Rule version: `trajectory-state-machine-v3-english-strict-recovery`.
- Judge prompt version: `split-poison-quality-six-metric-v7-english-full-search-content-no-score-normalization`.

These identify the public English-adapted implementation. This release does
not establish that it is the frozen implementation used for the paper's
archived results. Do not label this copy as the paper's exact frozen parser
without validating that correspondence against the original configuration
and recorded trajectories.

```bash
python3 collected_data/corpus_tools.py evaluator-inventory
```

`manifests/evaluator_inventory.json` records the file SHA256, version strings,
and named rule vocabularies. Full parser behavior also depends on its matching
logic and regex rules. The hash identifies this public file, not an independently
verified association with the paper's numerical results. This collection
pipeline does not modify the evaluator, prompts, or existing benchmark data.

## Validation

```bash
cd collected_data
npm test
python3 -m unittest discover -s tests -p 'test_*.py'
```

Tests use a local HTTP fixture and temporary corpora; they do not collect
third-party pages or call agent/Judge APIs.
