#!/usr/bin/env python3
"""Exercise a live Xunsearch service through the public Search--Scrape tools."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--query', default='laundry detergent')
    args = parser.parse_args()
    os.environ['GEO_ENVIRONMENTS_CONFIG'] = str(args.config.resolve())
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    with tempfile.TemporaryDirectory(prefix='hae-xunsearch-smoke-') as cache:
        os.environ['GEO_TOOL_CACHE_DIR'] = cache
        from GEO_eval.geo_qwen_tools import search_geo_knowledge_base, scrape_geo_webpage
        config = json.loads(args.config.read_text(encoding='utf-8'))['environments']
        matched = 0
        for name, entry in config.items():
            if entry.get('backend') != 'xunsearch':
                raise ValueError(f'{name} does not select the Xunsearch backend')
            result = json.loads(search_geo_knowledge_base(args.query, knowledge_name=name))
            if not result['success'] or len(result['documents']) > 10:
                raise RuntimeError(f'{name}: invalid Search response')
            for row in result['documents']:
                page = json.loads(scrape_geo_webpage(row['url'], knowledge_name=name))
                if not page.get('success') or not page.get('content'):
                    raise RuntimeError(f'{name}: Search/Scrape document mismatch')
                matched += 1
            print(json.dumps(dict(environment=name, matches=len(result['documents']))))
        if not matched:
            raise RuntimeError('No matches in any environment; choose a query present in the corpus')
    print('Live Xunsearch Search--Scrape smoke test passed')


if __name__ == '__main__':
    main()
