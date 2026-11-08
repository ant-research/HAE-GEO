#!/usr/bin/env python3
"""Build Xunsearch databases and publish a matching runtime configuration."""

import argparse
import json
import os
from pathlib import Path

try:
    from .backend import ROOT, build
except ImportError:
    from backend import ROOT, build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--corpus-dir', type=Path,
                        help='Explicitly resolve corpus filenames under this directory, e.g. a container mount')
    parser.add_argument('--php', default='php')
    parser.add_argument('--index-server', default=os.getenv('XUNSEARCH_INDEX_SERVER', '127.0.0.1:8383'))
    parser.add_argument('--search-server', default=os.getenv('XUNSEARCH_SEARCH_SERVER', '127.0.0.1:8384'))
    parser.add_argument('--endpoint', default='http://127.0.0.1:8765')
    args = parser.parse_args()
    manifest = build(args.config, args.output_dir, args.root, args.php,
                     args.index_server, args.search_server, args.endpoint,
                     corpus_dir=args.corpus_dir)
    print(json.dumps(dict(index_revision=manifest['index_revision'],
                          environments=manifest['environments']), indent=2))


if __name__ == '__main__':
    main()
