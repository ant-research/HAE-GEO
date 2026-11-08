#!/usr/bin/env bash
set -euo pipefail
cd /opt/xunsearch
mkdir -p tmp data

stop_engine() {
  bin/xs-ctl.sh -b inet stop || true
}
trap stop_engine EXIT
trap 'exit 0' INT TERM

bin/xs-ctl.sh -b inet start
while true; do
  for port in 8383 8384; do
    if [[ ! -f "tmp/pid.${port}" ]] || ! kill -0 "$(<"tmp/pid.${port}")" 2>/dev/null; then
      printf 'Xunsearch service %s exited\n' "$port" >&2
      exit 1
    fi
  done
  sleep 2 &
  wait $!
done
