#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v vips >/dev/null 2>&1; then
  echo "error: vips CLI not found in PATH" >&2
  exit 1
fi

if ! command -v vipsheader >/dev/null 2>&1; then
  echo "error: vipsheader CLI not found in PATH" >&2
  exit 1
fi

export VIPRS_REGENERATE_GOLDEN_FIXTURES=1

cargo test --test functional -- --test-threads=1
