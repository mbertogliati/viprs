#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
OUT_DIR=${VIPRS_FIXTURES_OUT_DIR:-"$ROOT_DIR/.artifacts/fixtures-release"}
VERSION=${VIPRS_FIXTURES_VERSION:-fixtures-v2}
ASSET=${1:-all}

mkdir -p "$OUT_DIR"

tar_create() {
    output=$1
    label=$2
    input_label=$3
    shift
    shift
    shift

    start=$(date +%s)
    echo "fixtures: $label: compressing $input_label into $(basename "$output")"
    (
        while :; do
            sleep 10
            now=$(date +%s)
            elapsed=$((now - start))
            echo "fixtures: $label: still compressing (${elapsed}s elapsed)"
        done
    ) &
    progress_pid=$!

    # COPYFILE_DISABLE and --no-xattrs keep macOS AppleDouble files and
    # com.apple.* extended attributes out of archives consumed by GNU tar in CI.
    if COPYFILE_DISABLE=1 tar --no-xattrs -C "$ROOT_DIR" -cJf "$output" "$@"; then
        kill "$progress_pid" 2>/dev/null || true
        wait "$progress_pid" 2>/dev/null || true
        now=$(date +%s)
        elapsed=$((now - start))
        echo "fixtures: $label: done (${elapsed}s)"
    else
        status=$?
        kill "$progress_pid" 2>/dev/null || true
        wait "$progress_pid" 2>/dev/null || true
        echo "fixtures: $label: failed" >&2
        exit "$status"
    fi
}

sha256_file() {
    file=$1
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$file" | awk '{print $1}'
    else
        shasum -a 256 "$file" | awk '{print $1}'
    fi
}

functional="$OUT_DIR/viprs-fixtures-functional-$VERSION.tar.xz"
bench="$OUT_DIR/viprs-fixtures-bench-$VERSION.tar.xz"
archives=""

case "$ASSET" in
    all|functional|bench) ;;
    *)
        echo "usage: $0 [all|functional|bench]" >&2
        exit 2
        ;;
esac

if [ "$ASSET" = "all" ] || [ "$ASSET" = "functional" ]; then
    tar_create "$functional" "functional" "tests/fixtures without bench_*" \
        --exclude='tests/fixtures/images/bench_*' \
        --exclude='tests/fixtures/.golden-script-work' \
        --exclude='tests/fixtures/.fixtures-*' \
        --exclude='tests/fixtures/fixtures.lock' \
        --exclude='tests/fixtures/images/._*' \
        tests/fixtures
    archives="$archives $functional"
fi

if [ "$ASSET" = "all" ] || [ "$ASSET" = "bench" ]; then
    tar_create "$bench" "bench" "tests/fixtures/images/bench_*" \
        --exclude='tests/fixtures/images/._*' \
        tests/fixtures/images/bench_*
    archives="$archives $bench"
fi

for archive in $archives; do
    size=$(wc -c < "$archive" | tr -d ' ')
    sha256=$(sha256_file "$archive")
    printf '%s  %s  %s bytes\n' "$sha256" "$archive" "$size"
done

echo "fixtures: wrote archives to $OUT_DIR"
