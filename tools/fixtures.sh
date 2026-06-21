#!/usr/bin/env sh
set -eu

ROOT_DIR=${VIPRS_ROOT_DIR:-$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)}
MANIFEST=${VIPRS_FIXTURES_MANIFEST:-"$ROOT_DIR/tests/fixtures/fixtures.lock"}
CACHE_DIR=${VIPRS_FIXTURES_CACHE:-"$ROOT_DIR/.artifacts/fixtures"}
ASSET=${1:-functional}

if [ ! -f "$MANIFEST" ]; then
    echo "fixtures: manifest not found: $MANIFEST" >&2
    exit 1
fi

manifest_hash() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$MANIFEST" | awk '{print $1}'
    else
        shasum -a 256 "$MANIFEST" | awk '{print $1}'
    fi
}

asset_value() {
    key=$1
    awk -v wanted="$ASSET" -v key="$key" '
        $0 == "[[assets]]" { in_asset = 1; name = ""; next }
        in_asset && /^name = / {
            name = $0
            sub(/^name = "/, "", name)
            sub(/"$/, "", name)
            next
        }
        in_asset && name == wanted && index($0, key " = ") == 1 {
            value = $0
            sub("^" key " = ", "", value)
            sub(/^"/, "", value)
            sub(/"$/, "", value)
            print value
            exit
        }
    ' "$MANIFEST"
}

sha256_file() {
    file=$1
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$file" | awk '{print $1}'
    else
        shasum -a 256 "$file" | awk '{print $1}'
    fi
}

check_required_paths() {
    paths=$1
    old_ifs=$IFS
    IFS='|'
    for path in $paths; do
        if [ -n "$path" ] && [ ! -e "$ROOT_DIR/$path" ]; then
            IFS=$old_ifs
            return 1
        fi
        if [ -n "$path" ] && [ -f "$ROOT_DIR/$path" ] && awk 'NR == 1 && $0 == "version https://git-lfs.github.com/spec/v1" { found = 1 } END { exit found ? 0 : 1 }' "$ROOT_DIR/$path"; then
            IFS=$old_ifs
            return 1
        fi
    done
    IFS=$old_ifs
    return 0
}

url=$(asset_value url)
sha256=$(asset_value sha256)
size_bytes=$(asset_value size_bytes)
filename=$(asset_value filename)
required_paths=$(asset_value required_paths)

if [ -z "$url" ] || [ -z "$sha256" ] || [ -z "$filename" ]; then
    echo "fixtures: asset '$ASSET' is incomplete in $MANIFEST" >&2
    exit 1
fi

if [ -n "${VIPRS_FIXTURES_BASE_URL:-}" ]; then
    url="${VIPRS_FIXTURES_BASE_URL%/}/$filename"
fi

current_manifest_hash=$(manifest_hash)
marker="$ROOT_DIR/tests/fixtures/.fixtures-$ASSET.manifest-sha256"

if [ -f "$marker" ] && [ "$(cat "$marker")" = "$current_manifest_hash" ] && check_required_paths "$required_paths"; then
    echo "fixtures: $ASSET ready"
    exit 0
fi

if check_required_paths "$required_paths"; then
    printf '%s' "$current_manifest_hash" > "$marker"
    echo "fixtures: $ASSET ready"
    exit 0
fi

if [ "${VIPRS_SKIP_FIXTURES_DOWNLOAD:-}" = "1" ]; then
    echo "fixtures: $ASSET missing or stale, and VIPRS_SKIP_FIXTURES_DOWNLOAD=1" >&2
    echo "fixtures: run 'make fixtures-$ASSET' without the skip flag or provide current fixtures locally" >&2
    exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
    echo "fixtures: curl is required to download $ASSET fixtures" >&2
    exit 127
fi

if ! command -v xz >/dev/null 2>&1; then
    echo "fixtures: xz is required to extract $filename" >&2
    exit 127
fi

mkdir -p "$CACHE_DIR"
archive="$CACHE_DIR/$filename"

if [ -f "$archive" ] && [ "$(sha256_file "$archive")" = "$sha256" ]; then
    echo "fixtures: using cached $filename"
else
    echo "fixtures: downloading $filename"
    curl --fail --location --retry 3 --retry-delay 2 --output "$archive.tmp" "$url"
    actual_size=$(wc -c < "$archive.tmp" | tr -d ' ')
    if [ -n "$size_bytes" ] && [ "$actual_size" != "$size_bytes" ]; then
        echo "fixtures: size mismatch for $filename: expected $size_bytes, got $actual_size" >&2
        rm -f "$archive.tmp"
        exit 1
    fi
    mv "$archive.tmp" "$archive"
fi

actual_sha256=$(sha256_file "$archive")
if [ "$actual_sha256" != "$sha256" ]; then
    echo "fixtures: sha256 mismatch for $filename" >&2
    echo "fixtures: expected $sha256" >&2
    echo "fixtures: got      $actual_sha256" >&2
    exit 1
fi

echo "fixtures: extracting $filename"
tar -C "$ROOT_DIR" -xJf "$archive"

if ! check_required_paths "$required_paths"; then
    echo "fixtures: extraction completed but required paths are still missing" >&2
    exit 1
fi

printf '%s' "$current_manifest_hash" > "$marker"
echo "fixtures: $ASSET ready"
