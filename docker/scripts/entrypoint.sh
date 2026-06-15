#!/bin/bash
# entrypoint.sh — Docker entrypoint for viprs perf benchmarks.
#
# Supports two hw-counter backends:
#   - cachegrind (default): deterministic cache simulation, always works
#   - perf stat: real PMU counters, needs kernel PMU access
#
# Usage:
#   /opt/bench/entrypoint.sh <input> <op> [op_args...] --iterations N --metrics hw|alloc|all

set -euo pipefail

INPUT=""
OP=""
OP_ARGS=()
ITERATIONS=10
METRICS="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --iterations) ITERATIONS="$2"; shift 2 ;;
        --metrics) METRICS="$2"; shift 2 ;;
        *)
            if [[ -z "$INPUT" ]]; then
                INPUT="$1"
            elif [[ -z "$OP" ]]; then
                OP="$1"
            else
                OP_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

if [[ -z "$INPUT" || -z "$OP" ]]; then
    echo "Usage: entrypoint.sh <input> <op> [args...] --iterations N --metrics hw|alloc|all" >&2
    exit 1
fi

# ── Cachegrind (deterministic cache simulation) ───────────────────────────────
run_cachegrind() {
    local binary="$1"
    local label="$2"
    shift 2

    local cg_out="/tmp/cg_${label}_$$"
    valgrind --tool=cachegrind --cachegrind-out-file="$cg_out" \
        "$binary" "$@" >/dev/null 2>&1 || true

    # Parse cachegrind output file
    if [[ -f "$cg_out" ]]; then
        local ir dr dw d1mr d1mw dLmr dLmw i1mr iLmr
        ir=$(grep "^summary:" "$cg_out" | awk '{print $2}')
        # Get detailed stats from cg_annotate
        local annotate
        annotate=$(cg_annotate "$cg_out" 2>/dev/null | head -30)

        # Extract from the totals line (format varies)
        local totals_line
        totals_line=$(cg_annotate "$cg_out" 2>/dev/null | grep -A1 "^-" | tail -1)

        # Simpler: parse the summary directly from the file
        echo "  \"${label}_cachegrind\": {"
        echo "    \"instruction_refs\": $(grep 'summary:' "$cg_out" | awk '{print $2}' | tr -d ','),"

        # Parse events line and data
        local events
        events=$(grep '^events:' "$cg_out" | sed 's/events: //')
        local data
        data=$(grep '^summary:' "$cg_out" | sed 's/summary: //')

        # Convert to JSON fields
        local i=0
        for event in $events; do
            local value
            value=$(echo "$data" | awk -v idx=$((i+1)) '{print $idx}')
            if [[ $i -gt 0 ]]; then echo ","
            fi
            printf "    \"%s\": %s" "$event" "${value:-0}"
            i=$((i+1))
        done
        echo ""
        echo "  }"
    else
        echo "  \"${label}_cachegrind\": {\"error\": \"cachegrind output not found\"}"
    fi
    rm -f "$cg_out"
}

# ── perf stat (real PMU, if available) ────────────────────────────────────────
run_perf_stat() {
    local binary="$1"
    local label="$2"
    shift 2

    if ! perf stat -e cycles true >/dev/null 2>&1; then
        echo "  \"${label}_perf\": {\"error\": \"PMU not available (Docker Desktop lacks hardware counter access)\"}"
        return
    fi

    local events="cycles,instructions,cache-references,cache-misses"
    events+=",L1-dcache-loads,L1-dcache-load-misses"
    events+=",branches,branch-misses"

    local perf_out="/tmp/perf_${label}_$$"
    perf stat -e "$events" -o "$perf_out" "$binary" "$@" >/dev/null 2>&1 || true

    if [[ -f "$perf_out" ]]; then
        echo "  \"${label}_perf\": {"
        local first=true
        while IFS= read -r line; do
            # perf stat output format: "  1,234,567  event-name  ..."
            local value event
            value=$(echo "$line" | awk '{print $1}' | tr -d ',')
            event=$(echo "$line" | awk '{print $2}')
            if [[ "$value" =~ ^[0-9]+$ && -n "$event" ]]; then
                if [[ "$first" != "true" ]]; then echo ","
                fi
                printf "    \"%s\": %s" "$event" "$value"
                first=false
            fi
        done < "$perf_out"
        echo ""
        echo "  }"
    fi
    rm -f "$perf_out"
}

# ── Allocation profiling (DHAT) ──────────────────────────────────────────────
run_dhat() {
    local binary="$1"
    local label="$2"
    shift 2

    local dhat_out="/tmp/dhat_${label}_$$.txt"
    valgrind --tool=dhat --dhat-out-file=/dev/null \
        "$binary" "$@" 2>"$dhat_out" || true

    if [[ -f "$dhat_out" ]]; then
        local total_blocks total_bytes peak_bytes
        total_blocks=$(grep "total blocks allocated:" "$dhat_out" | grep -o '[0-9,]*' | head -1 | tr -d ',')
        total_bytes=$(grep "total bytes allocated:" "$dhat_out" | grep -o '[0-9,]*' | head -1 | tr -d ',')
        peak_bytes=$(grep "max bytes live:" "$dhat_out" 2>/dev/null | grep -o '[0-9,]*' | head -1 | tr -d ',' || echo "0")

        echo "  \"${label}_dhat\": {"
        echo "    \"total_blocks\": ${total_blocks:-0},"
        echo "    \"total_bytes\": ${total_bytes:-0},"
        echo "    \"peak_live_bytes\": ${peak_bytes:-0}"
        echo "  }"
    else
        echo "  \"${label}_dhat\": {\"error\": \"dhat failed\"}"
    fi
    rm -f "$dhat_out"
}

# ── Main ──────────────────────────────────────────────────────────────────────
echo "{"
echo "  \"input\": \"$INPUT\","
echo "  \"operation\": \"$OP\","
echo "  \"iterations\": $ITERATIONS,"
echo "  \"arch\": \"$(uname -m)\","

LIBVIPS_ARGS=("$INPUT" "$OP" "${OP_ARGS[@]}" "--iterations" "$ITERATIONS")
VIPRS_ARGS=("bench" "$INPUT" "$OP" "${OP_ARGS[@]}" "--iterations" "$ITERATIONS")

COMMA=""

if [[ "$METRICS" == "hw" || "$METRICS" == "all" ]]; then
    echo "${COMMA}"
    # Cachegrind (always works)
    run_cachegrind libvips-runner "libvips" "${LIBVIPS_ARGS[@]}"
    echo ","
    run_cachegrind viprs-xtask "viprs" "${VIPRS_ARGS[@]}"

    # Also try real perf stat
    echo ","
    run_perf_stat libvips-runner "libvips" "${LIBVIPS_ARGS[@]}"
    echo ","
    run_perf_stat viprs-xtask "viprs" "${VIPRS_ARGS[@]}"
    COMMA=","
fi

if [[ "$METRICS" == "alloc" || "$METRICS" == "all" ]]; then
    echo "${COMMA}"
    run_dhat libvips-runner "libvips" "${LIBVIPS_ARGS[@]}"
    echo ","
    run_dhat viprs-xtask "viprs" "${VIPRS_ARGS[@]}"
    COMMA=","
fi

echo ""
echo "}"
