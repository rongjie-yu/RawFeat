"""Readable Markdown/HTML reports for completed HPatches evaluations."""

from __future__ import annotations

import csv
import html
import json
import statistics
from pathlib import Path
from typing import Any


CONDITIONS = ("clean", "ratio1", "ratio4", "ratio16", "ratio64", "ratio100")
NOISY_CONDITIONS = CONDITIONS[1:]
GROUPS = ("all", "illumination", "viewpoint")
METHOD_LABELS = {"rawfeat": "RawFeat", "superpoint": "SuperPoint"}
PERCENT_KEYS = (
    "h_auc_1", "h_auc_3", "h_auc_5", "threshold_success_1", "threshold_success_3",
    "threshold_success_5", "repeatability", "match_precision",
)
DETAIL_KEYS = (
    "h_auc_1", "h_auc_3", "h_auc_5", "threshold_success_5", "homography_failures",
    "matches", "correct_matches", "match_precision", "detected_a", "detected_b",
    "repeatability", "localization_error", "corner_error_median", "corner_error_p95",
    "corner_error_gt20",
)
PAIR_KEYS = (
    "method", "sequence", "type", "target_index", "condition", "h_auc_5",
    "threshold_success_5", "homography_failure", "corner_error", "matches",
    "correct_matches", "match_precision", "detected_a", "detected_b",
    "repeatability", "localization_error",
)
LATENCY_KEYS = (
    "method", "condition", "batch_size", "measurements", "mean_ms", "median_ms",
    "p10_ms", "p90_ms", "std_ms", "network_forward_mean_ms", "feature_extraction_mean_ms",
    "images_per_second", "peak_memory_bytes", "peak_allocated_bytes", "peak_reserved_bytes",
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}" if isinstance(value, float) else str(value)
    return str(value)


def _percent(value: Any) -> str:
    if value is None:
        return "—"
    return f"{100 * float(value):.2f}%"


def _delta(rawfeat: Any, superpoint: Any) -> str:
    if rawfeat is None or superpoint is None:
        return "—"
    return f"{100 * (float(rawfeat) - float(superpoint)):+.2f} pp"


def _method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _latency_summary(evaluation: Path, run_manifest: dict[str, Any]) -> dict[str, Any]:
    timing_path = evaluation / "image_inference.jsonl"
    if not timing_path.is_file():
        return {
            "available": False,
            "source": "image_inference.jsonl",
            "total_wall_clock_seconds": run_manifest.get("total_wall_clock_seconds"),
            "rows": [],
        }
    rows = _load_jsonl(timing_path)
    if not rows:
        raise ValueError("image_inference.jsonl is empty")
    required = {"method", "image_id", "condition", "batch_size", "single_image_inference",
                "network_forward_ms", "feature_extraction_ms", "total_inference_ms",
                "peak_memory_bytes"}
    if any(not required.issubset(row) for row in rows):
        raise ValueError("every timing row must contain the single-image timing schema")
    if any(int(row["batch_size"]) != 1 or not row["single_image_inference"] for row in rows):
        raise ValueError("latency summary refuses non-single-image timing records")
    keys = {(str(row["method"]), str(row["image_id"]), str(row["condition"])) for row in rows}
    if len(keys) != len(rows):
        raise ValueError("timing rows must identify one real inference per method/image/condition")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["method"]), str(row["condition"])), []).append(row)
    summaries = []
    network_only = []
    for (method, condition), group in sorted(groups.items()):
        total = [float(row["total_inference_ms"]) for row in group]
        network = [float(row["network_forward_ms"]) for row in group]
        feature = [float(row["feature_extraction_ms"]) for row in group]
        mean_total = statistics.mean(total)
        mean_network = statistics.mean(network)
        peak_allocated = max(int(row.get("peak_allocated_bytes", row["peak_memory_bytes"])) for row in group)
        peak_reserved = max(int(row.get("peak_reserved_bytes", row["peak_memory_bytes"])) for row in group)
        summary = {
            "method": method, "condition": condition, "batch_size": 1, "measurements": len(group),
            "mean_ms": mean_total, "median_ms": statistics.median(total),
            "p10_ms": _percentile(total, 0.10), "p90_ms": _percentile(total, 0.90),
            "std_ms": statistics.stdev(total) if len(total) > 1 else 0.0,
            "network_forward_mean_ms": mean_network,
            "network_forward_median_ms": statistics.median(network),
            "network_forward_p10_ms": _percentile(network, 0.10),
            "network_forward_p90_ms": _percentile(network, 0.90),
            "network_forward_std_ms": statistics.stdev(network) if len(network) > 1 else 0.0,
            "feature_extraction_mean_ms": statistics.mean(feature),
            "feature_extraction_median_ms": statistics.median(feature),
            "feature_extraction_p10_ms": _percentile(feature, 0.10),
            "feature_extraction_p90_ms": _percentile(feature, 0.90),
            "feature_extraction_std_ms": statistics.stdev(feature) if len(feature) > 1 else 0.0,
            "images_per_second": 1000.0 / mean_total if mean_total else None,
            "peak_memory_bytes": peak_allocated,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        }
        summaries.append(summary)
        network_only.append({
            "method": method, "condition": condition, "batch_size": 1, "measurements": len(group),
            "mean_ms": mean_network, "median_ms": statistics.median(network),
            "p10_ms": _percentile(network, 0.10), "p90_ms": _percentile(network, 0.90),
            "std_ms": statistics.stdev(network) if len(network) > 1 else 0.0,
            "images_per_second": 1000.0 / mean_network if mean_network else None,
            "peak_memory_bytes": peak_allocated, "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        })
    return {
        "available": True,
        "source": "image_inference.jsonl",
        "measurement_unit": "one actual image inference",
        "batch_size": 1,
        "stage": "network_forward_and_full_single_image_pipeline",
        "cache_generation_included": False,
        "single_image_inference": True,
        "notes": [
            "Cache generation time is excluded.",
            "SuperPoint CPU Bayer demosaic, uint16 conversion, MeanAD, GPU transfer, network forward, and feature extraction are included in total_inference_ms.",
            "network_forward_ms is measured with CUDA Events; feature extraction and total inference are measured after CUDA synchronization.",
        ],
        "total_wall_clock_seconds": run_manifest.get("total_wall_clock_seconds"),
        "rows": summaries,
        "network_only_rows": network_only,
    }


def _write_latency_csv(path: Path, latency: dict[str, Any]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(LATENCY_KEYS))
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in LATENCY_KEYS} for row in latency["rows"])


def _markdown_latency(latency: dict[str, Any]) -> str:
    if not latency["available"]:
        return "No `image_inference.jsonl` timing records were found."
    total = latency.get("total_wall_clock_seconds")
    total_text = "—" if total is None else f"{float(total):.3f} s ({float(total) / 60:.2f} min)"
    lines = [
        f"- Total wall-clock: **{total_text}**; cache generation included: **{latency['cache_generation_included']}**.",
        f"- Source: `{latency['source']}`; each row is **{latency['measurement_unit']}**; `batch_size=1`; stage: `{latency['stage']}`.",
        "- Network-only forward and the complete single-image pipeline are reported separately. SuperPoint MeanAD/demosaic preprocessing is included in total inference.",
        "",
        "| method | condition | images | total mean ms | total median ms | p10 ms | p90 ms | std ms | network mean ms | feature mean ms | images/s | peak allocated MiB | peak reserved MiB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in latency["rows"]:
        peak_mib = row["peak_allocated_bytes"] / (1024 ** 2)
        reserved_mib = row["peak_reserved_bytes"] / (1024 ** 2)
        lines.append(
            f"| {_method_label(row['method'])} | {row['condition']} | {row['measurements']} | "
            f"{row['mean_ms']:.3f} | {row['median_ms']:.3f} | {row['p10_ms']:.3f} | {row['p90_ms']:.3f} | "
            f"{row['std_ms']:.3f} | {row['network_forward_mean_ms']:.3f} | {row['feature_extraction_mean_ms']:.3f} | "
            f"{row['images_per_second']:.2f} | {peak_mib:.1f} | {reserved_mib:.1f} |"
        )
    return "\n".join(lines)


def _summary_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in GROUPS:
        for condition in CONDITIONS:
            rawfeat = summary["summaries"]["rawfeat"][group][condition]
            superpoint = summary["summaries"]["superpoint"][group][condition]
            rows.append({"group": group, "condition": condition, "rawfeat": rawfeat, "superpoint": superpoint})
    return rows


def _markdown_main_table(summary: dict[str, Any]) -> str:
    lines = [
        "| condition | RawFeat H-AUC@5 | SuperPoint H-AUC@5 | difference | RawFeat failures | SuperPoint failures |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        rawfeat = summary["summaries"]["rawfeat"]["all"][condition]
        superpoint = summary["summaries"]["superpoint"]["all"][condition]
        lines.append(
            f"| {condition} | {_percent(rawfeat['h_auc_5'])} | {_percent(superpoint['h_auc_5'])} | "
            f"{_delta(rawfeat['h_auc_5'], superpoint['h_auc_5'])} | {rawfeat['homography_failures']} | {superpoint['homography_failures']} |"
        )
    noisy_rawfeat = summary["main_h_auc_5"]["rawfeat"]
    noisy_superpoint = summary["main_h_auc_5"]["superpoint"]
    lines.append(
        f"| **noisy mean** | **{_percent(noisy_rawfeat)}** | **{_percent(noisy_superpoint)}** | "
        f"**{_delta(noisy_rawfeat, noisy_superpoint)}** | — | — |"
    )
    return "\n".join(lines)


def _markdown_detail_table(summary: dict[str, Any]) -> str:
    lines = [
        "| group | condition | method | H-AUC@1 | H-AUC@3 | H-AUC@5 | success≤5 | failures | MNN | correct | precision | points A/B | repeatability | localization px | corner median/p95 | >20px |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in GROUPS:
        for condition in CONDITIONS:
            for method in ("rawfeat", "superpoint"):
                row = summary["summaries"][method][group][condition]
                lines.append(
                    f"| {group} | {condition} | {_method_label(method)} | {_percent(row['h_auc_1'])} | {_percent(row['h_auc_3'])} | "
                    f"{_percent(row['h_auc_5'])} | {_percent(row['threshold_success_5'])} | {row['homography_failures']} | "
                    f"{_number(row['matches'], 1)} | {_number(row['correct_matches'], 1)} | {_percent(row['match_precision'])} | "
                    f"{_number(row['detected_a'], 1)}/{_number(row['detected_b'], 1)} | {_percent(row['repeatability'])} | "
                    f"{_number(row['localization_error'], 2)} | {_number(row['corner_error_median'], 2)}/{_number(row['corner_error_p95'], 2)} | "
                    f"{row['corner_error_gt20']} |"
                )
    return "\n".join(lines)


def _markdown_bootstrap(summary: dict[str, Any]) -> str:
    lines = [
        "| group | estimate | 95% interval | sequences | positive/negative |",
        "|---|---:|---:|---:|---:|",
    ]
    for group in GROUPS:
        row = summary["sequence_cluster_bootstrap"][group]
        lines.append(
            f"| {group} | {_delta(row['estimate'], 0)} | [{_percent(row['ci95'][0])}, {_percent(row['ci95'][1])}] | "
            f"{row['sequence_count']} | {row['positive_sequences']}/{row['negative_sequences']} |"
        )
    return "\n".join(lines)


def _write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    fields = ["group", "condition", "method", *DETAIL_KEYS]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in _summary_rows(summary):
            for method in ("rawfeat", "superpoint"):
                values = row[method]
                writer.writerow({"group": row["group"], "condition": row["condition"], "method": method,
                                 **{key: values.get(key) for key in DETAIL_KEYS}})


def _write_failures(path: Path, rows: list[dict[str, Any]]) -> int:
    failures = [row for row in rows if row.get("homography_failure")]
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in failures))
    return len(failures)


def _write_pairs_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(PAIR_KEYS))
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in PAIR_KEYS} for row in rows)


def _write_pairs_html(path: Path, rows: list[dict[str, Any]]) -> None:
    table_rows = []
    for row in rows:
        failure = bool(row.get("homography_failure"))
        cells = [
            _method_label(str(row.get("method"))), row.get("sequence"), row.get("type"), row.get("target_index"),
            row.get("condition"), _percent(row.get("h_auc_5")), _percent(row.get("threshold_success_5")),
            "yes" if failure else "no", _number(row.get("corner_error"), 2), row.get("matches"),
            row.get("correct_matches"), _percent(row.get("match_precision")), row.get("detected_a"),
            row.get("detected_b"), _percent(row.get("repeatability")), _number(row.get("localization_error"), 2),
        ]
        table_rows.append(
            f'<tr data-method="{html.escape(str(row.get("method")))}" data-type="{html.escape(str(row.get("type")))}" '
            f'data-condition="{html.escape(str(row.get("condition")))}" data-sequence="{html.escape(str(row.get("sequence")))}" '
            f'data-failure="{"yes" if failure else "no"}">' + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in cells) + "</tr>"
        )
    headers = ("method", "sequence", "type", "target", "condition", "H-AUC@5", "success≤5", "failure",
               "corner px", "MNN", "correct", "precision", "points A", "points B", "repeatability", "localization px")
    document = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HPatches per-pair rows</title>
<style>body{{font:13px system-ui,sans-serif;color:#172033;background:#f8fafc;margin:20px}}table{{border-collapse:collapse;background:white;width:100%}}th,td{{border:1px solid #dbe3ee;padding:5px 7px;white-space:nowrap;text-align:right}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2),th:nth-child(3),td:nth-child(3),th:nth-child(5),td:nth-child(5){{text-align:left}}th{{background:#e8eef7;position:sticky;top:0}}input,select{{padding:6px;margin:0 6px 10px 0}}.count{{color:#64748b}}</style></head><body>
<h1>HPatches per-pair rows</h1><div class="count">{len(rows)} rows; use filters to narrow the table.</div>
<label>method <select id="method"><option value="all">all</option><option value="rawfeat">RawFeat</option><option value="superpoint">SuperPoint</option></select></label>
<label>type <select id="type"><option value="all">all</option><option>illumination</option><option>viewpoint</option></select></label>
<label>condition <select id="condition"><option value="all">all</option>{"".join(f'<option>{c}</option>' for c in CONDITIONS)}</select></label>
<label>sequence <input id="sequence" placeholder="e.g. v_wall"></label>
<label><input type="checkbox" id="failures"> failures only</label>
<table><thead><tr>{"".join(f"<th>{header}</th>" for header in headers)}</tr></thead><tbody>{"".join(table_rows)}</tbody></table>
<script>function filter(){{const method=document.getElementById('method').value,type=document.getElementById('type').value,condition=document.getElementById('condition').value,sequence=document.getElementById('sequence').value.toLowerCase(),failures=document.getElementById('failures').checked;document.querySelectorAll('tbody tr').forEach(row=>{{const ok=(method==='all'||row.dataset.method===method)&&(type==='all'||row.dataset.type===type)&&(condition==='all'||row.dataset.condition===condition)&&(!sequence||row.dataset.sequence.toLowerCase().includes(sequence))&&(!failures||row.dataset.failure==='yes');row.style.display=ok?'':'none'}})}}document.querySelectorAll('select,input').forEach(x=>x.oninput=filter);</script>
</body></html>'''
    path.write_text(document)


def _svg_chart(summary: dict[str, Any]) -> str:
    width, height = 720, 250
    chart_left, chart_top, chart_width, chart_height = 55, 20, 640, 175
    bar_width = 42
    gap = chart_width / len(NOISY_CONDITIONS)
    elements = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Noisy H-AUC at 5 comparison">']
    elements.append(f'<line x1="{chart_left}" y1="{chart_top + chart_height}" x2="{chart_left + chart_width}" y2="{chart_top + chart_height}" stroke="#555"/>')
    for index, condition in enumerate(NOISY_CONDITIONS):
        x = chart_left + gap * index + gap / 2
        rawfeat = summary["summaries"]["rawfeat"]["all"][condition]["h_auc_5"]
        superpoint = summary["summaries"]["superpoint"]["all"][condition]["h_auc_5"]
        for offset, value, color, label in ((-bar_width / 2, rawfeat, "#2563eb", "RawFeat"),
                                             (bar_width / 2, superpoint, "#f97316", "SuperPoint")):
            bar_height = chart_height * float(value)
            y = chart_top + chart_height - bar_height
            elements.append(f'<rect x="{x + offset - bar_width / 2:.1f}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="{color}"><title>{condition} {label}: {100 * value:.2f}%</title></rect>')
        elements.append(f'<text x="{x:.1f}" y="{chart_top + chart_height + 22}" text-anchor="middle">{html.escape(condition)}</text>')
    elements.extend(['<rect x="560" y="8" width="12" height="12" fill="#2563eb"/><text x="578" y="18">RawFeat</text>', '<rect x="630" y="8" width="12" height="12" fill="#f97316"/><text x="648" y="18">SuperPoint</text>', '</svg>'])
    return "".join(elements)


def _write_html(
    path: Path,
    summary: dict[str, Any],
    failure_count: int,
    latency: dict[str, Any],
) -> None:
    rows = []
    for item in _summary_rows(summary):
        for method in ("rawfeat", "superpoint"):
            row = item[method]
            rows.append(
                f'<tr data-group="{item["group"]}" data-condition="{item["condition"]}"><td>{item["group"]}</td><td>{item["condition"]}</td><td>{_method_label(method)}</td>'
                f'<td>{_percent(row["h_auc_5"])}</td><td>{_percent(row["threshold_success_5"])}</td><td>{row["homography_failures"]}</td>'
                f'<td>{_number(row["matches"], 1)}</td><td>{_number(row["correct_matches"], 1)}</td><td>{_percent(row["match_precision"])}</td>'
                f'<td>{_percent(row["repeatability"])}</td><td>{_number(row["localization_error"], 2)}</td></tr>'
            )
    gallery = []
    for image in sorted((path.parent.parent / "visualizations").glob("*.png")):
        gallery.append(f'<a href="../visualizations/{html.escape(image.name)}"><img src="../visualizations/{html.escape(image.name)}" alt="{html.escape(image.stem)}"><span>{html.escape(image.stem)}</span></a>')
    condition_links = " ".join(
        f'<a href="../visualizations/condition_{condition}.html">{condition}</a>'
        for condition in CONDITIONS
    )
    bootstrap_items = []
    for group in GROUPS:
        row = summary["sequence_cluster_bootstrap"][group]
        bootstrap_items.append(f'<tr><td>{group}</td><td>{_delta(row["estimate"], 0)}</td><td>[{_percent(row["ci95"][0])}, {_percent(row["ci95"][1])}]</td><td>{row["positive_sequences"]}/{row["negative_sequences"]}</td></tr>')
    latency_rows = []
    for row in latency["rows"]:
        latency_rows.append(
            f'<tr><td>{html.escape(_method_label(row["method"]))}</td><td>{html.escape(row["condition"])}</td>'
            f'<td>{row["batch_size"]}</td><td>{row["measurements"]}</td><td>{row["mean_ms"]:.3f}</td>'
            f'<td>{row["median_ms"]:.3f}</td><td>{row["p10_ms"]:.3f}</td><td>{row["p90_ms"]:.3f}</td>'
            f'<td>{row["std_ms"]:.3f}</td><td>{row["network_forward_mean_ms"]:.3f}</td>'
            f'<td>{row["feature_extraction_mean_ms"]:.3f}</td><td>{row["images_per_second"]:.2f}</td>'
            f'<td>{row["peak_allocated_bytes"] / (1024 ** 2):.1f}</td>'
            f'<td>{row["peak_reserved_bytes"] / (1024 ** 2):.1f}</td></tr>'
        )
    latency_total = latency.get("total_wall_clock_seconds")
    latency_total_text = "—" if latency_total is None else f"{float(latency_total):.3f} s"
    latency_section = (
        f'<h2>Runtime and latency</h2><p class="muted">Total wall-clock: {latency_total_text}; '
        f'cache generation included: {latency.get("cache_generation_included", "—")}; '
        f'each row is one actual image inference at batch_size=1. SuperPoint MeanAD/demosaic preprocessing is included in total inference.</p>'
        '<table><thead><tr><th>Method</th><th>Condition</th><th>Batch</th><th>Images</th>'
        '<th>Total mean ms</th><th>Total median ms</th><th>p10 ms</th><th>p90 ms</th><th>Total std ms</th>'
        '<th>Network mean ms</th><th>Feature mean ms</th><th>Images/s</th><th>Peak allocated MiB</th><th>Peak reserved MiB</th></tr></thead><tbody>'
        + "".join(latency_rows)
        + '</tbody></table>'
    ) if latency["available"] else '<h2>Runtime and latency</h2><p>No timing records found.</p>'
    document = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HPatches RawFeat report</title>
<style>
body{{font:14px system-ui,sans-serif;color:#172033;max-width:1450px;margin:24px auto;padding:0 20px;background:#f8fafc}}h1,h2{{color:#0f172a}}.cards{{display:flex;gap:12px;flex-wrap:wrap}}.card{{background:white;border:1px solid #dbe3ee;border-radius:8px;padding:12px 16px;min-width:180px}}table{{border-collapse:collapse;background:white;margin:10px 0 24px;width:100%}}th,td{{padding:7px 9px;border:1px solid #dbe3ee;text-align:right}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2),th:nth-child(3),td:nth-child(3){{text-align:left}}th{{background:#e8eef7;position:sticky;top:0}}tr[data-method="rawfeat"]{{}}select{{padding:6px;margin-right:8px}}.gallery{{display:flex;gap:14px;flex-wrap:wrap}}.gallery a{{color:#334155;text-decoration:none;max-width:310px}}.gallery img{{width:310px;display:block;border:1px solid #cbd5e1}}.gallery span{{display:block;overflow-wrap:anywhere}}.muted{{color:#64748b}}
</style></head><body>
<h1>HPatches synthetic RawFeat evaluation</h1>
<p class="muted">Protocol: {html.escape(summary["protocol"])} · {summary["pairs"]} pairs/condition · {len(summary["conditions"])} conditions · full_protocol={summary["full_protocol"]}</p>
<div class="cards"><div class="card"><b>RawFeat noisy mean H-AUC@5</b><br>{_percent(summary["main_h_auc_5"]["rawfeat"])}</div><div class="card"><b>SuperPoint noisy mean H-AUC@5</b><br>{_percent(summary["main_h_auc_5"]["superpoint"])}</div><div class="card"><b>Difference</b><br>{_delta(summary["main_h_auc_5"]["rawfeat"], summary["main_h_auc_5"]["superpoint"])}</div><div class="card"><b>Homography failures</b><br>{failure_count} pair/method rows</div></div>
<h2>Noisy H-AUC@5</h2>{_svg_chart(summary)}
<h2>All / condition summary</h2><table><thead><tr><th>Group</th><th>Condition</th><th>Method</th><th>H-AUC@5</th><th>Success≤5</th><th>Failures</th><th>MNN</th><th>Correct</th><th>Precision</th><th>Repeatability</th><th>Localization px</th></tr></thead><tbody>{"".join(rows)}</tbody></table>
<label>Filter group <select id="group"><option value="all">all</option><option value="illumination">illumination</option><option value="viewpoint">viewpoint</option></select></label><label>Filter condition <select id="condition"><option value="all">all conditions</option>{"".join(f'<option value="{c}">{c}</option>' for c in CONDITIONS)}</select></label>
<h2>Sequence-cluster bootstrap</h2><table><thead><tr><th>Group</th><th>Estimate</th><th>95% interval</th><th>Positive/negative sequences</th></tr></thead><tbody>{"".join(bootstrap_items)}</tbody></table>
{latency_section}
<h2>Visualizations</h2><p>Condition pages: {condition_links}</p><p><a href="../visualizations/index.html">filterable index</a> · <a href="../visualizations/rawfeat_superpoint.html">RawFeat/SuperPoint comparison</a> · <a href="../visualizations/contact_sheet.html">six-condition overview</a></p><div class="gallery">{"".join(gallery)}</div>
<p class="muted">Download: <a href="summary.csv">summary.csv</a> · <a href="pairs.html">pairs.html</a> · <a href="pairs.csv">pairs.csv</a> · <a href="failures.jsonl">failures.jsonl</a> · <a href="latency_summary.json">latency_summary.json</a> · <a href="latency.csv">latency.csv</a> · <a href="../image_inference.jsonl">image_inference.jsonl</a> · <a href="../per_pair.jsonl">per_pair.jsonl</a> · <a href="../run_manifest.json">run_manifest.json</a></p>
<script>function filter(){{const g=document.getElementById('group').value,c=document.getElementById('condition').value;document.querySelectorAll('tbody tr[data-group]').forEach(r=>r.style.display=(g==='all'||r.dataset.group===g)&&(c==='all'||r.dataset.condition===c)?'':'none')}}document.getElementById('group').onchange=filter;document.getElementById('condition').onchange=filter;</script>
</body></html>'''
    path.write_text(document)


def write_hpatches_report(evaluation_dir: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    evaluation = Path(evaluation_dir)
    summary = json.loads((evaluation / "summary.json").read_text())
    run_manifest = json.loads((evaluation / "run_manifest.json").read_text())
    output = Path(output_dir) if output_dir is not None else evaluation / "report"
    output.mkdir(parents=True, exist_ok=True)
    rows = _load_jsonl(evaluation / "per_pair.jsonl")
    latency = _latency_summary(evaluation, run_manifest)
    failures = _write_failures(output / "failures.jsonl", rows)
    _write_summary_csv(output / "summary.csv", summary)
    _write_pairs_csv(output / "pairs.csv", rows)
    _write_pairs_html(output / "pairs.html", rows)
    (output / "latency_summary.json").write_text(json.dumps(latency, indent=2, allow_nan=False) + "\n")
    (evaluation / "latency_summary.json").write_text(json.dumps(latency, indent=2, allow_nan=False) + "\n")
    _write_latency_csv(output / "latency.csv", latency)
    markdown = f'''# HPatches synthetic RawFeat evaluation

- Protocol: `{summary["protocol"]}`
- Pairs per condition: **{summary["pair_count_per_condition"]}**
- Conditions: `{", ".join(summary["conditions"])}`
- RawFeat checkpoint SHA256: `{summary["rawfeat_checkpoint_sha256"]}`
- SuperPoint checkpoint SHA256: `{summary["superpoint_checkpoint_sha256"]}`
- Full protocol: **{bool(summary.get("full_protocol"))}**
- Single-image inference: **{bool(summary.get("single_image_inference", run_manifest.get("single_image_inference")))}** (`batch_size=1`)
- Visualization count: **{summary.get("visualizations", {}).get("count", 0)}**
- Pair/method rows with homography failure: **{failures}**

## Main H-AUC@5

{_markdown_main_table(summary)}

The noisy mean excludes `clean`; it averages `ratio1`, `ratio4`, `ratio16`, `ratio64`, and `ratio100` equally. H-AUC and threshold success are reported separately.

## Sequence-cluster bootstrap for RawFeat − SuperPoint noisy H-AUC@5

{_markdown_bootstrap(summary)}

Bootstrap seed: `{summary["sequence_cluster_bootstrap"]["all"]["seed"]}`, iterations: `{summary["sequence_cluster_bootstrap"]["all"]["iterations"]}`.

## Runtime and latency

{_markdown_latency(latency)}

## All / illumination / viewpoint details

{_markdown_detail_table(summary)}

## Visualizations

- [HTML report](report.html)
- [Summary CSV](summary.csv)
- [Per-pair browser](pairs.html)
- [Per-pair CSV](pairs.csv)
- [Failure rows](failures.jsonl)
- [Latency summary](latency_summary.json)
- [Latency CSV](latency.csv)
- [Single-image timing records](../image_inference.jsonl)
- [All per-pair rows](../per_pair.jsonl)
- [Visualization index](../visualizations/index.html)
- [RawFeat/SuperPoint comparison](../visualizations/rawfeat_superpoint.html)
- [Six-condition overview](../visualizations/contact_sheet.html)
- Condition pages: {" · ".join(f"[{condition}](../visualizations/condition_{condition}.html)" for condition in CONDITIONS)}

{chr(10).join(f'- ![{path.stem}](../visualizations/{path.name})' for path in sorted((evaluation / "visualizations").glob("*.png")))}
'''
    (output / "report.md").write_text(markdown)
    _write_html(output / "report.html", summary, failures, latency)
    return {"evaluation": str(evaluation.resolve()), "report": str(output.resolve()),
            "pairs": summary["pair_count_per_condition"], "failure_rows": failures,
            "rawfeat_main_h_auc_5": summary["main_h_auc_5"]["rawfeat"],
            "superpoint_main_h_auc_5": summary["main_h_auc_5"]["superpoint"],
            "full_protocol": bool(summary.get("full_protocol")), "command": run_manifest.get("command"),
            "total_wall_clock_seconds": latency.get("total_wall_clock_seconds"),
            "latency_rows": len(latency["rows"]),
            "latency_summary": str((output / "latency_summary.json").resolve()),
            "visualization_count": summary.get("visualizations", {}).get("count", 0),
            "report_generated": True}
