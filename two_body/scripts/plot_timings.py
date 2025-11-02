from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from perf_timings.io_utils import (
    REPORTS_DIR,
    TIMINGS_DIR,
    ensure_directory,
    filter_rows,
    latest_timing_csv,
    list_timing_csvs,
    parse_sections_arg,
    read_timings_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera graficas de timeline y distribucion de secciones a partir del CSV de timings."
    )
    parser.add_argument("--run-id", help="UUID del run a graficar (default: mas reciente).")
    parser.add_argument("--epoch", type=int, help="Filtra un epoch especifico.")
    parser.add_argument("--batch-id", type=int, help="Filtra un batch especifico.")
    parser.add_argument(
        "--top-n",
        type=int,
        default=0,
        help="Limita el numero de items por grafica para mejorar legibilidad.",
    )
    parser.add_argument(
        "--sections",
        help="Lista separada por comas para filtrar secciones (ej. simulation_step,fitness_eval).",
    )
    parser.add_argument(
        "--data-dir",
        default=str(TIMINGS_DIR),
        help="Directorio base que contiene los CSV (default: data/timings).",
    )
    parser.add_argument(
        "--batch-individual",
        action="store_true",
        help="Timeline de batches con filas por individuo en lugar de un unico renglón por batch.",
    )
    return parser.parse_args()


def resolve_csv(run_id: str | None, data_dir: Path) -> Path:
    files = list_timing_csvs(data_dir)
    if not files:
        raise FileNotFoundError(f"No se encontraron CSVs en {data_dir}.")
    if not run_id:
        return files[-1]
    candidates = [p for p in files if f"timings_{run_id}_" in p.name]
    if not candidates:
        raise FileNotFoundError(f"No se encontro CSV para run_id={run_id}.")
    return candidates[-1]


def _labels_with_suffix(rows: List[Tuple[str, dict[str, Any]]]) -> List[str]:
    counts: dict[str, int] = {}
    final: List[str] = []
    for label, _ in rows:
        if label not in counts:
            counts[label] = 1
            final.append(label)
        else:
            counts[label] += 1
            final.append(f"{label} #{counts[label]}")
    return final


def _build_color_map(rows: List[dict[str, Any]]) -> dict[str, Any]:
    sections = sorted({row.get("section", "") for row in rows if row.get("section")})
    cmap = plt.get_cmap("tab20")
    mapping: dict[str, Any] = {}
    for idx, section in enumerate(sections):
        mapping[section] = cmap(idx % cmap.N)
    return mapping


def timeline_plot(
    rows: List[dict[str, Any]],
    label_key: Callable[[dict[str, Any]], str | None],
    title: str,
    ylabel: str,
    output: Path,
    top_n: int = 0,
    aggregate: bool = False,
) -> None:
    color_map = _build_color_map(rows)
    default_color = "#1f77b4"

    if aggregate:
        grouped_map: dict[str, List[dict[str, Any]]] = {}
        for row in rows:
            label = label_key(row)
            if not label:
                continue
            grouped_map.setdefault(label, []).append(row)
        if not grouped_map:
            save_empty_plot(output, title)
            return
        grouped = sorted(grouped_map.items(), key=lambda item: item[1][0]["start_ns"])
        if top_n and top_n > 0:
            grouped = grouped[:top_n]
        base_ns = min(entry["start_ns"] for _, entries in grouped for entry in entries)
        fig, ax = plt.subplots(figsize=(12, max(3.0, len(grouped) * 0.6)))
        for idx, (label, entries) in enumerate(grouped):
            for row in entries:
                start_rel = (row["start_ns"] - base_ns) // 1_000
                width = max(1, row["duration_us"])
                color = color_map.get(row.get("section", ""), default_color)
                ax.barh(idx, width, left=start_rel, height=0.4, color=color)
        ax.set_yticks(range(len(grouped)))
        ax.set_yticklabels([label for label, _ in grouped])
    else:
        entries_list: List[Tuple[str, dict[str, Any]]] = []
        for row in rows:
            label = label_key(row)
            if not label:
                continue
            entries_list.append((label, row))
        if not entries_list:
            save_empty_plot(output, title)
            return
        entries_list.sort(key=lambda item: item[1]["start_ns"])
        if top_n and top_n > 0:
            entries_list = entries_list[:top_n]
        base_ns = min(item[1]["start_ns"] for item in entries_list)
        labels = _labels_with_suffix(entries_list)
        fig, ax = plt.subplots(figsize=(12, max(3.0, len(entries_list) * 0.35)))
        for idx, ((_, row), label) in enumerate(zip(entries_list, labels, strict=False)):
            start_rel = (row["start_ns"] - base_ns) // 1_000
            width = max(1, row["duration_us"])
            color = color_map.get(row.get("section", ""), default_color)
            ax.barh(idx, width, left=start_rel, height=0.4, color=color)
        ax.set_yticks(range(len(entries_list)))
        ax.set_yticklabels(labels)

    legend_handles = [
        Patch(facecolor=color_map[section], label=section) for section in color_map if section
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    ax.set_xlabel("Tiempo relativo (us)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def save_empty_plot(path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.axis("off")
    ax.text(0.5, 0.5, "Sin datos para los filtros indicados.", ha="center", va="center")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pie_chart(rows: List[dict[str, Any]], title: str, output: Path) -> None:
    totals: dict[str, int] = {}
    for row in rows:
        totals[row["section"]] = totals.get(row["section"], 0) + int(row["duration_us"])
    if not totals:
        save_empty_plot(output, title)
        return
    labels: list[str] = []
    values: list[int] = []
    legend_entries: list[str] = []
    grand_total = sum(totals.values())
    for section, value in totals.items():
        values.append(value)
        labels.append(section)
        pct = (value / grand_total) * 100 if grand_total else 0.0
        legend_entries.append(f"{section} — {pct:.1f}% ({value:,.0f} us)")
    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, _ = ax.pie(values, labels=None, startangle=90)
    ax.legend(
        wedges,
        legend_entries,
        title="Secciones",
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=9,
    )
    ax.set_title(title)
    ax.axis("equal")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    try:
        csv_path = resolve_csv(args.run_id, data_dir)
    except FileNotFoundError as err:
        print(err, file=sys.stderr)
        sys.exit(1)

    rows = read_timings_csv(csv_path)
    if not rows:
        print(f"El archivo {csv_path} no contiene datos.", file=sys.stderr)
        sys.exit(1)
    target_run = args.run_id or rows[0]["run_id"]
    sections = parse_sections_arg(args.sections)
    filtered = filter_rows(
        rows,
        run_id=target_run,
        epoch=args.epoch,
        batch_id=args.batch_id,
        sections=sections,
    )
    if not filtered:
        print("No hay filas que coincidan con los filtros. Se generaran graficas vacias.", file=sys.stderr)
    ensure_directory(REPORTS_DIR)

    top_n = max(0, int(args.top_n or 0))
    individual_rows = [r for r in filtered if r.get("individual_id", -1) >= 0]
    timeline_plot(
        individual_rows,
        label_key=lambda r: f"batch {r['batch_id']} / ind {r['individual_id']}",
        title=f"Timeline por individuo (run {target_run})",
        ylabel="Individuos",
        output=REPORTS_DIR / f"timeline_by_individual_{target_run}.png",
        top_n=top_n,
    )

    batch_rows = [r for r in filtered if r.get("batch_id", -1) >= 0]
    timeline_plot(
        batch_rows,
        label_key=lambda r: f"batch {r['batch_id']}",
        title=f"Timeline por batch (run {target_run})",
        ylabel="Batches",
        output=REPORTS_DIR / f"timeline_by_batch_{target_run}.png",
        top_n=top_n,
        aggregate=not args.batch_individual,
    )

    sim_rows = [
        r
        for r in filtered
        if r.get("section") in {"simulation_step", "lyapunov_compute"}
    ]

    def _sim_label(row: dict[str, Any]) -> str:
        if row["section"] == "simulation_step":
            return f"epoch {row.get('epoch', -1)} step {row.get('batch_id', -1)}"
        return f"epoch {row.get('epoch', -1)} {row['section']}"

    timeline_plot(
        sim_rows,
        label_key=_sim_label,
        title=f"Timeline de simulacion (run {target_run})",
        ylabel="Simulacion",
        output=REPORTS_DIR / f"timeline_simulation_{target_run}.png",
        top_n=top_n,
    )

    pie_chart(
        filtered,
        title=f"Distribucion por seccion (run {target_run})",
        output=REPORTS_DIR / f"pie_sections_{target_run}.png",
    )
    print("Graficas guardadas en", REPORTS_DIR)


if __name__ == "__main__":
    main()
