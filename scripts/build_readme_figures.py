#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

# Stabilize PDF metadata so repeated builds produce identical artifacts.
os.environ.setdefault("SOURCE_DATE_EPOCH", "1788134400")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
IMAGES = DATA / "images"

BUDGETS = (64, 128, 256, 512, 1024, 2048)
FULLKV_SCORE = 41.92

GROUPS = (
    ("Shared-token budget", ("H2O", "SnapKV", "SurrogateKV-Snap")),
    ("Layer-adaptive budget", ("PyramidKV", "DynamicKV", "SurrogateKV-Dynamic")),
    ("Head-aware budget", ("Ada-KV", "SurrogateKV-Ada")),
)

LIGHT_COLORS = {
    "H2O": "#477FC1",
    "SnapKV": "#078A96",
    "SurrogateKV-Snap": "#7355C6",
    "PyramidKV": "#C88A00",
    "DynamicKV": "#D45E36",
    "SurrogateKV-Dynamic": "#7656C5",
    "Ada-KV": "#C65391",
    "SurrogateKV-Ada": "#56349A",
}

DARK_COLORS = {
    "H2O": "#58A6FF",
    "SnapKV": "#39C5CF",
    "SurrogateKV-Snap": "#A78BFA",
    "PyramidKV": "#F2CC60",
    "DynamicKV": "#FF8A65",
    "SurrogateKV-Dynamic": "#C4A7FF",
    "Ada-KV": "#F778BA",
    "SurrogateKV-Ada": "#BC8CFF",
}

MARKERS = {
    "H2O": "o",
    "SnapKV": "s",
    "SurrogateKV-Snap": "D",
    "PyramidKV": "o",
    "DynamicKV": "s",
    "SurrogateKV-Dynamic": "D",
    "Ada-KV": "s",
    "SurrogateKV-Ada": "D",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def theme(name: str) -> dict[str, str]:
    if name == "dark":
        return {
            "text": "#E6EDF3",
            "axis": "#8B949E",
            "grid": "#30363D",
            "fullkv": "#B1BAC4",
        }
    return {
        "text": "#24292F",
        "axis": "#57606A",
        "grid": "#D8DEE4",
        "fullkv": "#57606A",
    }


def configure_matplotlib(colors: dict[str, str]) -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.labelcolor": colors["text"],
            "axes.titlecolor": colors["text"],
            "axes.edgecolor": colors["axis"],
            "xtick.color": colors["text"],
            "ytick.color": colors["text"],
            "text.color": colors["text"],
            "svg.fonttype": "none",
            "svg.hashsalt": "surrogatekv-readme",
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(
        path,
        bbox_inches="tight",
        pad_inches=0.045,
        transparent=True,
        metadata={"Date": None},
    )
    if path.suffix == ".svg":
        source = path.read_text(encoding="utf-8")
        path.write_text("\n".join(line.rstrip() for line in source.splitlines()) + "\n", encoding="utf-8")


def build_longbench(theme_name: str) -> None:
    colors = theme(theme_name)
    palette = DARK_COLORS if theme_name == "dark" else LIGHT_COLORS
    configure_matplotlib(colors)

    rows = read_csv(DATA / "longbench/llama3_8b_instruct/budget_scores.csv")
    scores = {(row["method"], int(row["budget"])): float(row["average"]) for row in rows}

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.15), sharex=True, sharey=True)
    fig.patch.set_alpha(0.0)
    x = np.asarray(BUDGETS, dtype=float)

    for group_index, (title, methods) in enumerate(GROUPS):
        ax = axes[group_index]
        ax.patch.set_alpha(0.0)
        ax.axhline(
            FULLKV_SCORE,
            color=colors["fullkv"],
            linewidth=1.35,
            linestyle=(0, (5, 3)),
            label="FullKV",
            zorder=1,
        )
        for method in methods:
            is_surrogate = method.startswith("SurrogateKV")
            ax.plot(
                x,
                [scores[(method, budget)] for budget in BUDGETS],
                color=palette[method],
                linewidth=2.25 if is_surrogate else 1.65,
                marker=MARKERS[method],
                markersize=5.0 if is_surrogate else 4.2,
                markeredgewidth=0.0,
                label=method,
                zorder=3 if is_surrogate else 2,
            )

        ax.set_title(title, fontsize=10.2, pad=7.0, fontweight="semibold")
        ax.set_xticks((0, 512, 1024, 1536, 2048), labels=("0", "512", "1K", "1.5K", "2K"))
        ax.set_xlabel("KV cache budget")
        ax.set_xlim(0, 2112)
        ax.set_ylim(32.5, 42.35)
        ax.set_yticks((34, 36, 38, 40, 42))
        ax.grid(axis="y", color=colors["grid"], linewidth=0.65, alpha=0.9)
        ax.tick_params(axis="both", length=3.0, width=0.75, pad=3.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(colors["axis"])
        ax.spines["bottom"].set_color(colors["axis"])
        ax.legend(
            loc="lower right",
            frameon=False,
            fontsize=7.7,
            handlelength=1.55,
            handletextpad=0.45,
            borderaxespad=0.25,
            labelspacing=0.35,
        )

    axes[0].set_ylabel("LongBench score")
    fig.subplots_adjust(left=0.065, right=0.995, top=0.90, bottom=0.18, wspace=0.13)

    suffix = "-dark" if theme_name == "dark" else ""
    save_figure(fig, IMAGES / f"longbench_budget_sweep{suffix}.svg")
    if theme_name == "light":
        save_figure(fig, IMAGES / "longbench_budget_sweep.pdf")
    plt.close(fig)


def read_heatmap(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = read_csv(path)
    context_lengths = np.asarray([int(value) for value in rows[0] if value != "depth_percent"])
    depths = np.asarray([float(row["depth_percent"]) for row in rows])
    values = np.asarray(
        [[float(row[str(context_length)]) for context_length in context_lengths] for row in rows],
        dtype=float,
    )
    return values, depths, context_lengths


def build_niah(theme_name: str) -> None:
    colors = theme(theme_name)
    configure_matplotlib(colors)
    directory = DATA / "niah/mistral_7b_instruct_v02/k128_ctx1000_32000_step200"
    averages = {row["method"]: float(row["average"]) for row in read_csv(directory / "niah_average_table.csv")}
    methods = (
        ("SnapKV", "snapkv"),
        ("SurrogateKV-Snap", "surrogatekv-snap"),
        ("DynamicKV", "dynamickv"),
        ("SurrogateKV-Dynamic", "surrogatekv-dynamic"),
        ("Ada-KV", "adakv"),
        ("SurrogateKV-Ada", "surrogatekv-ada"),
    )

    fig = plt.figure(figsize=(9.4, 6.65))
    fig.patch.set_alpha(0.0)
    grid = fig.add_gridspec(
        3,
        3,
        width_ratios=(1.0, 1.0, 0.035),
        left=0.075,
        right=0.96,
        top=0.96,
        bottom=0.08,
        hspace=0.43,
        wspace=0.14,
    )

    image = None
    for index, (method, slug) in enumerate(methods):
        row, column = divmod(index, 2)
        ax = fig.add_subplot(grid[row, column])
        ax.patch.set_alpha(0.0)
        values, depths, context_lengths = read_heatmap(directory / f"niah_heatmap_{slug}.csv")
        image = ax.imshow(
            values,
            cmap="RdYlGn",
            vmin=0.0,
            vmax=100.0,
            interpolation="nearest",
            aspect="auto",
            rasterized=True,
        )
        ax.set_title(f"{method}\nAvg. {averages[method]:.2f}", fontsize=9.4, pad=5.0, linespacing=1.10)
        x_values = (1000, 8000, 16000, 24000, 32000)
        x_positions = [int(np.abs(context_lengths - value).argmin()) for value in x_values]
        ax.set_xticks(x_positions, labels=("1K", "8K", "16K", "24K", "32K"))
        y_values = (0, 22, 44, 67, 89, 100)
        y_positions = [int(np.abs(depths - value).argmin()) for value in y_values]
        ax.set_yticks(y_positions, labels=tuple(str(value) for value in y_values))
        if row == 2:
            ax.set_xlabel("Context length (tokens)", labelpad=4.0)
        if column == 0:
            ax.set_ylabel("Needle depth (%)", labelpad=4.0)
        ax.tick_params(axis="both", length=2.8, width=0.7, pad=2.3, labelsize=8.2)
        for spine in ax.spines.values():
            spine.set_color(colors["axis"])
            spine.set_linewidth(0.75)

    if image is None:
        raise RuntimeError("No NIAH heatmaps were loaded")
    cbar_ax = fig.add_subplot(grid[:, 2])
    cbar = fig.colorbar(image, cax=cbar_ax)
    cbar.set_ticks((0, 50, 100))
    cbar.set_label("Retrieval accuracy", labelpad=5.0)
    cbar.ax.tick_params(labelsize=8.2, width=0.7, length=2.8, pad=2.0)
    cbar.outline.set_edgecolor(colors["axis"])

    suffix = "-dark" if theme_name == "dark" else ""
    save_figure(fig, IMAGES / f"mistral_niah_k128_method_comparison{suffix}.svg")
    if theme_name == "light":
        save_figure(fig, IMAGES / "mistral_niah_k128_method_comparison.pdf")
    plt.close(fig)


def build_overview_themes() -> None:
    light_path = IMAGES / "surrogatekv_overview.svg"
    source = light_path.read_text(encoding="utf-8")

    # pdftocairo emits one page-sized white path before the diagram. Removing
    # only that path preserves the diagram panels while making the canvas
    # transparent.
    page_pattern = re.compile(
        r'(<path fill-rule="nonzero" )fill="rgb\(100%, 100%, 100%\)"'
        r'( fill-opacity="1" d="M 0\.105469 0 L 1092\.601562 0 L 1092\.601562 344\.167969)'
    )
    light = page_pattern.sub(r'\1fill="none"\2', source, count=1)
    light_path.write_text(light, encoding="utf-8")

    dark = light
    dark = dark.replace(
        '<path fill-rule="nonzero" fill="rgb(100%, 100%, 100%)"',
        '<path fill-rule="nonzero" fill="rgb(8.24%, 10.59%, 14.12%)"',
    )
    dark = dark.replace('fill="rgb(0%, 0%, 0%)"', 'fill="rgb(90.20%, 92.94%, 95.29%)"')
    dark = dark.replace(
        'stroke="rgb(6.669617%, 6.669617%, 6.669617%)"',
        'stroke="rgb(69.41%, 72.94%, 76.86%)"',
    )
    dark = dark.replace(
        'stroke="rgb(41.958618%, 44.709778%, 50.19989%)"',
        'stroke="rgb(69.41%, 72.94%, 76.86%)"',
    )
    (IMAGES / "surrogatekv_overview-dark.svg").write_text(dark, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the theme-aware README figures from released data.")
    parser.add_argument(
        "--skip-overview",
        action="store_true",
        help="Do not derive the dark overview SVG from the archival overview.",
    )
    args = parser.parse_args()

    IMAGES.mkdir(parents=True, exist_ok=True)
    build_longbench("light")
    build_longbench("dark")
    build_niah("light")
    build_niah("dark")
    if not args.skip_overview:
        build_overview_themes()


if __name__ == "__main__":
    main()
