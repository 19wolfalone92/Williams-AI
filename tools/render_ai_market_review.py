"""Render exact pending chart lessons for private human review (matplotlib).

Run from the checkout with the primary Python runtime if plotting libraries
are unavailable in the default Python. No wave labels or future bars.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from williams_ai.chart_supervision import checked_chart_context  # noqa: E402
from williams_ai.chart_reader import indicators  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Private chart plots without model answers")
    parser.add_argument("queue", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    args.output.mkdir(parents=True, exist_ok=True)
    index = []
    for text in args.queue.read_text(encoding="utf-8").splitlines():
        lesson = json.loads(text)
        if lesson["review_status"] != "PENDING_INDEPENDENT_REVIEW":
            raise ValueError("Only unlabelled pending charts may enter a blind packet")
        ctx = lesson["market_context"]
        candles = checked_chart_context(ctx)["candles"]
        lines = indicators(candles)
        start = max(0, len(candles) - 140)
        xs = list(range(start, len(candles)))
        fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1, 1]})
        fig.patch.set_facecolor("#f5f7fb")
        for ax in axes:
            ax.set_facecolor("#ffffff")
            ax.grid(alpha=0.18)
            ax.spines[["top", "right"]].set_visible(False)
        for i in xs:
            bar = candles[i]
            color = "#168a6c" if bar.close >= bar.open else "#d25660"
            axes[0].vlines(i, bar.low, bar.high, color=color, linewidth=0.8)
            height = max(abs(bar.close - bar.open), ctx["tick_size"] / 2)
            axes[0].add_patch(Rectangle((i - 0.3, min(bar.open, bar.close)),
                                       0.6, height, color=color, linewidth=0))
        for key, color in (("jaw", "#4377c5"), ("teeth", "#d45f70"), ("lips", "#1b9d83")):
            axes[0].plot(xs, [lines[key][i] for i in xs], color=color,
                         linewidth=1.1, label=key)
        axes[0].legend(loc="upper left", ncol=3, frameon=False)
        axes[0].set_ylabel("Цена")
        for ax, key in zip(axes[1:], ("ao", "ac")):
            colors = ["#168a6c" if lines[key][i] >= lines[key][i - 1] else "#d25660" for i in xs]
            ax.bar(xs, [lines[key][i] for i in xs], color=colors, width=0.8)
            ax.axhline(0, color="#738096", linewidth=0.7)
            ax.set_ylabel(key.upper())
        ticks = xs[::24]
        axes[-1].set_xticks(ticks, [candles[i].at.strftime("%d.%m\n%H:%M") for i in ticks])
        axes[-1].set_xlabel("Закрытие свечи, UTC")
        fig.suptitle(f"{ctx['symbol']} · {ctx['timeframe']} · {ctx['venue']}\n"
                     f"{ctx['cutoff']}  |  {lesson['edition']}  |  PDF {lesson['pdf_page']}",
                     fontsize=15, fontweight="bold")
        fig.text(0.05, 0.025, f"{lesson['case_id']}\n"
                 "Видны последние 140 баров. Полный закрытый префикс — в JSONL. "
                 "Волновых ответов и будущих свечей нет.", fontsize=9, color="#44516a")
        fig.tight_layout(rect=(0.02, 0.065, 0.99, 0.94))
        filename = f"{lesson['case_id']}.png"
        fig.savefig(args.output / filename, dpi=100)
        plt.close(fig)
        index.append({"case_id": lesson["case_id"], "file": filename,
                      "market_sha256": ctx["market_sha256"]})
    (args.output / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(json.dumps({"rendered": len(index), "wave_answers_supplied": False}))


if __name__ == "__main__":
    main()
