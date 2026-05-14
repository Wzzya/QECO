"""
Compare delay logs: RandomMode, project root Delay.txt, and
UltraPowerSavingMode (paper-style smoothing + grid).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load_delay(path: Path) -> np.ndarray:
    vals = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals.append(float(line))
    return np.array(vals, dtype=float)


def _moving_average(y, window=31):
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 3:
        return y.copy()
    w = max(3, min(int(window), n))
    kernel = np.ones(w, dtype=float) / w
    return np.convolve(y, kernel, mode="same")


def _style_paper_axis(ax):
    ax.grid(True, linestyle="--", color="0.3", linewidth=0.85, alpha=0.9)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)
    ax.tick_params(axis="both", colors="black")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Microsoft YaHei"],
            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.frameon": True,
            "legend.edgecolor": "black",
            "legend.facecolor": "white",
        }
    )

    p_random = ROOT / "TrainedModel_20UE_2EN_RandomMode" / "Delay 06.16.34.txt"
    p_root = ROOT / "Delay.txt"
    p_ultra = ROOT / "TrainedModel_20UE_2EN_UltraPowerSavingMode" / "Delay.txt"

    a = load_delay(p_random)
    b = load_delay(p_root)
    c = load_delay(p_ultra)
    n = min(len(a), len(b), len(c))
    if n == 0:
        raise SystemExit("No numeric rows in one or more files.")

    ma = max(7, min(51, n // 15 or 7))
    a_s = _moving_average(a[:n], ma)
    b_s = _moving_average(b[:n], ma)
    c_s = _moving_average(c[:n], ma)
    diff_random_s = _moving_average(a[:n] - b[:n], ma)
    diff_ultra_s = _moving_average(c[:n] - b[:n], ma)

    x = np.arange(n)
    out_dir = Path(__file__).resolve().parent
    out_png = out_dir / "delay_compare.png"

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(
        x,
        a_s,
        linestyle="-",
        linewidth=2.0,
        color="#1f4e79",
        label="RandomMode (Delay 06.16.34.txt)",
    )
    axes[0].plot(
        x,
        b_s,
        linestyle="-",
        linewidth=2.0,
        color="#c0504d",
        label="Delay.txt (project root)",
    )
    axes[0].plot(
        x,
        c_s,
        linestyle="-",
        linewidth=2.0,
        color="#556b2f",
        label="UltraPowerSavingMode / Delay.txt",
    )
    axes[0].set_ylabel("Delay")
    axes[0].set_title("Delay overlay (first {} aligned steps, moving avg. window={})".format(n, ma))
    _style_paper_axis(axes[0])
    axes[0].legend(loc="lower right", fontsize=10, fancybox=False)

    axes[1].plot(
        x,
        diff_random_s,
        color="#2e7d32",
        linewidth=2.0,
        linestyle="-",
        label="Random − Root",
    )
    axes[1].plot(
        x,
        diff_ultra_s,
        color="#7c3aed",
        linewidth=2.0,
        linestyle="-",
        label="UltraPowerSaving − Root",
    )
    axes[1].axhline(0, color="black", linewidth=0.9)
    axes[1].set_ylabel("Δ vs project root")
    axes[1].set_xlabel("Index")
    axes[1].set_title("Difference vs project root Delay.txt (smoothed)")
    _style_paper_axis(axes[1])
    axes[1].legend(loc="lower right", fontsize=10, fancybox=False)

    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print("Wrote:", out_png)
    print(
        "RandomMode:",
        len(a),
        " Root:",
        len(b),
        " UltraPowerSaving:",
        len(c),
        " Plotted (aligned):",
        n,
    )


if __name__ == "__main__":
    main()
