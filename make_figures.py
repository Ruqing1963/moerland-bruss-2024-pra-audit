#!/usr/bin/env python3
"""
make_figures.py -- generates the three figures of
"Entanglement generation by unequal-dimension Bell-diagonal projections ..."

  fig_shift_mismatch.pdf   local shift S versus block-cyclic shift C on the product-basis grid
  fig_mechanism.pdf        populations of Pi_Bell(rho_star) and the pairs coupled by partial transposition
  fig_negativity.pdf       negativity of Pi_Bell(product state): analytic family versus numerical optimum

Figure 2 is computed from the exact projection; Figure 3 reads the table printed by
verify_unequal_bell_projection.py (file verify_output.txt in the same directory).
Requires numpy and matplotlib.   Run:  python make_figures.py
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE = Path(__file__).resolve().parent
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#6b6a66", "#d9d8d2"
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
    "axes.linewidth": 0.6, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})


def draw_grid(ax, dA, dB, fill, text=None, title=""):
    """Rows = label i of A (top to bottom), columns = label j of B."""
    for i in range(dA):
        for j in range(dB):
            ax.add_patch(Rectangle((j, dA - 1 - i), 1, 1, facecolor=fill(i, j), edgecolor="white", linewidth=1.6))
            if text is not None:
                s, col = text(i, j)
                ax.text(j + 0.5, dA - 1 - i + 0.5, s, ha="center", va="center", fontsize=8, color=col)
    for j in range(dB):
        ax.text(j + 0.5, dA + 0.18, f"${j}$", ha="center", va="bottom", fontsize=8, color=MUTED)
    for i in range(dA):
        ax.text(-0.18, dA - 1 - i + 0.5, f"${i}$", ha="right", va="center", fontsize=8, color=MUTED)
    ax.text(dB / 2, dA + 0.62, r"label $j$ of $B$", ha="center", va="bottom", fontsize=8, color=MUTED)
    ax.text(-0.62, dA / 2, r"label $i$ of $A$", ha="center", va="center", rotation=90, fontsize=8, color=MUTED)
    ax.set_xlim(-0.9, dB + 0.1)
    ax.set_ylim(-0.75, dA + 1.05)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(dB / 2, -0.28, title, ha="center", va="top", fontsize=9, color=INK)


def centre(i, j, dA):
    return (j + 0.5, dA - 1 - i + 0.5)


def arrow(ax, p, q, color, rad=0.0, ls="-", lw=1.4):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, color=color, lw=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=7, shrinkB=7, zorder=5))


# ------------------------------------------------------------------------------------------
# Figure 1: S versus C
# ------------------------------------------------------------------------------------------
def figure_shift():
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.75), gridspec_kw={"width_ratios": [3.9, 5.9]})
    for ax, (dA, dB), tag in zip(axes, [(3, 3), (3, 5)], ["(a)", "(b)"]):
        def fill(i, j, dB=dB):
            return "#cfe1f7" if (j - i) % dB == 0 else "#f1f0ec"

        def text(i, j, dB=dB):
            b = (j - i) % dB
            return (rf"$V_{{{b}}}$", INK if b == 0 else MUTED)

        title = rf"{tag} $d_A=d_B=3$: $C=X_A\otimes X_B$" if dA == dB else \
            rf"{tag} $d_A=3<d_B=5$: $X_A\otimes X_B$ leaves $V_0$"
        draw_grid(ax, dA, dB, fill, text, title)
        for k in range(dA - 1):                                   # common steps of S and C
            arrow(ax, centre(k, k, dA), centre(k + 1, k + 1, dA), BLUE)
        last = centre(dA - 1, dA - 1, dA)
        arrow(ax, last, centre(0, 0, dA), BLUE, rad=-0.45)        # wrap-around of C
        if dA < dB:
            arrow(ax, last, centre(0, dA, dA), ORANGE, rad=0.25, ls=(0, (4, 2)))   # wrap-around of S
    axes[1].plot([], [], color=BLUE, lw=1.4, label=r"block-cyclic shift $C$")
    axes[1].plot([], [], color=ORANGE, lw=1.4, ls=(0, (4, 2)), label=r"local shift $S=X_A\otimes X_B$")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False, fontsize=8,
                   handlelength=2.2, columnspacing=1.6)
    fig.subplots_adjust(wspace=0.08)
    fig.savefig(HERE / "fig_shift_mismatch.pdf")
    plt.close(fig)


# ------------------------------------------------------------------------------------------
# Figure 2: mechanism behind the negative eigenvalue
# ------------------------------------------------------------------------------------------
def bell_basis(dA, dB):
    N = dA * dB
    w = np.exp(2j * np.pi / dA)
    B = np.zeros((N, N), dtype=complex)
    for a in range(dA):
        for b in range(dB):
            for i in range(dA):
                B[i * dB + (i + b) % dB, a * dB + b] = w ** (a * i) / np.sqrt(dA)
    return B


def projected_family_state(dA, dB):
    a = np.zeros(dA); a[:2] = 1 / np.sqrt(2)
    b = np.zeros(dB); b[:2] = 1 / np.sqrt(2)
    psi = np.kron(a, b)
    B = bell_basis(dA, dB)
    p = np.abs(B.conj().T @ psi) ** 2
    return ((B * p) @ B.conj().T).real


def figure_mechanism():
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.95), gridspec_kw={"width_ratios": [3.9, 5.9]})
    shades = {0: "#ffffff", 1: "#bcd6f5", 2: "#6ea6e6"}
    for ax, (dA, dB), tag in zip(axes, [(3, 3), (3, 5)], ["(a)", "(b)"]):
        sig = projected_family_state(dA, dB)
        N = dA * dB
        PT = sig.reshape(dA, dB, dA, dB).transpose(2, 1, 0, 3).reshape(N, N)
        lam = np.linalg.eigvalsh(PT)[0]
        unit = 1 / (4 * dA)
        w = np.rint(np.diag(sig).reshape(dA, dB) / unit).astype(int)     # diagonal weights in units 1/(4 dA)

        def fill(i, j, w=w):
            return shades[w[i, j]]

        def text(i, j, w=w):
            return (str(w[i, j]) if w[i, j] else "0", INK if w[i, j] else MUTED)

        lam_txt = r"$\lambda_{\min}=0$" if abs(lam) < 1e-12 else r"$\lambda_{\min}=-1/(4d_A)$"
        dims = r"$d_A=d_B=3$" if dA == dB else r"$d_A=3<d_B=5$"
        draw_grid(ax, dA, dB, fill, text, rf"{tag} {dims}: {lam_txt}")
        for i in range(dA):                                               # cell frame for empty cells
            for j in range(dB):
                ax.add_patch(Rectangle((j + 0.02, dA - 1 - i + 0.02), 0.96, 0.96, fill=False, edgecolor=GRID, lw=0.6))
        for k in range(dA):                                               # pairs coupled by the partial transpose
            kp = (k + 1) % dA
            p, q = centre(k, kp, dA), centre(kp, k, dA)
            wrap = k == dA - 1
            col = ORANGE if (wrap and dA < dB) else MUTED
            ax.add_patch(FancyArrowPatch(p, q, arrowstyle="<|-|>", mutation_scale=8, color=col, lw=1.5 if wrap else 1.0,
                                         connectionstyle=f"arc3,rad={0.3 if wrap else 0.0}", shrinkA=8, shrinkB=8, zorder=5))
    axes[1].plot([], [], color=MUTED, lw=1.0, label="coupled pair with diagonal weight (compensated)")
    axes[1].plot([], [], color=ORANGE, lw=1.5, label="wrap-around pair on empty cells")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.32, 0.02), ncol=1, frameon=False, fontsize=8, handlelength=2.2)
    fig.subplots_adjust(wspace=0.08)
    fig.savefig(HERE / "fig_mechanism.pdf")
    plt.close(fig)


# ------------------------------------------------------------------------------------------
# Figure 3: negativity table
# ------------------------------------------------------------------------------------------
def figure_negativity():
    rows = []
    for line in (HERE / "verify_output.txt").read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.match(r"\s+(\d) x (\d):\s+N_max >= ([0-9.]+)\s+family value 1/\(4dA\) = ([0-9.]+)", line)
        if m:
            rows.append((int(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4))))
    assert len(rows) == 8, "run verify_unequal_bell_projection.py > verify_output.txt first"
    x = np.arange(len(rows))
    fam = np.array([r[3] for r in rows])
    num = np.array([r[2] for r in rows])
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    bw = 0.36
    ax.bar(x - bw / 2 - 0.02, fam, bw, color=BLUE, label=r"analytic family $\rho_\star$:  $1/(4d_A)$")
    ax.bar(x + bw / 2 + 0.02, num, bw, color=ORANGE, label="best product state found numerically")
    for xi, v in zip(x, num):
        ax.text(xi + bw / 2 + 0.02, v + 0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([rf"${a}\times{b}$" for a, b, _, _ in rows])
    ax.set_xlabel(r"local dimensions $d_A\times d_B$")
    ax.set_ylabel(r"negativity of $\Pi_{\rm Bell}(\rho)$")
    ax.set_ylim(0, 0.43)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=3, width=0.6)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    fig.savefig(HERE / "fig_negativity.pdf")
    plt.close(fig)


if __name__ == "__main__":
    figure_shift()
    figure_mechanism()
    figure_negativity()
    print("wrote fig_shift_mismatch.pdf, fig_mechanism.pdf, fig_negativity.pdf")
