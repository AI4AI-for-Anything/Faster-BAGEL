#!/usr/bin/env python3
"""Render the Faster-BAGEL README figures from archived measurements; no GPU work.

Inputs: history.json (full-check results), time-budget.json (latest M6 profile) and validation.json (two rounds of
2000-step training runs).
Outputs: 01-benchmark.svg, 02-history.svg, 03-time-budget.svg, 04-validation.svg and ../change-views/module-map.svg.
Usage: python docs/performance-views/render.py   (needs matplotlib and numpy)
"""
from pathlib import Path
import json
import math
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent
RED = '#8C0000'
LIGHT = '#D7BDBF'
GREY = '#A99A9C'
MUTED = '#79696B'
INK = '#292122'
GRID = '#EADFE0'
CARD = '#FAF5F5'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                     'svg.hashsalt': 'faster-bagel-crimson', 'font.size': 12})


def save(fig, path):
    target = path.with_suffix('.svg')
    fig.savefig(target, facecolor='white', metadata={'Date': None})
    target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines()) + '\n')
    plt.close(fig)


def frame(ax, axis='y'):
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(length=0, labelsize=11.5, colors=MUTED, pad=9)
    ax.grid(axis=axis, color=GRID, linewidth=.7, zorder=0)
    ax.set_axisbelow(True)


def load_history():
    data = json.loads((ROOT / 'history.json').read_text())
    for row in data['rows']:
        if row['status'] == 'PASS':
            assert math.isclose(row['reference_s'] / row['candidate_s'], row['speedup'], abs_tol=2e-3), row
    return data


def hero():
    final = [r for r in load_history()['rows'] if r['status'] == 'PASS'][-1]
    values = [final['reference_s'] * 1e3, final['candidate_s'] * 1e3]
    fig = plt.figure(figsize=(9, 4.5), facecolor='white')
    fig.text(.06, .88, 'Faster-BAGEL, optimized with AI agents', color=RED, weight='bold', fontsize=19)
    fig.text(.06, .78, 'BAGEL-7B-MoT supervised fine-tuning · eight 80 GB Hopper GPUs · FP32 state, BF16 compute, no FP8',
             color=MUTED, fontsize=10.5)
    ax = fig.add_axes([.24, .24, .39, .45])
    ax.barh([1, 0], values, color=[LIGHT, RED], height=.24, zorder=3)
    ax.set_yticks([1, 0], ['Reference', 'Faster-BAGEL'])
    for y, value in zip([1, 0], values):
        ax.text(value + 25, y, f'{value:,.0f} ms', va='center', color=INK, fontsize=11.5, weight='bold')
    ax.set_xlim(0, 1950)
    ax.set_ylim(-.6, 1.6)
    ax.set_xticks([0, 400, 800, 1200, 1600])
    frame(ax, 'x')
    ax.tick_params(labelsize=10)
    ax.set_xlabel('Time per training step (ms) ↓', color=MUTED, labelpad=12, fontsize=11)
    fig.text(.85, .54, f"{final['speedup']:.2f}×", color=RED, fontsize=35, weight='bold', ha='center')
    fig.text(.85, .43, 'step speedup', color=MUTED, fontsize=12, ha='center')
    fig.text(.85, .36, 'full check: PASS', color=MUTED, fontsize=10, ha='center')
    fig.text(.06, .075, 'Reference → candidate → reference in independent processes on one host; reference = mean of '
             'the two reference runs.', color=MUTED, fontsize=9.5)
    fig.text(.06, .03, f"Two 100-step windows per run after warm-up · exact ratio {final['speedup']:.4f}× · numerically "
             'equivalent by the statistical gate (not bitwise).', color=MUTED, fontsize=9.5)
    save(fig, ROOT / '01-benchmark')


CARD_LABELS = {          # short card texts; the full change descriptions are in history.json and docs/training.md
    'M1': 'No LM recomputation, lean checkpoints, expert-ordered MoT stream, FSDP prefetch',
    'M2': 'Compiled elementwise ops and decoder regions, FSDP post-divide, fused AdamW',
    'M3': 'No outer recomputation; MLP recomputed inside compiled regions',
    'M4': 'ViT kept gathered, tuned FlexAttention, hand-written ViT backward, async VAE',
    'M5': 'Fewer host syncs, deferred logging synchronization',
    'M6': 'Reversible whole-document LM balancing',
}


def history():
    data = load_history()
    rows = [r for r in data['rows'] if r['status'] == 'PASS']
    rejected = [r for r in data['rows'] if r['status'] != 'PASS']
    height = 8.4
    top = lambda inches: 1 - inches / height       # figure fraction from inches below the top edge
    fig = plt.figure(figsize=(9, height), facecolor='white')
    fig.text(.09, top(.36), 'Faster-BAGEL', fontsize=11, weight='bold', color=RED)
    fig.text(.09, top(.79), 'Agent Optimization History', fontsize=21.5, weight='bold', color=INK)
    fig.text(.94, top(.65), f"{rows[-1]['speedup']:.2f}×", fontsize=29, ha='right', weight='bold', color=RED)
    fig.text(.94, top(.88), 'step speedup', fontsize=10.5, ha='right', color=MUTED)
    fig.text(.09, top(1.31), f"Cumulative full-check speedup · 8 Hopper GPUs · {len(rows)} certified milestones in "
             f"33 rounds · M7 rejected", fontsize=11, color=MUTED)
    ax = fig.add_axes([.09, top(4.25), .85, 2.67 / height])
    x = list(range(len(rows)))
    values = [row['speedup'] for row in rows]
    ax.plot(x, values, color=RED, linewidth=2.3, zorder=2)
    ax.plot(x, values, linestyle='none', marker='o', markersize=7, color=RED, zorder=3)
    for i, value in enumerate(values):
        ax.annotate(f'{value:.2f}×', (x[i], value), xytext=(0, 11 if i % 2 == 0 else -23),
                    textcoords='offset points', ha='center', fontsize=11,
                    color=RED if i == len(rows) - 1 else INK, weight='bold')
    labels = [r['milestone'] for r in rows]
    for r in rejected:
        xr = len(labels)
        ax.plot([x[-1], xr], [values[-1], r['speedup']], color=GREY, linewidth=1.3, linestyle=':', zorder=1)
        ax.plot([xr], [r['speedup']], marker='X', markersize=9, color=GREY, zorder=3)
        ax.annotate(f"{r['speedup']:.2f}× timed\nINCORRECT", (xr, r['speedup']), xytext=(0, 11), textcoords='offset points',
                    ha='center', fontsize=9.5, color=MUTED, weight='bold', linespacing=1.15)
        labels.append(f"{r['milestone']} ✕")
    ax.set_xticks(range(len(labels)), labels)
    ax.tick_params(axis='x', labelsize=10)
    ax.set_yticks([1.0, 1.5, 2.0])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:.1f}×'))
    ax.set_ylim(1.0, 2.15)
    ax.set_xlim(-.6, len(labels) - .4)
    frame(ax)
    for i, row in enumerate(rows):
        bx, by = .09 + (i % 2) * .44, top(5.2 + (i // 2) * .78)
        fig.add_artist(FancyBboxPatch((bx, by - .48 / height), .41, .66 / height,
                       boxstyle='round,pad=0.004,rounding_size=0.007',
                       transform=fig.transFigure, facecolor=CARD, edgecolor='none'))
        fig.text(bx + .014, by - .15 / height, row['milestone'], fontsize=11.5, weight='bold', color=RED, va='center')
        label = textwrap.fill(CARD_LABELS[row['milestone']], width=40, break_long_words=False, break_on_hyphens=False)
        fig.text(bx + .065, by - .15 / height, label, fontsize=9.6, color=INK, weight='bold', va='center',
                 linespacing=1.22)
    fig.text(.09, top(7.95), 'Each point is a full check by the same frozen judge on the same host; M1 is bitwise identical '
             'to the reference,\nM2–M6 passed the statistical gate. M7 timed faster but the gate found a real deviation, '
             'so it was rejected.\nCumulative configurations, not isolated ablations. Target was 3×.',
             fontsize=9.3, color=MUTED, linespacing=1.4)
    save(fig, ROOT / '02-history')


def time_budget():
    data = json.loads((ROOT / 'time-budget.json').read_text())
    final = [r for r in load_history()['rows'] if r['status'] == 'PASS'][-1]
    cats = data['categories']
    ref_ms, cand_ms = final['reference_s'] * 1e3, final['candidate_s'] * 1e3
    parts = [('GEMM', cats['gemm']['s_per_step']), ('Attention', cats['attention']['s_per_step']),
             ('Copy / index', cats['copy/index']['s_per_step']), ('Triton (compiled)', cats['triton']['s_per_step']),
             ('Elementwise / reduce', cats['elementwise/reduce']['s_per_step']), ('Other kernels', cats['other']['s_per_step']),
             ('NCCL not hidden by compute', data['nccl_exposed_s_per_step'])]
    fig = plt.figure(figsize=(9, 6.2), facecolor='white')
    fig.text(.06, .925, f"Why Faster-BAGEL stops at {final['speedup']:.2f}×", color=INK, weight='bold', fontsize=19)
    fig.text(.06, .868, 'M6 · profiled GPU time per step on rank 0 (4 steps under the profiler) vs the step-time budgets',
             color=MUTED, fontsize=10.2)
    ax = fig.add_axes([.33, .33, .6, .47])
    ys = np.arange(len(parts))[::-1]
    for y, (name, s) in zip(ys, parts):
        color = LIGHT if name.startswith('NCCL') else RED
        ax.barh(y, s * 1e3, height=.55, color=color, zorder=3)
        ax.text(s * 1e3 + 8, y, f'{s * 1e3:.0f}', va='center', fontsize=9.5, color=INK, zorder=5)
    ax.set_yticks(ys, [p[0] for p in parts])
    for xv, lab, ls in ((ref_ms / 3, '3× budget', '--'), (ref_ms / 2, '2× budget', ':')):
        ax.axvline(xv, color=MUTED, linewidth=1.1, linestyle=ls, zorder=4)
        ax.text(xv, len(parts) - .35, f'{lab}\n{xv:.0f} ms', ha='center', va='bottom', fontsize=9, color=MUTED)
    ax.set_xlim(0, 800)
    ax.set_ylim(-.6, len(parts) - .1)
    ax.set_xticks([0, 200, 400, 600, 800])
    frame(ax, 'x')
    ax.tick_params(axis='y', labelsize=10.5, colors=INK)
    ax.tick_params(axis='x', labelsize=10)
    ax.set_xlabel('ms per step on one GPU', color=MUTED, labelpad=8, fontsize=10.5)
    union = data['compute_union_s_per_step'] * 1e3
    fig.text(.06, .17, f"Compute alone (union of all kernels): {union:.0f} ms per step vs {ref_ms / 3:.0f} ms for 3× "
             f"(measured {cand_ms:.0f} ms; reference {ref_ms:.0f} ms).", fontsize=10, color=INK)
    fig.text(.06, .035, f"Packed tokens per rank are fixed by the recipe and uneven (heaviest/mean "
             f"{data['earlier_max_to_mean_packed_length']:.2f}); the forward is CPU-launch-bound;\nGEMMs run near "
             f"{data['earlier_gemm_peak_fraction'] * 100:.0f}% of the BF16 peak. An earlier estimate put the ceiling at about "
             f"{data['earlier_compute_ceiling_speedup']:.1f}× with these kernels.\nProfiled kernel sums overlap in time and "
             "include profiler overhead; they are estimates, not checks.", fontsize=9.3, color=MUTED, linespacing=1.4)
    save(fig, ROOT / '03-time-budget')


def validation():
    data = json.loads((ROOT / 'validation.json').read_text())
    curves, held, r2 = data['curves'], data['heldout'], data['round2']
    refs = sorted(a for a in curves if a.startswith('ref'))
    opts = sorted(a for a in curves if a.startswith('opt'))
    n = len(curves[refs[0]]['ce'])
    x = (np.arange(n) + 1) * (2000 // n)
    fig = plt.figure(figsize=(12, 9.6), facecolor='white')
    fig.text(.06, .955, f'Faster-BAGEL vs reference: {len(refs)} seeds × 2,000 steps each', color=INK, weight='bold',
             fontsize=18)
    fig.text(.06, .922, 'Official SFT recipe, eight 80 GB Hopper GPUs · training losses as 10-step means · held-out: '
             '100 VLM rows (8,407 tokens) with images disjoint from training', color=MUTED, fontsize=10.5)
    axs = [fig.add_axes(r) for r in ([.07, .54, .4, .32], [.57, .54, .4, .32], [.07, .11, .4, .32],
                                     [.57, .11, .4, .32])]

    def panel(ax, title):
        frame(ax, 'both')
        ax.spines['left'].set_visible(True)
        ax.spines['left'].set_color(GRID)
        ax.tick_params(labelsize=9.5, pad=5)
        ax.set_title(title, loc='left', color=INK, fontsize=11.5, weight='bold', pad=8)

    def band(ax, metric):
        for arms, color, lw, z, lab in ((refs, GREY, 2.4, 2, 'Reference'), (opts, RED, 1.1, 3, 'Faster-BAGEL')):
            m = np.array([curves[a][metric] for a in arms])
            for i, row in enumerate(m):
                ax.plot(x, row, color=color, linewidth=lw * .55, alpha=.55, zorder=z)
            ax.plot(x, m.mean(0), color=color, linewidth=lw, zorder=z + 1, label=f'{lab}, mean of {len(arms)} seeds')
        ax.set_yscale('log')
        ax.set_xlim(0, 2000)
        ax.set_xlabel('step', color=MUTED, fontsize=10)
        ax.legend(loc='upper right', frameon=False, fontsize=9, labelcolor=INK)

    band(axs[0], 'ce')
    panel(axs[0], 'Training CE (text), each seed and the mean')
    band(axs[1], 'mse')
    panel(axs[1], 'Training MSE (flow matching), each seed and the mean')

    ax = axs[2]
    wx = np.arange(50, 2000, 100)
    ax.axhspan(-1, 1, color=GRID, alpha=.6, linewidth=0, zorder=1, label='|D| ≤ δ: consistent')
    ax.axhline(0, color=MUTED, linewidth=.8, zorder=2)
    styles = {'ce': dict(color=RED, linestyle='-'), 'mse': dict(color=RED, linestyle='--'),
              'log_grad_norm': dict(color=GREY, linestyle='-')}
    names = {'ce': 'CE', 'mse': 'MSE', 'log_grad_norm': 'log gradient norm'}
    for metric, st in styles.items():
        w = r2['training'][metric]['windows']
        ratio = [win['D'] / win['delta'] for win in w]
        ax.plot(wx, ratio, marker='o', markersize=4, linewidth=1.5, zorder=3, label=f"{names[metric]} "
                f"({len(w) - r2['training'][metric]['failed_windows']}/{len(w)})", **st)
    lim = max(1.9, max(abs(w['D'] / w['delta']) for m in styles for w in r2['training'][m]['windows']) + .4)
    ax.set_xlim(0, 2000)
    ax.set_ylim(-lim, lim)
    panel(ax, 'Pooled difference per 100-step window, D / δ (δ = 2σ of the reference seeds)')
    ax.title.set_fontsize(10.5)
    ax.set_xlabel('step', color=MUTED, fontsize=10)
    ax.legend(loc='upper left', frameon=False, fontsize=8.6, labelcolor=INK, ncol=2, columnspacing=1.2)

    axs[3].remove()
    seeds = list(range(len(refs)))
    for k, (key, title, rect) in enumerate((('model', 'Held-out CE, final model', [.585, .11, .16, .32]),
                                            ('ema', 'Held-out CE, final EMA', [.815, .11, .16, .32]))):
        ax = fig.add_axes(rect)
        rv = [held[a][key] for a in refs]
        ov = [held[a][key] for a in opts]
        st = r2['heldout'][key]
        ax.axhspan(st['reference_mean'] - st['delta'], st['reference_mean'] + st['delta'], color=GRID, alpha=.6,
                   linewidth=0, zorder=1)
        ax.axhline(st['reference_mean'], color=GREY, linewidth=1.2, zorder=2)
        ax.axhline(st['candidate_mean'], color=RED, linewidth=1.4, zorder=2)
        ax.scatter([s - .12 for s in seeds], rv, s=55, color=GREY, zorder=3, label='Reference')
        ax.scatter([s + .12 for s in seeds], ov, s=32, color=RED, marker='D', zorder=4, label='Faster-BAGEL')
        ax.set_xticks(seeds, [f's{i + 1}' for i in seeds])
        ax.set_xlim(-.6, len(seeds) - .4)
        panel(ax, title)
        ax.title.set_fontsize(10.5)
        ax.text(.5, -.2, f"D = {st['D']:+.4f}   δ = {st['delta']:.4f}", transform=ax.transAxes, ha='center',
                fontsize=9.2, color=INK)
        if k == 0:
            fig.legend(handles=ax.get_legend_handles_labels()[0], labels=['Reference seed', 'Faster-BAGEL seed'],
                       loc=(.585, .455), ncol=2, frameon=False, fontsize=8.8, labelcolor=INK)
    t = r2['training']
    fig.text(.06, .008, f"Round 2 (four seeds, criterion fixed before the runs): consistent windows CE "
             f"{20 - t['ce']['failed_windows']}/20, MSE {20 - t['mse']['failed_windows']}/20, log gradient norm "
             f"{20 - t['log_grad_norm']['failed_windows']}/20, all final windows and both held-out values within δ: "
             "CONSISTENT.\nRound 1 (two seeds, |E| ≤ |Y| with a two-seed yardstick) was NOT CONSISTENT and is kept as "
             "recorded. No run was repeated; all values finite.", fontsize=9.3, color=MUTED, linespacing=1.4)
    save(fig, ROOT / '04-validation')


def modulemap():
    fig = plt.figure(figsize=(9, 6.8), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.text(.055, .925, 'What changed in Faster-BAGEL', fontsize=23, color=INK, weight='bold')
    cards = [
        ('Memory and recomputation', 'No outer activation recomputation\nLean checkpoints for norms and SwiGLU,\n'
         'MLP recompute in compiled regions',
         'train/fsdp_utils.py\nmodeling/bagel/qwen2_navit.py\nmodeling/bagel/fused_layer.py', '3 files'),
        ('Mixture-of-transformers layers', 'Residual stream kept in expert order\nCompiled elementwise ops and\n'
         'two-region decoder layers · tuned Flex',
         'modeling/bagel/qwen2_navit.py\nmodeling/bagel/fused_ops.py\nmodeling/bagel/fused_layer.py', '3 files'),
        ('FSDP and balancing', 'Forward prefetch · one post-divide\nViT kept gathered for backward\n'
         'Fused AdamW · whole-document LM balancing',
         'train/fsdp_utils.py\nmodeling/bagel/lm_balance.py\nmodeling/bagel/bagel.py', '3 files'),
        ('Host path and ViT', 'Hand-written SigLIP layer backward\nVAE encode on a side stream\n'
         'Deferred metric syncs, fewer host syncs',
         'modeling/bagel/: vit_layer_fn.py · siglip_navit.py\nmodeling/autoencoder.py · train/async_log.py\n'
         'data/: data_utils.py · dataset_base.py', '7 files'),
    ]
    for i, (title, description, paths, count) in enumerate(cards):
        x, y = .055 + (i % 2) * .455, .475 - (i // 2) * .42
        ax.add_patch(FancyBboxPatch((x, y), .425, .37,
                     boxstyle='round,pad=0.009,rounding_size=0.012', facecolor=CARD, edgecolor=GRID))
        ax.plot([x, x + .425], [y + .37, y + .37], color=RED, linewidth=2, solid_capstyle='round')
        ax.text(x + .014, y + .33, f'{i + 1:02d}', fontsize=11, color=RED, weight='bold')
        ax.text(x + .41, y + .33, count, fontsize=10, color=MUTED, ha='right')
        ax.text(x + .014, y + .284, title, fontsize=12.5, color=INK, weight='bold')
        ax.text(x + .014, y + .232, description, fontsize=10.2, color=INK, va='top', linespacing=1.6)
        ax.text(x + .014, y + .03, paths, fontsize=9.2, color=MUTED, linespacing=1.5)
    ax.text(.055, .02, 'Every switch is off by default; directions that were tried and rejected stay in the code, '
            'switched off.', fontsize=9.5, color=MUTED)
    save(fig, ROOT.parent / 'change-views/module-map')


if __name__ == '__main__':
    hero()
    history()
    time_budget()
    validation()
    modulemap()
