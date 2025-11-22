#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# optional: use corner if available, otherwise fall back
try:
    import corner
    HAVE_CORNER = True
except Exception:
    HAVE_CORNER = False


def summarize(name, arr):
    m = float(np.mean(arr)); s = float(np.std(arr))
    q16, q50, q84 = [float(np.quantile(arr, q)) for q in (0.16, 0.50, 0.84)]
    return f"{name:>14s}: mean={m: .4f}  std={s: .4f}  16/50/84%: {q16:.4f}/{q50:.4f}/{q84:.4f}"


def load_params(npz_path: Path):
    """
    Returns (samples_matrix, labels_list) with columns in:
      m0, b, alpha, beta, log10_Nphi, log10_Lstar
    Automatically converts natural logs (log_*) to base-10 for plotting.
    """
    d = np.load(npz_path)
    ln10 = np.log(10.0)

    # Required (if missing, raise)
    req_keys = ["m0", "b", "alpha", "beta"]
    for k in req_keys:
        if k not in d:
            raise KeyError(f"Missing '{k}' in {npz_path.name}")

    m0    = d["m0"]
    b     = d["b"]
    alpha = d["alpha"]
    beta  = d["beta"]

    # Handle log params in either base
    if "log10_Nphi" in d:
        log10_Nphi = d["log10_Nphi"]
    elif "log_Nphi" in d:
        log10_Nphi = d["log_Nphi"] / ln10
    else:
        raise KeyError(f"Missing log10_Nphi/log_Nphi in {npz_path.name}")

    if "log10_Lstar" in d:
        log10_Lstar = d["log10_Lstar"]
    elif "log_Lstar" in d:
        log10_Lstar = d["log_Lstar"] / ln10
    else:
        raise KeyError(f"Missing log10_Lstar/log_Lstar in {npz_path.name}")

    samples = np.vstack([m0, b, alpha, beta, log10_Nphi, log10_Lstar]).T
    labels  = [r"$m_0$", r"$b$", r"$\alpha$", r"$\beta$",
               r"$\log_{10} N_\phi$", r"$\log_{10} L_\star$"]
    return samples, labels


def plot_corner(samples, labels, out_png: Path):
    if HAVE_CORNER:
        fig = corner.corner(
            samples,
            labels=labels,
            show_titles=True,
            quantiles=[0.16, 0.50, 0.84],
            title_fmt=".2f",
            title_kwargs={"fontsize": 10},
            bins=40,
            smooth=0.9,
        )
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        # simple fallback pair-grid
        import itertools
        K = samples.shape[1]
        fig, axes = plt.subplots(K, K, figsize=(2.6*K, 2.6*K))
        for i, j in itertools.product(range(K), range(K)):
            ax = axes[i, j]
            if i == j:
                ax.hist(samples[:, i], bins=40, histtype="step")
            else:
                ax.scatter(samples[:, j], samples[:, i], s=1, alpha=0.15)
            if i == K-1:
                ax.set_xlabel(labels[j])
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(labels[i])
            else:
                ax.set_yticklabels([])
        fig.tight_layout()
        fig.savefig(out_png, dpi=200)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Directory with NPZ posteriors")
    ap.add_argument("--pattern", default="y3_bin*_z*.npz",
                    help="Glob pattern for NPZ files (default: y3_bin*_z*.npz)")
    args = ap.parse_args()

    in_dir = Path(args.dir)
    files = sorted(in_dir.glob(args.pattern))
    if not files:
        print(f"No files matched: {in_dir}/{args.pattern}")
        return

    print(f"Found {len(files)} files in {in_dir}")
    print(f"corner installed: {HAVE_CORNER}")

    for p in files:
        try:
            samples, labels = load_params(p)
        except Exception as e:
            print(f"Skip {p.name}: {e}")
            continue

        # save into figs/ next to the NPZ
        outdir = p.parent / "figs"
        outdir.mkdir(parents=True, exist_ok=True)
        out_png = outdir / f"corner_{p.stem}.png"

        print(f"Plotting {p.name} -> {out_png}")
        plot_corner(samples, labels, out_png)

        # print summaries
        print("  Summaries:")
        for lab, col in zip(labels, samples.T):
            print("   ", summarize(lab, col))
        print()

    print("Done.")


if __name__ == "__main__":
    main()