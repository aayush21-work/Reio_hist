import numpy as np
from getdist import plots, loadMCSamples

CHAIN_ROOT     = "chains/classy_planck_TTEE"   
PLANCK_TAU     = 0.0544
PLANCK_TAU_ERR = 0.0073

samples = loadMCSamples(file_root=CHAIN_ROOT, settings={"ignore_rows": 0.3})

names  = ["zeta", "log10_Mmin", "tau_reio"]
labels = [r"\zeta", r"\log_{10} M_{\rm min}", r"\tau"]


ibest  = np.argmin(samples.loglikes)           
bfrow  = samples.samples[ibest]
pnames = [p.name for p in samples.paramNames.names]
best   = {n: bfrow[pnames.index(n)] for n in names}
print("best-fit (MAP):", best, "  -logL =", samples.loglikes[ibest])


def plot_data(sample_obj, fil):
    g = plots.get_subplot_plotter()
    g.triangle_plot(sample_obj, names, filled=True, title_limit=1, markers=best)

    
    for i, n in enumerate(names):
        ax = g.subplots[i, i]
        ax.set_title(f"${labels[i]} = {best[n]:.4f}$", fontsize=9)

    
    lo, hi = PLANCK_TAU - PLANCK_TAU_ERR, PLANCK_TAU + PLANCK_TAU_ERR
    tcol = names.index("tau_reio")
    for row in range(len(names)):
        ax = g.subplots[row, tcol]
        if ax is None:
            continue
        ax.axvspan(lo, hi, color="orange", alpha=0.3, zorder=0)
        ax.axvline(PLANCK_TAU, color="orange", ls="--", lw=1.0, zorder=1)
        ax.axvline(best["tau_reio"], color="crimson", ls="-", lw=1.2, zorder=3)

    # legend on the tau 1D panel
    ax_tau = g.subplots[tcol, tcol]
    ax_tau.axvline(PLANCK_TAU, color="orange", ls="--", lw=1.2,
                   label=fr"Planck $\tau={PLANCK_TAU}$")
    ax_tau.axvline(best["tau_reio"], color="crimson", ls="-", lw=1.2,
                   label=fr"best-fit $\tau={best['tau_reio']:.4f}$")
    ax_tau.legend(fontsize=6, loc="upper right")

    g.export(fil + "_bestfit_triangle.png", dpi=600)
    print("saved ->", fil + "_bestfit_triangleTTEE.png")


plot_data(samples, "classy_planckTTEE")