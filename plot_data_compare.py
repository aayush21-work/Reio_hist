import numpy as np
import matplotlib.pyplot as plt
import os, sys

PROJECT = "/home/aayush/PROJECT_NCRA"
os.chdir(PROJECT + "/data_files")
sys.path.insert(0, PROJECT)
import reion_uvlf_funcs as rf
from classy import Class

H0, omega_b, omega_cdm, n_s, A_s = 67.4, 0.0224, 0.120, 0.965, 2.097e-9
h = H0 / 100.0
omega_m = (omega_cdm + omega_b) / h**2
omega_l = 1.0 - omega_m
omega_b_frac = omega_b / h**2
cdict = {"omega_M_0": omega_m, "omega_b_0": omega_b_frac, "h": h,
         "YHe": 0.24, "omega_L_0": omega_l}

asum, adiff = 9.4376e-01, 2.8934e-01
log10_fesc10, alpha_esc, log10Mcrit = -8.1220e-01, -7.8249e-02, 1.0174e+01

c = Class()
c.set({"output": "mPk", "P_k_max_1/Mpc": 500., "z_max_pk": 1.,
       "H0": H0, "omega_b": omega_b, "omega_cdm": omega_cdm,
       "n_s": n_s, "A_s": A_s})
c.compute()
kk = np.logspace(-4, np.log10(500), 2000)
Pk0 = np.array([c.pk_lin(k, 0.0) for k in kk])
lnk = np.log(kk / h)
lnpk = np.log(Pk0 * h**3)
c.struct_cleanup()


def get_uvlf(lsum, ldiff, l2, l3):
    m, d, s = rf.model_and_data_allUVLF(
        [], log10Mcrit, lsum, ldiff, l2, l3, asum, adiff, l2, l3,
        log10_fesc10, alpha_esc, lnk, lnpk, cdict)
    return np.array(m), np.array(d), np.array(s)


def get_qhi(lsum, ldiff, l2, l3):
    z, Q, tau = rf.reionHist_model(lsum, ldiff, l2, l3, asum, adiff, l2, l3,
                                   log10Mcrit, log10_fesc10, alpha_esc,
                                   lnk, lnpk, cdict)
    m, d, s = rf.model_and_data_QHI(z, Q)
    return z, Q, np.array(m), np.array(d), np.array(s)


sets = [((-0.458, -0.883, 10.78, 1.148), "mode 1", "tab:blue"),
        ((2.0068, -0.9335, 16.42, 4.510), "mode 2", "tab:red")]

zbins = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.5, 13.2]
muv, npz = [], []
for zz in zbins:
    tag = str(zz).replace(".", "p")
    fn = f"UVLF_datafiles/UVLF_z{tag}.txt"
    col = np.loadtxt(fn, usecols=1, skiprows=2)
    muv.append(np.atleast_1d(col))
    npz.append(len(np.atleast_1d(col)))

fig, axes = plt.subplots(3, 3, figsize=(14, 11))
axes = axes.ravel()

first = True
for params, label, color in sets:
    m, d, s = get_uvlf(*params)
    chi2 = ((m - d) / s) ** 2
    i0 = 0
    for j, zz in enumerate(zbins):
        n = npz[j]
        sl = slice(i0, i0 + n)
        ax = axes[j]
        if first:
            ax.errorbar(muv[j], d[sl], yerr=s[sl], fmt="k.", ms=5,
                        capsize=2, lw=1, label="data")
        ax.plot(muv[j], m[sl], color=color, lw=1.5,
                label=f"{label} ($\\chi^2$={chi2[sl].sum():.0f})")
        ax.set_yscale("log")
        ax.set_title(f"z = {zz}")
        ax.set_xlabel(r"$M_{UV}$")
        ax.set_ylabel(r"$\phi$ [mag$^{-1}$ Mpc$^{-3}$]")
        ax.legend(fontsize=7)
        i0 += n
    first = False

plt.tight_layout()
plt.show()

fig2, ax2 = plt.subplots(figsize=(7, 5))
firstq = True
for params, label, color in sets:
    z, Q, m, d, s = get_qhi(*params)
    if firstq:
        zq = np.loadtxt("QHI_datafiles/fullQHIdata.txt", usecols=1, skiprows=2)
        ax2.errorbar(zq, d, yerr=s, fmt="k.", ms=6, capsize=2, lw=1, label="QHI data")
        firstq = False
    chi2q = ((m - d) / s) ** 2
    ax2.plot(z, 1.0 - Q, color=color, lw=1.6,
             label=f"{label} ($\\chi^2$={chi2q.sum():.1f})")
ax2.set_xlim(4, 16)
ax2.set_ylim(-0.05, 1.05)
ax2.set_xlabel("z")
ax2.set_ylabel(r"$Q_{HI}$")
ax2.legend()
plt.tight_layout()
plt.show()