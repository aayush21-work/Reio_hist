import numpy as np
import matplotlib.pyplot as plt
import os, sys
import numpy as np
from cobaya.model import get_model
import mcmc_uvlf_cmb as M
model = get_model(M.info)


PROJECT = "/home/aayush/PROJECT_NCRA"
DATAFILE = PROJECT + "/Planck_EE_unbinned.txt"
os.chdir(PROJECT + "/data_files")
sys.path.insert(0, PROJECT)
import reion_uvlf_4params as uvlf
from classy import Class

H0, omega_b, omega_cdm, n_s, A_s = 67.4, 0.0224, 0.120, 0.965, 2.097e-9
h = H0 / 100.0
omega_m = (omega_cdm + omega_b) / h**2
omega_l = 1.0 - omega_m
omega_b_frac = omega_b / h**2

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

ell = np.arange(2501)
fac = ell * (ell + 1) / (2 * np.pi) * (2.7255e6)**2

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
fig,ax=plt.subplots()
# ax[0].set_xlabel("l"); ax[0].set_ylabel("D_l TT [uK^2]"); ax[0].set_title("TT")
ax[1].set_xlabel("l"); ax[1].set_ylabel("D_l EE [uK^2]"); ax[1].set_title("EE")
# ax[2].set_xlim(0, 20); ax[2].set_xlabel("z"); ax[2].set_ylabel("Q_HII")
# ax[2].set_title("reionization history")

dat = np.loadtxt(DATAFILE)
l_dat, D_dat = dat[:, 0], dat[:, 1]
err = [np.abs(dat[:, 2]), dat[:, 3]]
ax[1].errorbar(l_dat, D_dat, yerr=err, fmt="k.", ms=4, capsize=2, lw=1, label="Planck EE")


def get_cmb(lsum, ldiff, l2, l3, plot=True):
    logl, d = uvlf.log_likelihood(lsum, ldiff, l2, l3, lnk, lnpk,omega_m, omega_l, h, omega_b_frac, YHe=0.24)
    z = np.asarray(d["z_arr"])
    Q = np.asarray(d["QHII_arr"])
    order = np.argsort(z)
    z, Q = z[order], Q[order]

    xe = np.where(z >= 3.0, Q * 1.0789, Q * 1.1578)
    xe = np.where(xe < 1e-4, 0.0, xe)
    last = np.where(xe > 0)[0][-1]
    z_inj = np.append(z[:last+1], z[last+1] if last+1 < len(z) else z[-1] + 0.2)
    xe_inj = np.append(xe[:last+1], 0.0)

    cc = Class()
    cc.set({"output": "tCl,pCl,lCl", "lensing": "yes", "N_ur": 3.044,
            "H0": H0, "omega_b": omega_b, "omega_cdm": omega_cdm,
            "n_s": n_s, "A_s": A_s,
            "reio_parametrization": "reio_inter",
            "reio_inter_num": int(len(z_inj)),
            "reio_inter_z": ",".join(f"{v:g}" for v in z_inj),
            "reio_inter_xe": ",".join(f"{v:.6f}" for v in xe_inj)})
    cc.compute()
    cls = cc.lensed_cl(2500)
    tau = cc.get_current_derived_parameters(['tau_reio'])['tau_reio']
    cc.struct_cleanup()

    chi2 = -2 * logl
    label = f"({lsum}, {ldiff}, {l2}, {l3})"
    print(f"{label}: chi2_uvlf={chi2:.2f}  tau={tau:.5f}")

    if plot:
        # ax[0].plot(ell[2:], fac[2:] * cls['tt'][2:],ls=None,ms=1, label=label)
        # ax[1].loglog(ell[2:], fac[2:] * cls['ee'][2:], label=f"{label} tau={tau:.4f}")
        ax[1].loglog(ell[2:], fac[2:] * cls['ee'][2:], label=f"{label} tau={tau:.4f}")
        ax[1].set_xlim(2, 30)
        # ax[2].plot(z, Q, label=label)
        for a in ax:
            a.legend(fontsize=8)

    return z, Q, cls, tau



def get_like(lsum, ldiff, l2, l3, **cosmo):
    point = {"lsum": lsum, "ldiff": ldiff, "l2": l2, "l3": l3}
    point.update(cosmo)

    loglikes, derived = model.loglikes(point, as_dict=True, return_derived=True)

    chi2 = {name: -2 * val for name, val in loglikes.items()}
    chi2_cmb = sum(v for k, v in chi2.items() if k != "uvlf")
    total = sum(chi2.values())

    print(f"({lsum}, {ldiff}, {l2}, {l3})")
    for name in sorted(chi2):
        print(f"  {name:42s} chi2 = {chi2[name]:10.3f}")
    print(f"  {'chi2_CMB (sum of Planck)':42s}      = {chi2_cmb:10.3f}")
    print(f"  {'chi2_TOTAL':42s}      = {total:10.3f}")
    print(f"  tau_e = {derived.get('tau_e', float('nan')):.5f}")
    print(f"  logpost = {model.logpost(point):.4f}")
    print()

    return chi2, derived


get_cmb(-0.458, -0.883, 10.78, 1.148)
get_cmb(1.86, -0.91, 16.27, 4.80)

# get_like(-0.458, -0.883, 10.78, 1.148)
# get_like(2.0068, -0.9335, 16.42, 4.510)


plt.tight_layout()
plt.show()
