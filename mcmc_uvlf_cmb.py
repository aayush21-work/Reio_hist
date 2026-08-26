import os
import sys
import numpy as np
from pathlib import Path
from cobaya.theory import Theory
from cobaya.likelihood import Likelihood
from cobaya.theories.classy import classy as BaseClassy

PROJECT  = Path(__file__).resolve().parent
DATA_DIR = PROJECT / "data_files"
PACKAGES = str(PROJECT / "cobaya_packages")


sys.path.insert(0, str(PROJECT))
os.chdir(DATA_DIR)    


import reion_uvlf_4params as uvlf



# helium factors for QHII -> x_e
HE_FACTOR     = 1.0789     # z >= 3
HE_FACTOR_LOW = 1.1578     # z < 3
ZMIN_COMPLETE = 5.0        # require QHII ~ 1 by this z, else reject
DZ_ANCHOR     = 0.4



class UVLFReio(Theory):
    params = {"lsum": None, "ldiff": None, "l2": None, "l3": None,
              "H0": None, "omega_b": None, "omega_cdm": None,
              "n_s": None, "A_s": None}

    def initialize(self):
        from classy import Class
        self._pk_classy = Class() 

    def get_requirements(self):
        return {}  #pass for now 

    def get_can_provide(self):
        return ["reio_inter_z", "reio_inter_xe", "reio_inter_num", "uvlf_logl"]

    def get_can_provide_params(self):
        return ["tau_e"]

    def calculate(self, state, want_derived=True, **params):
        lsum  = params["lsum"]
        ldiff = params["ldiff"]
        l2    = params["l2"]
        l3    = params["l3"]
        H0    = params["H0"]


        self._pk_classy.set({
            "output": "mPk", "P_k_max_1/Mpc": 500.0, "z_max_pk": 1.0, "k_per_decade_for_pk": 20,
            "H0": params["H0"], "omega_b": params["omega_b"],
            "omega_cdm": params["omega_cdm"], "n_s": params["n_s"],
            "A_s": params["A_s"],
        })
        self._pk_classy.compute()
        h = params["H0"] / 100.0
        kk = np.logspace(-4, np.log10(500), 500)     # 1/Mpc
        Pk0 = np.array([self._pk_classy.pk_lin(k, 0.0) for k in kk])  # Mpc^3
        lnk  = np.log(kk / h)              # -> h/Mpc
        lnpk = np.log(Pk0 * h**3)          # -> (Mpc/h)^3


        #pk_grid, k_grid, z_grid = self._pk_classy.get_pk_and_k_and_z(nonlinear=False)
        #Pk0 = pk_grid[:, -1]              # z=0 column
        #lnk  = np.log(k_grid / h)
        #lnpk = np.log(Pk0 * h**3)
        self._pk_classy.struct_cleanup()

        omega_m = (params["omega_cdm"] + params["omega_b"]) / h**2
        omega_l = 1.0 - omega_m
        omega_b = params["omega_b"] / h**2

        # logl, derived = uvlf.log_likelihood(
        #     params["lsum"], params["ldiff"], params["l2"], params["l3"],
        #     lnk, lnpk, omega_m, omega_l, h, omega_b, YHe=0.24)

        # UVLF + QHI model 
        try:
            logl, derived = uvlf.log_likelihood(lsum, ldiff, l2, l3, lnk, lnpk,omega_m, omega_l, h, omega_b,
             YHe=0.24)
        except Exception as e:
            self.log.warning(f"UVLF model failed: {e!r}")
            return False
        if not np.isfinite(logl):
            return False

        self._uvlf_logl = float(logl)
        state["uvlf_logl"] = float(logl)

        # build reio_inter from the model QHII(z) 
        z_arr = np.asarray(derived["z_arr"], dtype=float)     # model z grid
        QHII  = np.asarray(derived["QHII_arr"], dtype=float)

        # sort ascending in z, keep z and Q locked together
        order = np.argsort(z_arr)
        zc = z_arr[order]
        Qc = QHII[order]

        # completion check: QHII ~ 1 at z = 5
        i5 = int(np.argmin(np.abs(zc - 5.0)))
        if Qc[i5] < 0.99:
            return False

        # helium-corrected x_e (same length as zc)
        xe = np.where(zc >= 3.0, Qc * HE_FACTOR, Qc * HE_FACTOR_LOW)

        # clamp the tiny high-z tail to exactly zero
        xe = np.where(xe < 1e-4, 0.0, xe)

        # find the last index where x_e is still > 0
        nz = np.where(xe > 0)[0]
        if len(nz) == 0:
            return False
        last = int(nz[-1])

        # keep z and xe UP TO AND INCLUDING that last nonzero point,
        # then append exactly ONE zero anchor. z and xe stay equal length.
        zc = zc[:last + 1]
        xe = xe[:last + 1]

        # append single zero anchor just beyond the last real point
        z_next = zc[-1] + DZ_ANCHOR
        zc = np.append(zc, z_next)
        xe = np.append(xe, 0.0)

        # z and xe are now guaranteed equal length
        # assert len(zc) == len(xe), f"length mismatch z={len(zc)} xe={len(xe)}"


        state["reio_inter_z"]   = ",".join(f"{v:g}"   for v in zc)
        state["reio_inter_xe"]  = ",".join(f"{v:.6f}" for v in xe)
        state["reio_inter_num"] = int(len(xe))

        if want_derived:
            state["derived"] = {"tau_e": float(derived["tau_e"])}

# Likelihood: returns the UVLF/QHI logL that the theory computed
class UVLFLike(Likelihood):
    def get_requirements(self):
        return {"uvlf_logl": None}

    def logp(self, **params_values):
        return self.provider.get_result("uvlf_logl")


# classy subclass: inject the UVLF QHII history each step
class classy_reio(BaseClassy):
    def get_requirements(self):
        reqs = super().get_requirements()
        reqs = dict(reqs) if reqs else {}
        reqs.update({"reio_inter_z": None, "reio_inter_xe": None,
                     "reio_inter_num": None})
        return reqs

    def calculate(self, state, want_derived=True, **params):
        params.pop("logA", None)
        self.extra_args["reio_parametrization"] = "reio_inter"
        self.extra_args["reio_inter_num"] = int(self.provider.get_result("reio_inter_num"))
        self.extra_args["reio_inter_z"]   = self.provider.get_result("reio_inter_z")
        self.extra_args["reio_inter_xe"]  = self.provider.get_result("reio_inter_xe")
        return super().calculate(state, want_derived=want_derived, **params)


info = {
    "theory": {
        "uvlf_reio": {"external": UVLFReio},
        "classy_reio": {
            "external": classy_reio,
            "extra_args": {"N_ur": 3.044},
        },
    },
    "likelihood": {
        #"uvlf": {"external": UVLFLike},
        "planck_2018_lowl.TT": None,
        "planck_2018_lowl.EE": None,
        "planck_2018_highl_plik.TTTEEE_lite": None,
    },
    "params": {
        # galaxy params
        "lsum":  {"prior": {"min": -2.0, "max": 2.0}, "ref": -0.2, "proposal": 0.01, "latex": r"\ell_{\rm sum}"}, #-0.2
        "ldiff": {"prior": {"min": -2.0, "max": 1.0}, "ref": -0.75, "proposal": 0.01, "latex": r"\ell_{\rm diff}"}, # -0.75
        "l2":    {"prior": {"min": 8, "max": 18.0}, "ref": 13, "proposal": 0.01, "latex": r"\ell_2"}, #8,18, ref:13
        "l3":    {"prior": {"min": -3.0, "max": 6}, "ref": 2.16, "proposal": 0.01, "latex": r"\ell_3"}, #0.5,6,ref:2.16
        # cosmology 
        # "H0":        {"prior": {"min": 60., "max": 75.}, "ref": 65.0, "proposal": 0.5, "latex": r"H_0"},
        # "omega_b":   {"prior": {"min": 0.01, "max": 0.5}, "ref": 0.02, "proposal": 0.0001, "latex": r"\omega_b"},
        # "omega_cdm": {"prior": {"min": 0.01, "max": 0.5}, "ref": 0.1, "proposal": 0.0001, "latex": r"\omega_{cdm}"},
        "n_s":       {"prior": {"min": 0.92, "max": 1.00}, "ref": 0.96, "proposal": 0.001, "latex": r"n_s"},
        "logA":      {"prior": {"min": 2.5, "max": 3.5}, "ref": 3.0, "proposal": 0.01, "drop": True, "latex": r"\ln(10^{10}A_s)"},
        "A_s": {"value": "lambda logA: 1e-10*np.exp(logA)", "latex": r"A_s"},
        # "A_planck":  1,
        "H0" : 67.4 ,
        "omega_b" : 0.0224 ,
        "omega_cdm": 0.120,
        #"n_s" : 0.965,
        #"logA" : 3.043,
        "A_planck" : 1,
        # derived 
        "tau_e":     {"latex": r"\tau_e"},
    },
    "sampler": {"mcmc": {"Rminus1_stop": 0.01, "learn_proposal": True, "max_samples": 200000}},
    "output": str(PROJECT / "chains" / "uvlf_cmb_without_uvlf"),
    "packages_path": PACKAGES,
    "resume": True,
}


if __name__ == "__main__":
    from cobaya.run import run
    run(info,allow_changes=True)
