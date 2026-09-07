import sys
import numpy as np
from pathlib import Path
from cobaya.theory import Theory

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_pipeline as P

PACKAGES = "/home/aayush/cobaya_packages"


def set_global(zeta, log10_Mmin):
    import configparser
    cfg = configparser.ConfigParser()
    cfg.read(P.GLOBAL_INI)
    cfg["script"]["zeta"]       = f"{zeta:.8g}"
    cfg["script"]["log10_Mmin"] = f"{log10_Mmin:.8g}"
    with open(P.GLOBAL_INI, "w") as f:
        cfg.write(f)


def build_reio_inter(zeta, log10_Mmin):
    set_global(zeta, log10_Mmin)
    # P.run_script_parallel()                      # writes xe_history.dat
    result = P.run_script_parallel()
    if result is None:
        raise RuntimeError(
            f"SCRIPT reionisation rejected for zeta={zeta}, log10_Mmin={log10_Mmin}")
    s = P.load_script_cfg()
    he      = float(s["helium_factor"])
    he_low  = float(s["helium_factor_lowz"])

    data = np.loadtxt(P.PROJECT_DIR /"xe_history.dat", comments="#")
    z, xe = data[:, 0], data[:, 2]
    order = np.argsort(z)

    z, xe = z[order], xe[order]

    
    if xe[-1] != 0.0:
        z  = np.append(z, z[-1] + P.step)
        xe = np.append(xe, 0.0)

    
    z_low  = np.round(np.arange(0, P.zmin, P.step), 1)
    xe_low = np.array([he if zz > 3 else he_low for zz in z_low])

    z_full  = np.concatenate([z_low, z])
    xe_full = np.concatenate([xe_low, xe])

    

    nz = np.where(xe_full > 0)[0]
    if len(nz):
        last = nz[-1]
        z_full  = np.append(z_full[:last + 1],  z_full[last + 1])
        xe_full = np.append(xe_full[:last + 1], 0.0)
    
    Q_lowz = xe[0] / he  # assuming reionisation is complete by z=5 (already enforced below)

    return z_full, xe_full, Q_lowz



class ScriptReio(Theory):
    params = {"zeta": None, "log10_Mmin": None}

    def initialize(self):
        self._reio_z = None
        self._reio_xe = None
        self._tau = np.nan

    def get_requirements(self):
        return {}

    def get_can_provide_params(self):
        return ["tau_script","Q_lowz"]

    # classy will ask us what extra CLASS args to use this step
    def calculate(self, state, want_derived=True, **params):
        zeta  = params["zeta"]
        lmmin = params["log10_Mmin"]
        try:
            z, xe,Q_lowz = build_reio_inter(zeta, lmmin)
        except Exception as e:
            self.log.warning(f"SCRIPT failed zeta={zeta} Mmin={lmmin}: {e!r}")
            return False

        # store as CLASS-style comma strings
        z_str  = ",".join(f"{v:g}"   for v in z)
        xe_str = ",".join(f"{v:.6f}" for v in xe)

        # these get injected into classy via the provider mechanism below
        state["reio_inter_z"]   = z_str
        state["reio_inter_xe"]  = xe_str
        state["reio_inter_num"] = len(z)
        if want_derived:
            state["derived"] = {"tau_script": np.nan, "Q_lowz": Q_lowz}   # classy will give real tau

    def get_can_provide(self):
        return ["reio_inter_z", "reio_inter_xe", "reio_inter_num"]


Q_COMPLETE_THRESHOLD = 0.99
 
def reio_complete_veto(_self=None, Q_lowz=None):
    if Q_lowz is None or not np.isfinite(Q_lowz) or Q_lowz < Q_COMPLETE_THRESHOLD:
        return -np.inf
    return 0.0


from cobaya.theories.classy import classy as BaseClassy


class classy_reio(BaseClassy):
    def get_requirements(self):
        # require what ScriptReio provides
        reqs = super().get_requirements()
        reqs = dict(reqs) if reqs else {}
        reqs.update({"reio_inter_z": None,
                     "reio_inter_xe": None,
                     "reio_inter_num": None})
        return reqs

    def calculate(self, state, want_derived=True, **params):
        # fetch the per-step reio history from the provider (ScriptReio)
        z_str  = self.provider.get_result("reio_inter_z")
        xe_str = self.provider.get_result("reio_inter_xe")
        num    = self.provider.get_result("reio_inter_num")
        # inject into CLASS for this step
        self.extra_args["reio_parametrization"] = "reio_inter"
        self.extra_args["reio_inter_num"] = num
        self.extra_args["reio_inter_z"]   = z_str
        self.extra_args["reio_inter_xe"]  = xe_str
        return super().calculate(state, want_derived=want_derived, **params)


info = {
    "theory": {
        "script_reio": {"external": ScriptReio},
        "classy_reio": {
            "external": classy_reio,
            "extra_args": {
                "output": "tCl,pCl,lCl",
                "lensing": "yes",
                "N_ur": 3.044,
            },
        },
    },
    "likelihood": {
        "reio_complete_veto": {
            "external": reio_complete_veto,
            "requires": {"Q_lowz": None},
        },
        "planck_2018_lowl.EE": None,
        "planck_2018_lowl.TT": None,
        
        
        
    },
    "params": {
        "zeta":       {"prior": {"min": 5.0,  "max": 40.0},
                       "ref": 15, "proposal": 0.01, "latex": r"\zeta"},
        "log10_Mmin": {"prior": {"min": 7.0,  "max": 15.0},
                       "ref": 9, "proposal": 0.01, "latex": r"\log_{10} M_{\min}"},
        "H0":        67.36,  # same config as global.ini, to do make it automatic to load from global.ini 
        "omega_b":   0.02237,
        "omega_cdm": 0.1200,
        "n_s":       0.9649,
        "A_s":       2.1e-9,
        # derived
        "tau_reio":  {"latex": r"\tau"},
        "Q_lowz":    {"latex": r"Q_{z\to0}"},
    },
    "sampler": {"mcmc": {"Rminus1_stop": 0.01, "learn_proposal": True,
                         "max_samples": 5000}},
    "output": "chains/classy_planck_TTEE",
    "packages_path": PACKAGES,
    "force": True,
}


if __name__ == "__main__":
    from cobaya.run import run
    run(info,allow_changes=True)