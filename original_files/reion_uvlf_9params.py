import numpy as np

import reion_uvlf_funcs

def log_likelihood(lsum, ldiff, l2, l3,asum, adiff, log10_fesc10, alpha_esc, log10Mcrit,lnk, lnpk, omega_m, omega_l, h, omega_b, YHe=0.24):


    """
    Compute the log-likelihood  log L = -chi^2 / 2.

    Free parameters : lsum, ldiff, l2, l3   (redshift evolution of log10_fstar10_by_cstar)
                      asum, adiff           (redshift evolution of alpha_star)
                      log10_fesc10, alpha_esc, log10Mcrit

    lnk  = np.log(k),    k in h/Mpc
    lnpk = np.log(P(k)), P(k) at z=0 in (Mpc/h)^3
    """

    cdict = {"omega_M_0": omega_m, "omega_b_0": omega_b, "h": h, "YHe": YHe, "omega_L_0": omega_l}

    # a2, a3 share the l2, l3 redshift-evolution parameters
    a2 = l2
    a3 = l3

    ################## Generate reionisation history for this particular choice of parameters ##########################

    model_z_arr, model_QHII_arr, model_tau_value = reion_uvlf_funcs.reionHist_model(
        lsum, ldiff, l2, l3, asum, adiff, a2, a3,
        log10Mcrit, log10_fesc10, alpha_esc, lnk, lnpk, cdict)

    ################## Build the model and data arrays for this particular choice of parameters ##########################

    model_QHI_UVLF_arr = []
    model_allUVLF_arr, data_allUVLF_arr, sigma_allUVLF_arr = reion_uvlf_funcs.model_and_data_allUVLF(
        model_QHI_UVLF_arr, log10Mcrit, lsum, ldiff, l2, l3, asum, adiff, a2, a3,
        log10_fesc10, alpha_esc, lnk, lnpk, cdict)
    chisq_allUVLF = reion_uvlf_funcs.get_chisq(model_allUVLF_arr, data_allUVLF_arr, sigma_allUVLF_arr)

    model_QHI_arr, data_QHI_arr, sigma_QHI_arr = reion_uvlf_funcs.model_and_data_QHI(model_z_arr, model_QHII_arr)
    chisq_QHI = reion_uvlf_funcs.get_chisq(model_QHI_arr, data_QHI_arr, sigma_QHI_arr)

    chisq_total = chisq_QHI + chisq_allUVLF

    ################## Build the log-likelihood function for this particular choice of parameters ##########################

    log_like = -0.5 * chisq_total

    if not np.isfinite(log_like):
        log_like = -1.e6  # guard against infinities: treat as logL = -inf

    ################## Derived parameters ##########################

    l0 = (lsum + ldiff) / 2.0
    l1 = (lsum - ldiff) / 2.0
    a0 = (asum + adiff) / 2.0
    a1 = (asum - adiff) / 2.0

    twolone = 2.0 * l1   # total jump in log10_fstar10_by_cstar
    twoaone = 2.0 * a1   # total jump in alpha_star

    derived = {
        "z_arr": model_z_arr, "QHII_arr": model_QHII_arr,
        "tau_e":   model_tau_value,
        "twolone": twolone, "twoaone": twoaone,
        "l0": l0, "l1": l1,
        "a0": a0, "a1": a1,
    }

    return log_like, derived