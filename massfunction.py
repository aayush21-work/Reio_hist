import numpy as np
import scipy as sp
from transferfunction import *

rho_crit_by_hsq = 2.7755e11  ## in Msun / Mpc^3

def get_massfunction(log10Mmin, log10Mmax, dlog10m, z, lnk, lnpk, cdict, fit="ST", delta_c = 1.686):

    log10M = np.arange(log10Mmin, log10Mmax, dlog10m)
    M = 10. ** log10M

    dpl = growth_factor(z, cdict)
    delta_c_z = delta_c / dpl
    mean_dens = cdict["omega_M_0"] * rho_crit_by_hsq

    sgma = sigma(M, lnpk, lnk, mean_dens) ## sigma at z = 0
    dlsigmadlm = dlnsigmadlnm(M, sgma, lnpk, lnk, mean_dens)

    nu = (delta_c_z / sgma) ** 2
    fsigma = get_fsigma(sgma, fit, delta_c_z, z)
    fs = get_fs(delta_c_z, sgma**2, fit, cdict, z)

    dndm = get_dndm(M, sgma, fs, dlsigmadlm, mean_dens)
    dndlnm = M * dndm

    ### M in Msun / h, dndlnm in h^3/Mpc^3
    return M, sgma, nu, fsigma, fs, dlsigmadlm, dndm, dndlnm

def get_dndm(M, sgma, fs, dlsigmadlm, mean_dens):
    dndm = 2 * sgma ** 2 * fs * mean_dens * np.abs(dlsigmadlm) / M ** 2
    return dndm

def get_fsigma(sgma, fit, delta_c_z, z=0):
    #returns f(sigma) = vfv = 2 f(s) s

    nu = (delta_c_z / sgma) ** 2
    if fit == "ST":
        A = 0.322
        a = 0.707
        p = 0.3
        a_nu = a * nu
        return A * np.sqrt(2.0 * a_nu / np.pi) * np.exp(- 0.5 * a_nu) * (1 + (1.0 / (a_nu)) ** p)

    elif fit == "ST_Jenkins":
        A = 0.353
        a = 0.73
        p = 0.175
        a_nu = a * nu
        return A * np.sqrt(2.0 * a_nu / np.pi) * np.exp(- 0.5 * a_nu) * (1 + (1.0 / (a_nu)) ** p)

    elif fit == "PS":
        return np.sqrt(2.0 * nu / np.pi) * np.exp(- 0.5 * nu)

    elif fit == "Tinker08":
        A0 = 0.186
        a0 = 1.47
        b0 = 2.57
        c = 1.19
        Delta = 200.0

        A = A0 * (1 + z) ** (-0.14)
        a = a0 * (1 + z) ** (-0.06)
        alpha = 10. ** (- (0.75 / np.log10(Delta / 75.0)) ** 1.2)
        b = b0 * (1 + z) ** (-alpha)

        Dz = 1.686 / delta_c_z
        
        return A * ( ((Dz * sgma) / b) ** (-a) + 1 ) * np.exp( - c / (Dz * sgma) ** 2)

def get_fs(delta, s, fit, cdict, z=0):
    #returns f(s)

    if delta <= 0:
        return 0

    elif fit == "ST":
        A = 0.322
        a = 0.707
        p = 0.3
        a_nu = a * delta ** 2 / s
        return A * np.sqrt(0.5 * a_nu / np.pi) * (1 + (1.0 / (a_nu)) ** p) * (1 / s) * np.exp(- 0.5 * a_nu)

    elif fit == "ST_Jenkins":
        A = 0.353
        a = 0.73
        p = 0.175
        a_nu = a * delta ** 2 / s
        return A * np.sqrt(0.5 * a_nu / np.pi) * (1 + (1.0 / (a_nu)) ** p) * (1 / s) * np.exp(- 0.5 * a_nu)

    elif fit == "PS":
        nu = delta ** 2 / np.abs(s)
        #print s, delta
        #return np.sqrt(0.5 * nu / np.pi) * (1 / s) * np.exp(- 0.5 * nu)
        return np.where(s > 0, np.sqrt(0.5 * nu / np.pi) * (1 / s) * np.exp(- 0.5 * nu), 0)

    elif fit == "Tinker08":
        A0 = 0.186
        a0 = 1.47
        b0 = 2.57
        c = 1.19
        Delta = 200.0

        A = A0 * (1 + z) ** (-0.14)
        a = a0 * (1 + z) ** (-0.06)
        alpha = 10. ** (- (0.75 / np.log10(Delta / 75.0)) ** 1.2)
        b = b0 * (1 + z) ** (-alpha)

        Dz = growth_factor(z, cdict)
        
        sgma = Dz * np.sqrt(s)

        return (0.5 / s) * A * ( (sgma / b) ** (-a) + 1 ) * np.exp( - c / sgma ** 2)


def get_dndm_cond(Mbig, delta0, sgma_Mbig, M, sgma, dlsigmadlm, mean_dens, delta_c_z):

    ## only for z = 0
    if delta0 >= delta_c_z:
        return 0

    s = sgma ** 2 - sgma_Mbig ** 2
    fs = np.where(s > 0, get_fs(delta_c_z - delta0, s, fit = 'PS'), 0)

    dndm_cond = 2 * sgma ** 2 * fs * mean_dens * np.abs(dlsigmadlm) / M ** 2
    return dndm_cond

def linear_bias(nu_ST,delta_c):
    # Sheth-Tormen bias

    #q = 0.707
    #p = 0.3
    q = 0.73
    p = 0.353
    return 1 + (q * nu_ST - 1) / delta_c + (2 * p / delta_c) / (1 + (q * nu_ST) ** p)
    # a = 0.707
    # b = 0.5
    # c = 0.6
    #
    # a_nu = a * nu_ST
    # return 1 + 1 / (np.sqrt(a) * delta_c) * (np.sqrt(a) * a_nu + np.sqrt(a) * b * a_nu ** (1 - c) - a_nu ** c / (a_nu ** c + b * (1 - c) * (1 - c / 2) ))

def get_fcoll(s, fs):
    #res = - sp.integrate.trapezoid(fs, s)
    res = sp.integrate.cumtrapz(fs, s, initial=0)
    res = res - res[-1]
    return res

###########################################
###########################################

def barrier_ellipsoidal(S, delta_c_z, a, beta, alpha):
    nu = delta_c_z ** 2 / S
    return np.sqrt(a) * delta_c_z * (1 + beta / (a * nu) ** alpha)

def get_T_series(s_small, S_big, delta_lin_0, delta_c_z, a, beta, alpha):
    #### delta_lin_0 = delta_lin / D(z)

    B_s = barrier_ellipsoidal(s_small, delta_c_z, a=a, beta=beta, alpha=alpha)
    delta_1 = np.sqrt(a) * delta_c_z
    fac = beta * delta_1 ** (1 - 2 * alpha)

    T0 = B_s - delta_lin_0
    T1 = (S_big - s_small) * alpha * fac * s_small ** (alpha - 1)
    T2 = ((S_big - s_small) ** 2 / 2) * alpha * (alpha - 1) * fac * s_small ** (alpha - 2)
    T3 = ((S_big - s_small) ** 3 / 6) * alpha * (alpha - 1) * (alpha - 2) * fac * s_small ** (alpha - 3)
    T4 = ((S_big - s_small) ** 4 / 24) * alpha * (alpha - 1) * (alpha - 2) * (alpha - 3) * fac * s_small ** (alpha - 4)
    T5 = ((S_big - s_small) ** 5 / 120) * alpha * (alpha - 1) * (alpha - 2) * (alpha - 3) * (alpha - 4) * fac * s_small ** (alpha - 5)


    return T0 + T1 + T2 + T3 + T4 + T5

def get_fs_cond_ellipsoidal(s_small, S_big, delta_lin_0, delta_c_z, a=0.707, beta=0.485, alpha=0.615):

    res = np.zeros(len(s_small))
    mask =  s_small > S_big

    B_s = barrier_ellipsoidal(s_small[mask], delta_c_z, a=a, beta=beta, alpha=alpha)

    sdiff = s_small[mask] - S_big
    T = get_T_series(s_small[mask], S_big, delta_lin_0, delta_c_z, a=a, beta=beta, alpha=alpha)  #### delta_lin_0 = delta_lin / D(z)
    res[mask] = np.abs(T) * np.exp( - 0.5 * (B_s - delta_lin_0) ** 2 / sdiff) / np.sqrt(2 * np.pi * sdiff ** 3)

    return res

def get_massfunction_ellipsoidal(log10Mmin, log10Mmax, dlog10m, z, lnk, lnpk, cdict, delta_c = 1.686, a=0.707, beta=0.485, alpha=0.615):

    log10M = np.arange(log10Mmin, log10Mmax, dlog10m)
    M = 10. ** log10M

    dpl = growth_factor(z, cdict)
    delta_c_z = delta_c / dpl
    mean_dens = cdict["omega_M_0"] * rho_crit_by_hsq

    sgma = sigma(M, lnpk, lnk, mean_dens) ## sigma at z = 0
    dlsigmadlm = dlnsigmadlnm(M, sgma, lnpk, lnk, mean_dens)

    fs = get_fs_cond_ellipsoidal(sgma ** 2, 0., 0., delta_c_z, a=a, beta=beta, alpha=alpha)
    nu = (delta_c_z / sgma) ** 2
    fsigma = 2 * sgma ** 2 * fs          ### vfv

    dndm = get_dndm(M, sgma, fs, dlsigmadlm, mean_dens)
    dndlnm = M * dndm

    ### M in Msun / h, dndlnm in h^3/Mpc^3
    return M, sgma, nu, fsigma, fs, dlsigmadlm, dndm, dndlnm

def get_dndm_cond_ellipsoidal(Mbig, delta_lin_0, sgma_Mbig, M, sgma, dlsigmadlm, mean_dens, delta_c_z, a=0.707, beta=0.485, alpha=0.615):
    #### delta_lin_0 = delta_lin / D(z)

    if delta_lin_0 >= delta_c_z:
        return np.zeros_like(M), np.zeros_like(M)

    fs = get_fs_cond_ellipsoidal(sgma ** 2, sgma_Mbig ** 2, delta_lin_0, delta_c_z, a=a, beta=beta, alpha=alpha)

    dndm_cond = get_dndm(M, sgma, fs, dlsigmadlm, mean_dens)
    return dndm_cond, fs

def set_delta_NL_spline(num=100):

    theta_arr = np.linspace(-2*np.pi, 2*np.pi, num=num, endpoint=False)
    delta_lin_arr = np.zeros_like(theta_arr)
    delta_NL_arr = np.zeros_like(theta_arr)

    mask_theta_neg = theta_arr < -1.e-12
    mask_theta_zero = np.abs(theta_arr) <= 1.e-12
    mask_theta_pos = theta_arr > 1.e-12

    delta_lin_arr[mask_theta_neg] = - (3. / 20.) * 6 ** (2./3.) * (np.abs(np.sinh(theta_arr[mask_theta_neg]) - theta_arr[mask_theta_neg])) ** (2./3.)
    delta_NL_arr[mask_theta_neg] =  - (10 * delta_lin_arr[mask_theta_neg] / (3 * (np.cosh(theta_arr[mask_theta_neg]) - 1))) ** 3 - 1

    delta_lin_arr[mask_theta_zero] = 0
    delta_NL_arr[mask_theta_zero] = 0


    delta_lin_arr[mask_theta_pos] = (3. / 20.) * 6 ** (2./3.) * (theta_arr[mask_theta_pos] - np.sin(theta_arr[mask_theta_pos])) ** (2./3.)
    delta_NL_arr[mask_theta_pos] =  (10 * delta_lin_arr[mask_theta_pos] / (3 * (1 - np.cos(theta_arr[mask_theta_pos])))) ** 3 - 1
    # print theta_arr[mask_theta_neg]
    # print theta_arr[mask_theta_zero]
    # print theta_arr[mask_theta_pos]
    # print delta_lin_arr
    # print delta_NL_arr

    return delta_lin_arr, delta_NL_arr


def get_poisson_err_fcoll(dNdlnm_arr, mass_arr, num_rea=100, seed_poisson=None, seed_halo_dist=None, bin_edges=[0]):

    m_tot_arr = np.zeros(num_rea)

    if len(bin_edges) > 1:
        n_bins = len(bin_edges) - 1
    else:
        n_bins = len(mass_arr)

    #print n_bins, len(bin_edges)

    dN_poisson_arr = np.zeros((num_rea, n_bins))

    lnm_arr = np.log(mass_arr)
    dlnm  = lnm_arr[1] - lnm_arr[0]
    Nhalo = sp.integrate.trapezoid(dNdlnm_arr, lnm_arr)
    #print Nhalo
    if Nhalo < 1.e-16: return m_tot_arr, dN_poisson_arr

    if len(bin_edges) == 1:
        lnm_edges = np.linspace(lnm_arr[0] - dlnm / 2, lnm_arr[-1] + dlnm / 2, num=n_bins+1, endpoint=True)

    cumprob_arr = sp.integrate.cumtrapz(dNdlnm_arr, lnm_arr, initial=0)
    #print cumprob_arr
    cumprob_arr = cumprob_arr[-1] + cumprob_arr[0] - cumprob_arr
    cumprob_arr = cumprob_arr / cumprob_arr[0]
    #print cumprob_arr[0], cumprob_arr[-1], np.min(cumprob_arr), np.max(cumprob_arr)

    np.random.seed(seed=seed_poisson)
    Nhalo_poisson_arr = np.random.poisson(Nhalo, size=num_rea)

    np.random.seed(seed=seed_halo_dist)
    for i, Nhalo_poisson in enumerate(Nhalo_poisson_arr):
        if (Nhalo_poisson > 0):
            rand_uniform_arr = np.random.uniform(size=Nhalo_poisson)
            lnm_halo_arr = np.interp(rand_uniform_arr, cumprob_arr[::-1], lnm_arr[::-1])
            m_halo_arr = np.exp(lnm_halo_arr)
            m_tot_arr[i] = np.sum(m_halo_arr)
            if len(bin_edges) > 1:
                dN_poisson_arr[i,:], bin_edges = np.histogram(lnm_halo_arr, bins=bin_edges, normed=False)
            else:
                dN_poisson_arr[i,:], lnm_edges = np.histogram(lnm_halo_arr, bins=lnm_edges, normed=False)

    dN_poisson_arr = dN_poisson_arr * 1.0
    return m_tot_arr, dN_poisson_arr
