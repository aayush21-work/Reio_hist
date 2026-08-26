import numpy as np
import scipy as sp
import collections
from scipy import interpolate

from colossus.cosmology import cosmology

def initialize_power_spectrum(lnkmin, lnkmax, numbin_lnk, cdict, tf='EH', mDM=0.):
    lnk = np.linspace(lnkmin, lnkmax, num = numbin_lnk) 
    k = np.exp(lnk)  ## k in h/Mpc
    if tf=='EH':
        lnpk_unnorm = cdict["n"] * lnk + 2 * np.log(cosmology.power_spectrum.modelEisenstein98(k, cdict["h"], cdict["omega_M_0"], cdict["omega_b_0"], 2.7255))
    elif tf=='BBKS':
        q = np.exp(lnk) / (cdict["omega_M_0"] * cdict["h"])
        aa, bb, cc, powr = 6.4 , 3.0, 1.7, 1.13
        lnpk_unnorm = cdict["n"] * lnk - 2 * np.log((1 + (aa*q + (bb*q) ** 1.5 + (cc*q) ** 2) ** powr) ** (1.0/powr))

    if mDM > 0.:
        ### from Barkana, Haiman & Ostriker (2001) eqs (3) and (4)
        ### mDM is in keV
        # gX = 1.5
        # eps = 0.361
        # eta = 5.0
        # nuDM = 1.2
        # mX = mDM
        # h =  cdict["h"]
        # OmegaX_hsq = cdict["omega_M_0"] * h ** 2
        # Rc = 0.201 * (OmegaX_hsq / 0.15) ** 0.15 * (gX / 1.5) ** -0.29 * (mX / 1.0) ** -1.15 ### in Mpc
        # Rc = Rc * h ### in Mpc / h because the k is in h/Mpc

        ### from Schneider et al (2011) eqs (4) and (5)
        ### mDM is in keV
        eta = 5.0
        nuDM = 1.12
        mX = mDM
        h =  cdict["h"]
        OmegaX = cdict["omega_M_0"]
        Rc = 0.049 * (OmegaX / 0.25) ** 0.11 * (h / 0.7) ** 1.22 * (mX / 1.0) ** -1.15 ### in Mpc / h

        lnpk_unnorm = lnpk_unnorm - 2 * (eta / nuDM) * np.log(1 + (k * Rc) ** (2 * nuDM))

    rho_crit_by_hsq = 2.7755e11  ## in Msun / Mpc^3
    mean_dens = cdict["omega_M_0"] * rho_crit_by_hsq  ## in Msun / Mpc^3
    lnpk, normfac = normalize(cdict["sigma_8"], lnpk_unnorm, lnk, mean_dens)

    return lnk, lnpk
    

    
def normalize(sigma_8, lnpk_unnorm, lnk, mean_dens):

    # Calculate the value of sigma_8 without prior normalization.
    unnorm_sigma_8 = sigma(4.*np.pi * 8 ** 3 * mean_dens / 3., lnpk_unnorm, lnk, mean_dens)[0]

    # Calculate the normalization factor
    normfac = sigma_8 / unnorm_sigma_8

    # Normalize the previously calculated power spectrum.
    lnpk = 2 * np.log(normfac) + lnpk_unnorm

    return lnpk, normfac

def sigma(M, lnpk, lnk, mean_dens, scheme='trapezoid'):
    
    # If we input a scalar as M, then just make it a one-element list.
    if not isinstance(M, collections.abc.Iterable):
        M = [M]


    dlnk = lnk[1] - lnk[0]
    sigmasq = np.zeros_like(M)
    kcube_pk = np.exp(lnpk + 3 * lnk)
    for i, m in enumerate(M):
        integrand = kcube_pk * top_hat_window(m, lnk, mean_dens)
        #print integrand
        if scheme == "trapezoid":
            sigmasq[i] = (0.5 / np.pi ** 2) * sp.integrate.trapezoid(integrand, dx=dlnk)
        elif scheme == "simpson":
            sigmasq[i] = (0.5 / np.pi ** 2) * sp.integrate.simpson(integrand, dx=dlnk)
        elif scheme == 'romb':
            sigmasq[i] = (0.5 / np.pi ** 2) * sp.integrate.romb(integrand, dx=dlnk)
        if sigmasq[i] <= 0: print('error sigmasq: ', sigmasq[i], m, mass_to_radius(m, mean_dens))
        #print i,sigmasq[i],sp.integrate.trapezoid(integrand, dx=dlnk)
    return np.sqrt(sigmasq)

def xi_two(M1, M2, lnpk, lnk, mean_dens, scheme='trapezoid'):
    
    # If we input a scalar as M, then just make it a one-element list.
    if not isinstance(M1, collections.abc.Iterable):
        M1 = [M1]
    if not isinstance(M2, collections.abc.Iterable):
        M2 = [M2]


    dlnk = lnk[1] - lnk[0]
    sigmasq = np.zeros([len(M1), len(M2)])
    kcube_pk = np.exp(lnpk + 3 * lnk)
    for i, m1 in enumerate(M1):
        for j, m2 in enumerate(M2):
            integrand = kcube_pk * np.sqrt(top_hat_window(m1, lnk, mean_dens)) * np.sqrt(top_hat_window(m2, lnk, mean_dens))
            #print integrand
            if scheme == "trapezoid":
                sigmasq[i,j] = (0.5 / np.pi ** 2) * sp.integrate.trapezoid(integrand, dx=dlnk)
            elif scheme == "simpson":
                sigmasq[i,j] = (0.5 / np.pi ** 2) * sp.integrate.simpson(integrand, dx=dlnk)
            elif scheme == 'romb':
                sigmasq[i,j] = (0.5 / np.pi ** 2) * sp.integrate.romb(integrand, dx=dlnk)
            if sigmasq[i,j] <= 0: print('error xi_two: ', sigmasq[i,j], m1, mass_to_radius(m1, mean_dens), m2, mass_to_radius(m2, mean_dens))
            #print i,sigmasq[i],sp.integrate.trapezoid(integrand, dx=dlnk)
    return np.sqrt(sigmasq)

def top_hat_window(M, lnk, mean_dens):

    ## M is a scalar, lnk is an array
    kR = np.exp(lnk) * mass_to_radius(M, mean_dens)
    # # The following 2 lines cut the integral at small scales to prevent numerical error.
    Wsq = np.ones(len(kR))
    kR = kR[kR > 1.4e-6]
    if len(kR) > 0:
        Wsq[-len(kR):] = (3 * (np.sin(kR) / kR ** 3 - np.cos(kR) / kR ** 2)) ** 2
    return Wsq

def mass_to_radius(M, mean_dens):
    return (3.*M / (4.*np.pi * mean_dens)) ** (1. / 3.)

def radius_to_mass(R, mean_dens):
    return 4 * np.pi * R ** 3 * mean_dens / 3

def e_z(z, cdict):
    omega_M_0 = cdict["omega_M_0"]
    omega_lambda_0 = cdict.get("omega_lambda_0", None)
    if omega_lambda_0 is None:
        omega_lambda_0 = 1.0 - omega_M_0
    return np.sqrt(omega_M_0 * (1 + z) ** 3 + omega_lambda_0 + (1.0 - omega_M_0 - omega_lambda_0) * (1 + z) ** 2)


def d_plus(z, cdict):

    a_upper = 1.0 / (1.0 + z)
    lna = np.linspace(np.log(1e-8), np.log(a_upper), 1000)
    z_vec = 1.0 / np.exp(lna) - 1.0

    integrand = 1.0 / (np.exp(lna) * e_z(z_vec, cdict)) ** 3
    integral = sp.integrate.simpson(np.exp(lna) * integrand, dx=lna[1] - lna[0])
    dplus = 5.0 * cdict["omega_M_0"] * e_z(z, cdict) * integral / 2.0

    return dplus

def growth_factor(z, cdict):
    growth = d_plus(z, cdict) / d_plus(0.0, cdict)
    return growth

def dlnsigmadlnm(M, sgma, lnpk, lnk, mean_dens):
    
    dlnk = lnk[1] - lnk[0]
    R = mass_to_radius(M, mean_dens)
    dlnsigmadlnM = np.zeros_like(M)
    for i, r in enumerate(R):
        kr = np.exp(lnk) * r
        w = dw2dm(kr)  # Derivative of W^2
        integrand = w * np.exp(lnpk - lnk)
        dlnsigmadlnM[i] = (3.0 / (2.0 * sgma[i] ** 2 * np.pi ** 2 * r ** 4)) * sp.integrate.simpson(integrand, dx=dlnk)
    return dlnsigmadlnM

def dw2dm(kR):
    #The derivative of the top-hat window function squared
    return (np.sin(kR) - kR * np.cos(kR)) * (np.sin(kR) * (1 - 3.0 / (kR ** 2)) + 3.0 * np.cos(kR) / kR)


def power_to_corr(lnk, lnpk, R):

    if not np.iterable(R):
        R = [R]

    corr = np.zeros_like(R)

    # the number of steps to fit into a half-period at high-k. 6 is better than 1e-4.
    minsteps = 8

    # set min_k, 1e-6 should be good enough
    mink = 1e-6

    temp_min_k = 1.0

    tck = interpolate.splrep(lnk, lnpk, s=0)

    for i, r in enumerate(R):
        # getting maxk here is the important part. It must be a half multiple of
        # pi/r to be at a "zero", it must be >1 AND it must have a number of half
        # cycles > 38 (for 1E-5 precision).

        #min_k = (2 * np.ceil((temp_min_k * r / np.pi - 1) / 2) + 0.5) * np.pi / r
        #maxk = max(501.5 * np.pi / r, min_k)
        maxk = 501.5 * np.pi / r


        # Now we calculate the requisite number of steps to have a good dk at hi-k.
        nk = np.ceil(np.log(maxk / mink) / np.log(maxk / (maxk - np.pi / (minsteps * r))))
        #print 'nk = ', nk, maxk, mink, r
        lnkk, dlnkk = np.linspace(np.log(mink), np.log(maxk), nk, retstep=True)
        lnP = interpolate.splev(lnkk, tck, der=0)
        P = np.exp(lnP)
        integ = P * np.exp(lnkk) ** 2 * np.sin(np.exp(lnkk) * r) / r

        corr[i] = (0.5 / np.pi ** 2) * sp.integrate.simpson(integ, dx=dlnkk)

    return corr

