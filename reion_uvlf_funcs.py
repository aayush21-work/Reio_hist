import numpy as np
import scipy

import massfunction

# --- Unit conversions ---
YR    = 365*24*60*60    # seconds in a year
MYR   = 1e6*YR          # seconds in a Myr
PC    = 3.086e16         # metres in a parsec
MPC   = 1e6*PC           # metres in a Mpc
tconv = (MPC/1e3)/MYR   # convert inverse Hubble to Myr

# --- Physical constants (CGS) ---
mproton_cgs     = 1.672623e-24  # proton mass [g]
Msun_cgs        = 1.9891e33     # solar mass [g]
mproton         = mproton_cgs/Msun_cgs  # proton mass [Msun]
Mpc_cm          = 3.0856e24     # Mpc to cm
rho_crit_by_hsq = 2.7755e11     # critical density [h^2 Msun Mpc^-3]
sigmaT_cgs      = 6.65246e-25   # Thomson cross section [cm^2]
c_cgs           = 2.99792458e10 # speed of light [cm/s]

# --- Reionisation model parameters ---
alphaREC      = 2.6e-13        # case-B recombination coeff. at T_e=1e4 K [cm^3 s^-1]
eta_gamma_fid = 4.62175e60     # ionising photons per solar mass [Msun^-1]
clumping      = 3.0            # clumping factor

# --- Halo mass function settings ---
userhmf_model = 'ST_Jenkins'
usertf_model  = 'EH'

# --- UV luminosity function ---
KUV_fid   = 1.15485e-28  # SFR-to-UV luminosity conversion [(Msun/yr)(erg/s/Hz)^-1]
zUVLF_arr = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.5, 13.2])

# UVLF observation data files indexed by redshift
_UVLF_DATAFILES = {
     5.0: 'UVLF_datafiles/UVLF_z5p0.txt',
     6.0: 'UVLF_datafiles/UVLF_z6p0.txt',
     7.0: 'UVLF_datafiles/UVLF_z7p0.txt',
     8.0: 'UVLF_datafiles/UVLF_z8p0.txt',
     9.0: 'UVLF_datafiles/UVLF_z9p0.txt',
    10.0: 'UVLF_datafiles/UVLF_z10p0.txt',
    11.0: 'UVLF_datafiles/UVLF_z11p0.txt',
    12.5: 'UVLF_datafiles/UVLF_z12p5.txt',
    13.2: 'UVLF_datafiles/UVLF_z13p2.txt',
}


# =============================================================================
# UV LUMINOSITY FUNCTION CALCULATION
# =============================================================================


def get_log10_fstar_by_cstar_z(z, l0, l1, l2, l3):
    '''
    Equation 2.14 of arxiv:2404.02879
    '''

    return l0+l1*np.tanh((z-l2)/l3)


def get_alpha_star_z(z, a0, a1, a2, a3):
    '''
    Equation 2.15 of arxiv:2404.02879
    '''

    return a0+a1*np.tanh((z-a2)/a3)


def read_UVLF_data(filename):
    """
    Routine to read in UVLF observations data file
    """
    obsMUV_cen, obsPhiUV, obsPhiUV_hierr, obsPhiUV_loerr = np.loadtxt(
        filename, unpack=True, skiprows=2, usecols=(1, 2, 3, 4))
    obsDataset = np.loadtxt(filename, unpack=True,
                            skiprows=2, usecols=(0), dtype='U15')

    return obsMUV_cen, obsPhiUV, obsPhiUV_hierr, obsPhiUV_loerr, obsDataset


def read_QHI_data(filename):
    """
    Routine to read in QHI observations data file
    """
    obs_zarr, obs_xHI_arr, hi_sigmaxHI_arr, lo_sigmaxHI_arr = np.loadtxt(
        filename, unpack=True, skiprows=2, usecols=(1, 2, 3, 4))

    return obs_zarr, obs_xHI_arr, hi_sigmaxHI_arr, lo_sigmaxHI_arr


def Hub(z, omega_m, omega_l, h):
    """
    Routine to calculate the Hubble parameter at redshift z  ; units : (km/s) / Mpc

    """

    omega_k = 1 - omega_m - omega_l
    H0 = 100*h
    H_z = H0 * np.sqrt((omega_m)*(1 + z)**3 + omega_l + omega_k*(1 + z)**2)

    return H_z


def cal_MUV_neut_for_Mhalo(Mhalo, omega_m, omega_l, h, omega_b, *params):
    """
    Routine to calculate galaxy luminosity for NEUTRAL REGIONS for a given halo mass and set of parameters

    Mhalo is in Msun and the supplied LUV is in ergs/s/Hz
    """

    # unpack the individual parameter values

    fstar10_by_cstar, alpha_star, z, Mcrit = params
    # print(params)

    # should have been just fstar10 * ((Mhalo/10**10)**alpha_star) here
    eps_star_by_cstar = fstar10_by_cstar * ((Mhalo/(10**10))**alpha_star)

    baryonfrac = omega_b/omega_m

    Mstar = eps_star_by_cstar * baryonfrac * Mhalo  # units of Msun

    # Calculate SFR timescale and SFR value

    tHub_z = 1.0/Hub(z, omega_m, omega_l, h)

    tHub_z = tHub_z * tconv   # Convert from units of inverse Hubble to Mega-year

    tHub_z = tHub_z * 1e6   # Convert from units of inverse Hubble to year

    # This should have been  tstar= cstar* tHub_z  but cstar already accounted for in Mstar  via "eps_star_by_cstar"
    tstar = tHub_z

    SFR = Mstar/tstar  # units : Msun/yr ;  the combination eps_star_by_cstar at the start in Mstar  NOW takes care of everything

    LUV_1500 = SFR/KUV_fid     # LUV in units of ergs/s/Hz
    MUV_1500 = -2.5*np.log10(LUV_1500)+51.6

    return MUV_1500


def cal_MUV_ion_for_Mhalo(Mhalo, omega_m, omega_l, h, omega_b, *params):
    """
    Routine to calculate galaxy UV magnitude for IONIZED REGIONS for a given halo mass and set of parameters
    Mhalo is in Msun and the supplied LUV is in ergs/s/Hz
    """
    fstar10_by_cstar, alpha_star, z, Mcrit = params

    eps_star_by_cstar = fstar10_by_cstar * ((Mhalo/(10**10))**alpha_star)
    baryonfrac = omega_b/omega_m
    Mstar = eps_star_by_cstar * baryonfrac * Mhalo  

    tHub_z = 1.0/Hub(z, omega_m, omega_l, h)
    tHub_z = tHub_z * tconv * 1e6   
    tstar = tHub_z

    fgas = 2**(-Mcrit/Mhalo)
    SFR = (fgas*Mstar)/tstar

    LUV_1500 = SFR/KUV_fid     

    # Vectorized and safe log10 evaluation
    MUV_1500 = np.full_like(LUV_1500, 150.0)
    valid_mask = LUV_1500 > 1e-3
    MUV_1500[valid_mask] = -2.5 * np.log10(LUV_1500[valid_mask]) + 51.6

    return MUV_1500


def find_dMh_by_dMUV_neut(Mhalo, MUV, fstar10_by_cstar, alpha_star, Mcrit, z, omega_m, omega_l, h, omega_b):
    """
    Routine to find the derivative of halo mass w.r.t galaxy UV magnitude for NEUTRAL REGIONS

    The supplied Mhalo is in Msun and MUV is in mag

    """
    # Calculate SFR timescale and SFR value

    tHub_z = 1.0/Hub(z, omega_m, omega_l, h)  # should have been cstar * 1.0/Hub(z)  here

    tHub_z = tHub_z * tconv   # Convert from units of inverse Hubble to Mega-year

    tHub_z = tHub_z * 1e6   # Convert from units of inverse Hubble to year

    # Cosmoligical baryon fraction

    baryonfrac = omega_b/omega_m

    # neutral regions : Luv_neut =  (1/KUV_fid) *  (1/tstar)  * baryonfrac * [ fstar10 * ((Mhalo/(10**10))**alpha_star)) ] *  Mhalo

    # mass derivate of luminosity (dL/dM) ;  units :  (erg s^-1 Hz^-1).(M_sun)^-1
    dLUV_by_dMh = (fstar10_by_cstar/KUV_fid)*baryonfrac * (1/tHub_z)*(1+alpha_star) * \
        ((Mhalo/(10**10))**alpha_star)  # This is just (alpha_star+1)*LUV/Mh

    # units :  (erg s^-1 Hz^-1)^-1 (M_sun)
    dMh_by_dLUV = np.abs(1/dLUV_by_dMh)

    # derivative of AB magnitude wrt luminosity ; units : mag. (erg s^-1 Hz^-1)^-1
    # Remember : luv_neut =  ((fstar10_by_cstar/KUV_fid) * ((Mhalo/(10**10))**alpha_star)) * baryonfrac * Mhalo * (1/tHub_z)

    luv_neut = 10**(0.4*(51.6-MUV))

    dMUV_by_dLUV = np.abs(-2.5/(luv_neut*np.log(10)))

    dMh_by_dMUV = dMh_by_dLUV / dMUV_by_dLUV  # units :  M_sun . (mag)^-1

    return dMh_by_dMUV


def find_dMh_by_dMUV_ion(Mhalo, MUV, fstar10_by_cstar, alpha_star, Mcrit, z, omega_m, omega_l, h, omega_b):
    """
    Routine to find the derivative of halo mass w.r.t galaxy UV magnitude for IONISED REGIONS

    The supplied Mhalo is in Msun and MUV is in mag

    """
    # Calculate SFR timescale and SFR value

    tHub_z = 1.0/Hub(z, omega_m, omega_l, h)  # should have been cstar * 1.0/Hub(z)  here

    tHub_z = tHub_z * tconv   # Convert from units of inverse Hubble to Mega-year

    tHub_z = tHub_z * 1e6   # Convert from units of inverse Hubble to year

    # Cosmoligical baryon fraction

    baryonfrac = omega_b/omega_m

    # mass derivate of luminosity (dL/dM) ;  units :  (erg s^-1 Hz^-1).(M_sun)^-1

    # Ionized regions : Luv_ion =  (1/KUV_fid) *  (1/tstar)  * baryonfrac * [ fstar10 * ((Mhalo/(10**10))**alpha_star)) ] *  Mhalo  * (2**(Mcrit/Mhalo))

    # Ionized regions : Luv_ion =  (1/KUV_fid) *  (1/tstar) * baryonfrac * f_star(Mhalo) * Mhalo  * fgas(Mhalo)

    # derivative of " f_star(Mhalo)*Mhalo " part

    prefix1 = (fstar10_by_cstar/KUV_fid) * baryonfrac * \
        (1/tHub_z) * (2**(-Mcrit/Mhalo))

    term1 = prefix1 * (1+alpha_star)*((Mhalo/(10**10))**alpha_star)

    # derivative of " fgas(Mhalo) " part
    prefix2 = (fstar10_by_cstar/KUV_fid) * baryonfrac * \
        (1/tHub_z) * ((Mhalo/(10**10)) ** alpha_star) * Mhalo

    term2 = prefix2 * np.log(2) * (Mcrit / (Mhalo**2)) * (2**(-Mcrit/Mhalo))

    dLUV_by_dMh = term1 + term2

    # units :  (erg s^-1 Hz^-1)^-1 (M_sun)
    dMh_by_dLUV = np.abs(1/dLUV_by_dMh)

    # derivative of AB magnitude wrt luminosity ; units : mag. (erg s^-1 Hz^-1)^-1

    # luv_ion =  ((fstar10_by_cstar/KUV_fid) * ((Mhalo/(10**10))**alpha_star)) * baryonfrac * Mhalo * (1/tHub_z) *(2**(Mcrit/Mhalo))

    luv_ion = 10**(0.4*(51.6-MUV))

    dMUV_by_dLUV = np.abs(-2.5/(luv_ion*np.log(10)))

    dMh_by_dMUV = dMh_by_dLUV / dMUV_by_dLUV  # units :  M_sun . (mag)^-1

    return dMh_by_dMUV


def UVLF_model(log10_fstar10_by_cstar, alpha_star, log10Mcrit, QHI, MUV_arr, z, lnk, lnpk, cdict):
    """
    Routine to calculate the model UVLF in units of Mpc^-3 mag^-1 for each sample point in the parameter space
    """
    Mcrit = 10**log10Mcrit                                          
    fstar10_by_cstar = 10**log10_fstar10_by_cstar  

    nbins = len(MUV_arr)
    Mcool_z = get_Mcool(z)                                     
    log10Mcool = np.log10(Mcool_z, dtype=np.float32)

    # -------------------------------------------------------------------------
    # 1. Precompute HMF over the entire valid mass range
    # -------------------------------------------------------------------------
    log10Mmin_global = np.log10(Mcool_z * cdict["h"])
    log10Mmax_global = 15.5  # Upper bound safely beyond max possible mapped mass
    dlog10m_grid = 0.01
    
    M_grid_h, _, _, _, _, _, dndm_grid_h, _ = massfunction.get_massfunction(
        log10Mmin_global, log10Mmax_global, dlog10m_grid, z, lnk, lnpk, cdict, fit=userhmf_model, delta_c=1.686
    )

    # Convert to physical units
    M_grid_msun = M_grid_h / cdict["h"]
    dndm_grid_std = dndm_grid_h * (cdict["h"]**4)
    
    # Safe guard against zero values in dndm before log10
    dndm_grid_std = np.where(dndm_grid_std > 1e-100, dndm_grid_std, 1e-100)

    # Create log-log spline for HMF
    hmf_spline = scipy.interpolate.InterpolatedUnivariateSpline(
        x=np.log10(M_grid_msun), y=np.log10(dndm_grid_std), k=3, ext="extrapolate"
    )

    # -------------------------------------------------------------------------
    # 2. Map Halo Mass to MUV
    # -------------------------------------------------------------------------
    log10Mh_axis = np.arange(log10Mcool, 15, step=0.01)         
    fparams = (fstar10_by_cstar, alpha_star, z, Mcrit)

    # Neutral Spline
    MUV_neut_axis = cal_MUV_neut_for_Mhalo(10**log10Mh_axis, cdict["omega_M_0"], cdict["omega_L_0"], cdict["h"], cdict["omega_b_0"], *fparams)
    
    if np.all(np.diff(MUV_neut_axis) < 0):  
        MhMUVneut_spline = scipy.interpolate.InterpolatedUnivariateSpline(
            x=np.flip(MUV_neut_axis), y=np.flip(log10Mh_axis), k=3, ext="zeros")
    elif np.all(np.diff(MUV_neut_axis) > 0):  
        MhMUVneut_spline = scipy.interpolate.InterpolatedUnivariateSpline(
            x=MUV_neut_axis, y=log10Mh_axis, k=3, ext="zeros")
    else:  
        return MUV_arr, 1e30 * np.ones(nbins), 1e30 * np.ones(nbins), 1e30 * np.ones(nbins), 1e30 * np.ones(nbins), 1e30 * np.ones(nbins)

    # Ionized Spline
    MUV_ion_axis = cal_MUV_ion_for_Mhalo(10**log10Mh_axis, cdict["omega_M_0"], cdict["omega_L_0"], cdict["h"], cdict["omega_b_0"], *fparams)
    
    _ion_150 = np.argwhere(MUV_ion_axis == 150)
    if len(_ion_150) > 0:
        ion_idx = _ion_150[-1][0]
        MUV_ion_axis_cropped = MUV_ion_axis[ion_idx:]
        log10Mh_axis_cropped = log10Mh_axis[ion_idx:]

        if np.all(np.diff(MUV_ion_axis_cropped) < 0):
            MhMUVion_spline = scipy.interpolate.InterpolatedUnivariateSpline(x=np.flip(MUV_ion_axis_cropped), y=np.flip(log10Mh_axis_cropped), k=3, ext="zeros")
        elif np.all(np.diff(MUV_ion_axis_cropped) > 0):
            MhMUVion_spline = scipy.interpolate.InterpolatedUnivariateSpline(x=MUV_ion_axis_cropped, y=log10Mh_axis_cropped, k=3, ext="zeros")
        else:
            return MUV_arr, 1e30 * np.ones(nbins), 1e30 * np.ones(nbins), 1e30 * np.ones(nbins), 1e30 * np.ones(nbins), 1e30 * np.ones(nbins)
    else:
        if np.all(np.diff(MUV_ion_axis) < 0): 
            MhMUVion_spline = scipy.interpolate.InterpolatedUnivariateSpline(x=np.flip(MUV_ion_axis), y=np.flip(log10Mh_axis), k=3, ext="zeros")
        elif np.all(np.diff(MUV_ion_axis) > 0): 
            MhMUVion_spline = scipy.interpolate.InterpolatedUnivariateSpline(x=MUV_ion_axis, y=log10Mh_axis, k=3, ext="zeros")
        else: 
            return MUV_arr, 1e30 * np.ones(nbins), 1e30 * np.ones(nbins), 1e30 * np.ones(nbins), 1e30 * np.ones(nbins), 1e30 * np.ones(nbins)


    # -------------------------------------------------------------------------
    # 3. Vectorized UVLF Calculation (No 'for' loop)
    # -------------------------------------------------------------------------
    
    # Initialize output arrays
    Phi_UV_neut = np.zeros(nbins)
    Phi_UV_ion = np.zeros(nbins)
    
    # --- NEUTRAL REGIONS ---
    Mhalo_UV_neut = 10**MhMUVneut_spline(MUV_arr)
    valid_neut = Mhalo_UV_neut > 1.0  # Mask for valid masses

    if np.any(valid_neut):
        # Evaluate HMF spline
        dndm_MhaloUV_neut = 10**(hmf_spline(np.log10(Mhalo_UV_neut[valid_neut])))
        
        # Calculate Derivative 
        dMhbydMUV_neut = find_dMh_by_dMUV_neut(
            Mhalo_UV_neut[valid_neut], MUV_arr[valid_neut], fstar10_by_cstar, 
            alpha_star, Mcrit, z, cdict["omega_M_0"], cdict["omega_L_0"], 
            cdict["h"], cdict["omega_b_0"]
        )
        
        Phi_UV_neut[valid_neut] = dndm_MhaloUV_neut * dMhbydMUV_neut

    # --- IONIZED REGIONS ---
    Mhalo_UV_ion = 10**MhMUVion_spline(MUV_arr)
    valid_ion = Mhalo_UV_ion > 1.0

    if np.any(valid_ion):
        # Evaluate HMF spline
        dndm_MhaloUV_ion = 10**(hmf_spline(np.log10(Mhalo_UV_ion[valid_ion])))
        
        # Calculate Derivative
        dMhbydMUV_ion = find_dMh_by_dMUV_ion(
            Mhalo_UV_ion[valid_ion], MUV_arr[valid_ion], fstar10_by_cstar, 
            alpha_star, Mcrit, z, cdict["omega_M_0"], cdict["omega_L_0"], 
            cdict["h"], cdict["omega_b_0"]
        )
        
        Phi_UV_ion[valid_ion] = dndm_MhaloUV_ion * dMhbydMUV_ion

    # --- TOTAL UVLF ---
    Phi_UV_total = QHI * Phi_UV_neut + (1 - QHI) * Phi_UV_ion

    return MUV_arr, Phi_UV_total, Phi_UV_neut, Phi_UV_ion, Mhalo_UV_neut, Mhalo_UV_ion


# =============================================================================
# REIONISATION HISTORY CALCULATION
# =============================================================================


def dt_by_dz(z, omega_m, omega_l, h):
    """
    Routine to calculate dt/dz at redshift z  ; units : year
    """

    omega_k = 1 - omega_m - omega_l
    H0 = 100*h
    H_z = H0 * np.sqrt((omega_m)*(1 + z)**3 + omega_l + omega_k*(1 + z)**2)

    dtdz = -1.0/((1.0+z)*H_z)  # units : Mpc/ (km/s)

    dtdz = dtdz * tconv  # units : mega-year

    dtdz = dtdz * 1e6  # units : year

    return dtdz


def chi(zargs):
    """Factor accounting for helium reionisation (assumed at z = 3)."""
    return 1.08 if zargs >= 3.0 else 1.16


def get_Mcool(zargs, Tvir=1.e4, mu=0.59, omega_m=0.3, omega_l=0.7, h=0.7):
    '''
    Returns minimum mass of haloes (in units of Msun) at redshift zargs where the gas can cool via atomic transitions and form stars
    '''
    omega_k = 1-omega_m-omega_l
    omega_m_z = omega_m*(1+zargs)**3/(omega_m*(1+zargs) **
                                      3+omega_l+omega_k*(1+zargs)**2)

    d = (omega_m_z-1)
    Delta_crit = 18.0*np.pi**2+82*d-39*d*d

    # Eq (25) of Barkana & Loeb (2001)

    Mcool = (1.e8 / h) * (Tvir / 1.98e4) ** 1.5 * (0.6 / mu) ** 1.5 * (omega_m_z /
                                                                       omega_m) ** (1./2.) * (18 * np.pi ** 2 / Delta_crit) ** (1./2.) * (10 / (1 + zargs)) ** 1.5

    return Mcool


def integral_neut(alpha_star, alpha_esc, zinp, lnk, lnpk, cdict):

    Mcool_z = get_Mcool(zinp, omega_m=cdict["omega_M_0"], omega_l=cdict["omega_L_0"], h=cdict["h"])									# Msun

    # converting units of Mcool_z to Msun/h
    log10Mcoolh = np.log10(Mcool_z*cdict["h"])

    mass_arr, _, _, _, _, _, dndm_z_arr, _ = massfunction.get_massfunction(log10Mcoolh, 15, 0.1, zinp, lnk, lnpk, cdict, fit=userhmf_model, delta_c = 1.686)

    """
    mass_arr	                : units Msun/h
    dndm_z_arr	                : units: h^4 Msun^{-1} Mpc^{-3}

    """

    # units : Msun    ; array of halo masses greater than Mcool(zinp)
    m_arr = (mass_arr)/cdict["h"]

    # units : Msun^{-1} Mpc^{-3} ; array of dndm at zinp for halo masses greater than Mcool(zinp)
    dndm_arr = (dndm_z_arr) * (cdict["h"]**4)

    # quantity array for halo masses greater than Mcool(zinp)
    mpow_arr = np.power((m_arr/1e10), (alpha_star+alpha_esc))

    # integrand array contains halo masses greater than Mcool(zinp)

    integrand = mpow_arr*m_arr*dndm_arr

    integral = scipy.integrate.simpson(
        y=integrand, x=m_arr)  # units : Msun Mpc^{-3}

    return integral


def integral_ion(alpha_star, alpha_esc, zinp, Mcrit, lnk, lnpk, cdict):

    Mcool_z = get_Mcool(zinp, omega_m=cdict["omega_M_0"], omega_l=cdict["omega_L_0"], h=cdict["h"])									# Msun

    # converting units of Mcool_z to Msun/h
    log10Mcoolh = np.log10(Mcool_z*cdict["h"])

    mass_arr, _, _, _, _, _, dndm_z_arr, _ = massfunction.get_massfunction(log10Mcoolh, 15, 0.1, zinp, lnk, lnpk, cdict, fit=userhmf_model, delta_c = 1.686)

    """
    mass_arr	                : units Msun/h
    dndm_z_arr	                : units: h^4 Msun^{-1} Mpc^{-3}

    """

    # units : Msun    ; array of halo masses greater than Mcool(zinp)
    m_arr = (mass_arr)/cdict["h"]

    # units : Msun^{-1} Mpc^{-3} ; array of dndm at zinp for halo masses greater than Mcool(zinp)
    dndm_arr = (dndm_z_arr) * (cdict["h"]**4)

    # quantity array for halo masses greater than Mcool(zinp)
    mpow_arr = np.power((m_arr/1e10), (alpha_star+alpha_esc))

    # gas fraction array for halo masses greater than Mcool(zinp)
    fgas_arr = np.power(2.0, -1.0*np.divide(Mcrit, m_arr))

    # integrand array contains halo masses greater than Mcool(zinp)

    integrand = fgas_arr*mpow_arr*m_arr*dndm_arr

    integral = scipy.integrate.simpson(
        y=integrand, x=m_arr)  # units : Msun Mpc^{-3}

    return integral


def niondot_by_nH_neut(fstar10_by_cstar, fesc10, alpha_star, alpha_esc, zinp, lnk, lnpk, cdict):

    tHub = 1.0/Hub(zinp, cdict["omega_M_0"], cdict["omega_L_0"], cdict["h"])  # units : Mpc / (km/s)

    tHub = tHub * tconv  # units : mega-years

    tHub = tHub * 1e6  # units : year

    prefactor = (fstar10_by_cstar * fesc10 * eta_gamma_fid * mproton) / \
        ((1-cdict["YHe"])*tHub)  # year ^-1

    # units : Msun Mpc^{-3}  --- same units as rho_m
    intg_val_neut = integral_neut(alpha_star, alpha_esc, zinp, lnk, lnpk, cdict)

    rho_m = cdict["omega_M_0"] * rho_crit_by_hsq * cdict["h"]**2  # units : Msun Mpc^{-3}
    return prefactor*(intg_val_neut/rho_m)  # year^-1


def niondot_by_nH_ion(fstar10_by_cstar, fesc10, alpha_star, alpha_esc, zinp, Mcrit, lnk, lnpk, cdict):

    tHub = 1.0/Hub(zinp, cdict["omega_M_0"], cdict["omega_L_0"], cdict["h"])  # units : Mpc / (km/s)

    tHub = tHub * tconv  # units : mega-years

    tHub = tHub * 1e6  # units : year

    prefactor = (fstar10_by_cstar * fesc10 * eta_gamma_fid * mproton) / \
        ((1-cdict["YHe"])*tHub)  # year^-1

    # units : Msun Mpc^{-3}  --- same units as rho_m
    intg_val_ion = integral_ion(alpha_star, alpha_esc, zinp, Mcrit, lnk, lnpk, cdict)

    rho_m = cdict["omega_M_0"] * rho_crit_by_hsq * cdict["h"]**2  # units : Msun Mpc^{-3}
    return prefactor*(intg_val_ion/rho_m)  # units : year^-1


def get_QII_arr(z_arr, l0, l1, l2, l3, a0, a1, a2, a3, fesc10, alpha_esc, Mcrit, lnk, lnpk, cdict):
    '''
    Implementation of numerical derivative is as per Euler Backward Method 
    '''

    QII_arr = np.empty(len(z_arr))
    # for num=201, dz = 0.1 and for num = 401, dz = 0.05
    dz = np.float32(z_arr[1]-z_arr[0])
    QII_arr[0] = 0  # QII at z_arr[0] (z = 20) is taken to be 0.0

    # Precompute redshift-independent recombination quantities
    rho_m   = cdict["omega_M_0"] * rho_crit_by_hsq * cdict["h"]**2                                # Msun Mpc^{-3}
    n_H     = (1 - cdict["YHe"]) * (cdict["omega_b_0"] / cdict["omega_M_0"]) * (rho_m / mproton)  # Mpc^{-3}
    n_H_cgs = n_H / Mpc_cm**3                                                                      # cm^{-3}

    for i, z in enumerate(z_arr[:-1]):

        if (QII_arr[i] >= 1.0):

            QII_arr[i] = 1.0
            QII_arr[i+1] = 1.0

        else:
            z_next = z_arr[i+1]
            log10fstar10_by_cstar = get_log10_fstar_by_cstar_z(z_next, l0, l1, l2, l3)
            fstar10_by_cstar = 10.0**log10fstar10_by_cstar
            alpha_star = get_alpha_star_z(z_next, a0, a1, a2, a3)

            # Cache — each is used twice below
            dtdz_next    = dt_by_dz(z_next, cdict["omega_M_0"], cdict["omega_L_0"], cdict["h"])
            niondot_neut = niondot_by_nH_neut(fstar10_by_cstar, fesc10, alpha_star, alpha_esc, z_next, lnk, lnpk, cdict)

            # n_H_cgs : cm^{-3} and (alphaREC*YR) : cm^3 yr^{-1} as alphaREC : cm^3 sec^{-1}
            recomb_term  = chi(z_next) * n_H_cgs * (1.0 + z_next)**3 * alphaREC * YR * clumping   # year^-1

            niondot_ion  = niondot_by_nH_ion(fstar10_by_cstar, fesc10, alpha_star, alpha_esc, z_next, Mcrit, lnk, lnpk, cdict)

            numerator_term   = QII_arr[i] + dz * dtdz_next * niondot_neut
            denominator_term = 1.0 - dz * dtdz_next * (niondot_ion - niondot_neut - recomb_term)

            QII_arr[i+1] = numerator_term / denominator_term

    return QII_arr


def tau_integrand(QII_arr, z_arr, omega_m, omega_l, h):
    chi_arr   = np.where(z_arr >= 3.0, 1.08, 1.16)
    invHz_arr = tconv * MYR / Hub(z_arr, omega_m, omega_l, h)  # Mpc/(km/s) -> Myr -> secs
    return chi_arr * QII_arr * (1.0 + z_arr)**2 * invHz_arr   # units : secs


def get_tau(QII_arr, z_arr, cdict):

    # z_arr goes from 20 to 0 and QII_arr goes from 0 to 1
    tau_intgrnd = tau_integrand(QII_arr, z_arr, cdict["omega_M_0"], cdict["omega_L_0"], cdict["h"])

    integral = scipy.integrate.simpson(y=tau_intgrnd, x=z_arr)  # units : sec

    # c_cgs : cm / sec  ; n_H_cgs : cm^-3 ; sigmaT_cgs : cm^2 ; integrand : sec
    
    rho_m = cdict["omega_M_0"] * rho_crit_by_hsq * cdict["h"]**2  # units : Msun Mpc^{-3}
    n_H = (1-cdict["YHe"])*(cdict["omega_b_0"]/cdict["omega_M_0"])*(rho_m/mproton)  # units : Mpc^{−3}

    n_H_cgs = n_H * (Mpc_cm)**(-3)  # units : cm^-3

    # unitless - take absolute value to get rid of hassle of integration limits
    return np.abs(c_cgs*n_H_cgs*sigmaT_cgs*integral)


def reionHist_model(lsum, ldiff, l2, l3, asum, adiff, a2, a3, log10Mcrit, log10_fesc10, alpha_esc, lnk, lnpk, cdict):

    l0 = (lsum+ldiff)/2.0
    l1 = (lsum-ldiff)/2.0

    a0 = (asum+adiff)/2.0
    a1 = (asum-adiff)/2.0

    fesc10 = 10.0**log10_fesc10
    Mcrit = 10.0**log10Mcrit

    z_arr = np.linspace(20.0, 0.0, num=201)

    model_QII_arr = get_QII_arr(
        z_arr, l0, l1, l2, l3, a0, a1, a2, a3, fesc10, alpha_esc, Mcrit, lnk, lnpk, cdict)

    tau_model = get_tau(model_QII_arr, z_arr, cdict)

    return z_arr, model_QII_arr, tau_model

 #########################################################################################


def model_and_data_tau_elec(model_tau):
    """

    Routine to compute the chi-square for the observable : Electron Scattering Optical Depth (tau_el)

    """

    tau_e_obs = 0.054
    sigma_tau_e_obs = 0.007

    return np.array([model_tau]), np.array([tau_e_obs]), np.array([sigma_tau_e_obs])


def model_and_data_QHI(z_arr, QHII_arr):
    """

    Routine to compute the chi-square for the observable : neutral hydrogen fraction (Q_HI)

    """

    QHI_arr = 1.0-QHII_arr

    obs_zarr, obs_xHI_arr, hi_sigmaxHI_arr, lo_sigmaxHI_arr = read_QHI_data(
        'QHI_datafiles/fullQHIdata.txt')

    # Vectorised asymmetric-error -> symmetric (mean, sigma) conversion
    neg_mask  = lo_sigmaxHI_arr < 0.0
    data_arr  = np.where(
        neg_mask,
        obs_xHI_arr + 0.5 * (hi_sigmaxHI_arr + lo_sigmaxHI_arr),
        obs_xHI_arr + 0.5 * (hi_sigmaxHI_arr - lo_sigmaxHI_arr),
    )
    sigma_arr = np.where(
        neg_mask,
        0.5 * (hi_sigmaxHI_arr - lo_sigmaxHI_arr),
        0.5 * (hi_sigmaxHI_arr + lo_sigmaxHI_arr),
    )

    # Model values at each observed redshift
    indx_arr  = np.argmin(np.abs(obs_zarr[:, None] - z_arr[None, :]), axis=1)
    model_arr = QHI_arr[indx_arr]

    return model_arr, data_arr, sigma_arr


def model_and_data_eachUVLF(log10_fstar10_by_cstar, alpha_star, log10Mcrit, QHI_zval, zval, obsdatafile, lnk, lnpk, cdict):
    """

    Routine to compute the chi-square for each UV Luminosity Function (UVLF) at a given redshift "zval"

    """

    ################## Obtain the observational UVLF data for this particular redshift ##########################

    obsMUV_cen, obsPhiUV, obsPhiUV_hierr, obsPhiUV_loerr, obsDataset = read_UVLF_data(
        obsdatafile)

    ################## Obtain the model UVLF for this particular choice of parameters ##########################

    modelMUV_cen, modelUVLF_total, modelUVLF_neut, modelUVLF_ion, modelMhaloUV_neut, modelMhaloUV_ion = UVLF_model(
        log10_fstar10_by_cstar, alpha_star, log10Mcrit, QHI_zval, obsMUV_cen, zval, lnk, lnpk, cdict)

    ################## Build the chi squared function for this particular choice of parameters ##########################

    neg_mask  = obsPhiUV_loerr < 0.0
    data_arr  = np.where(
        neg_mask,
        obsPhiUV + 0.5 * (obsPhiUV_hierr + obsPhiUV_loerr),
        obsPhiUV + 0.5 * (obsPhiUV_hierr - obsPhiUV_loerr),
    )
    sigma_arr = np.where(
        neg_mask,
        0.5 * (obsPhiUV_hierr - obsPhiUV_loerr),
        0.5 * (obsPhiUV_hierr + obsPhiUV_loerr),
    )

    return modelUVLF_total, data_arr, sigma_arr


def model_and_data_allUVLF(QHI_UVLF_arr, log10Mcrit, lsum, ldiff, l2, l3, asum, adiff, a2, a3, log10_fesc10, alpha_esc, lnk, lnpk, cdict):
    """

    Routine to compute the chi-square for the observable : ALL UV Luminosity Functions (UVLF) in the range  6.0 <= z <= 13.2

    """
    l0 = (lsum+ldiff)/2.0
    l1 = (lsum-ldiff)/2.0

    a0 = (asum+adiff)/2.0
    a1 = (asum-adiff)/2.0

    model_parts, data_parts, sigma_parts = [], [], []

    model_z_arr, model_QHII_arr, model_tau_value = reionHist_model(
        lsum, ldiff, l2, l3, asum, adiff, a2, a3, log10Mcrit, log10_fesc10, alpha_esc, lnk, lnpk, cdict)

    model_QHI_UVLF_arr = np.empty(len(zUVLF_arr))

    for i, zdata in enumerate(zUVLF_arr):

        loop_indx = np.argmin(np.abs(zdata - model_z_arr))
        QHI = 1 - model_QHII_arr[loop_indx]

        # QHI at each redshift where model UVLF is compared with data
        model_QHI_UVLF_arr[i] = QHI

        log10fstar10_by_cstar = get_log10_fstar_by_cstar_z(zdata, l0, l1, l2, l3)
        alpha_star = get_alpha_star_z(zdata, a0, a1, a2, a3)

        QHI = model_QHI_UVLF_arr[i]
        obsdatafile = _UVLF_DATAFILES[zdata]

        m, d, s = model_and_data_eachUVLF(
            log10fstar10_by_cstar, alpha_star, log10Mcrit, QHI, zdata, obsdatafile, lnk, lnpk, cdict)
        model_parts.append(m)
        data_parts.append(d)
        sigma_parts.append(s)

    return np.concatenate(model_parts), np.concatenate(data_parts), np.concatenate(sigma_parts)


def get_chisq(model_arr, data_arr, sigma_arr):

    return np.sum(((data_arr - model_arr) / sigma_arr) ** 2)


def model_and_data(lsum, ldiff, l2, l3, asum, adiff,  log10_fesc10, alpha_esc, log10Mcrit):

    # TIE UP the transition redshift and transition redshift interval for log10_fstar10_by_cstar and alpha_star together

    a2 = l2
    a3 = l3

    ################## Generate reionisation history for this particular choice of parameters ##########################

    model_z_arr, model_QHII_arr, model_tau_value = reionHist_model(
        lsum, ldiff, l2, l3, asum, adiff, a2, a3, log10Mcrit, log10_fesc10, alpha_esc)

    model_QHI_UVLF_arr = np.empty(len(zUVLF_arr))

    for i, zdata in enumerate(zUVLF_arr):

        loop_indx = np.argmin(np.abs(zdata - model_z_arr))
        QHI = 1 - model_QHII_arr[loop_indx]

        # QHI at each redshift where model UVLF is compared with data
        model_QHI_UVLF_arr[i] = QHI

    ################## Build the model and data arrays for this particular choice of parameters ##########################

    model_allUVLF_arr, data_allUVLF_arr, sigma_allUVLF_arr = model_and_data_allUVLF(
        model_QHI_UVLF_arr, log10Mcrit, lsum, ldiff, l2, l3, asum, adiff, a2, a3, log10_fesc10, alpha_esc)

    model_QHI_arr, data_QHI_arr, sigma_QHI_arr = model_and_data_QHI(
        model_z_arr, model_QHII_arr)

    model_tau_arr, data_tau_arr, sigma_tau_arr = model_and_data_tau_elec(
        model_tau_value)

    model_arr = np.concatenate([model_allUVLF_arr, model_QHI_arr, model_tau_arr])
    data_arr  = np.concatenate([data_allUVLF_arr,  data_QHI_arr,  data_tau_arr])
    sigma_arr = np.concatenate([sigma_allUVLF_arr, sigma_QHI_arr, sigma_tau_arr])

    return model_arr, data_arr, sigma_arr
