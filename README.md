# Reionization History Pipeline

This repository computes and constrains the epoch of reionization for a fixed
(Planck-like) cosmology, using **two complementary tracks**:

1. **A numerical track** (`run_pipeline.py`): CLASS -> MUSIC -> `script`
   simulation pipeline that builds reionization boxes from N-body initial
   conditions and extracts `Q_HII(z)`/`x_e(z)`.
2. **An analytic + MCMC track** (`reion_uvlf_*`, `mcmc_uvlf_*`): reionization model driven by a halo mass function and the UV luminosity
   function (UVLF), whose free astrophysics parameters are constrained against
   UVLF + `Q_HI` observations and the Planck 2018 CMB likelihoods with Cobaya.

Both tracks feed the resulting `x_e(z)` reionization history into
**CLASS** (via the `reio_inter` parameterization) to compute the CMB power
spectra and the reionization optical depth `tau_e`.

---

## Table of contents

- [Pipeline overview](#pipeline-overview)
- [Repository layout](#repository-layout)
- [Setup & build](#setup--build)
- [Config files reference](#config-files-reference)
- [Track A: numerical pipeline](#track-a-numerical-pipeline)
- [Track B: analytic model & MCMC](#track-b-analytic-model--mcmc)
- [Cobaya wiring](#cobaya-wiring)
- [Notebooks](#notebooks)
- [Utility & test scripts](#utility--test-scripts)
- [File reference](#file-reference)

---

## Pipeline overview

```
                            global.ini (cosmology + [script] params)
                                     |
             +-----------------------+------------------------+
             |                                                |
   Track A (numerical)                    Track B (analytic + Cobaya)
             |                                                |
     CLASS reiotest.ini                      CLASS P(k)  (classy / Class)
             |  z_pk ladder                       |
     reformat_tk -> camb13.dat                    v
             |                              massfunction + transferfunction
      MUSIC reiotest.conf                   (halo mass function, sigma, D+)
             |  zstart ladder                       |
     ics.gadget_z.dat                        reion_uvlf_funcs
             |                             (fstar(z), alpha_star(z), UVLF,
             v                              QHII(z), tau_e)
      script  (reionization boxes)                  |
             |                                      v
     Q_HII(z)  ->  x_e(z) = Q_HII * f_He        x_e(z)  ->  reio_inter_*
             |                                      |
             +----------------+---------------------+
                              v
                 CLASS final (reiotest_1.ini, reio_inter)
                              v
              C_l  (TT/EE/TE/BB)  +  tau_e
```

The end result of either track is an `x_e(z)` curve written into the
`reio_inter` block of `class_public/reiotest_1.ini`, so CLASS recomputes the
CMB spectra (and `tau_e`) with the model's reionization history.

---

## Repository layout

| path | purpose |
|---|---|
| `class_public/` | CLASS (cloned by `build.sh`; compiled binary `./class`, Python binding `classy`). Gitignored. |
| `music/` | MUSIC initial-conditions generator (cloned by `build.sh`). Gitignored. |
| `script/` | "script" reionization-box package (cloned by `build.sh`, pip-installed). Gitignored. |
| `script_files/` | scratch output of `script` (matter fields, ionization maps per redshift). Gitignored. |
| `data_files/` | observation data: `UVLF_datafiles/`, `QHI_datafiles/`, `Planck_EE_unbinned.txt`. Gitignored. |
| `cobaya_packages/` | Cobaya external packages (e.g. the Planck 2018 `.clik` likelihoods and data). Gitignored. |
| `chains/` | Cobaya MCMC output (`.txt` samples, `.covmat`, `.checkpoint`, `.input.yaml`, ...). Gitignored. |
| `*.py` | pipeline modules, MCMC infos, plotting/test scripts (see [file reference](#file-reference)). |
| `check.ipynb`, `check_9params_point.ipynb` | interactive walkthrough notebooks. |
| `global.ini` | cosmology + `[script]` parameters (the single input you edit). |
| `reiotest.ini`, `reiotest_1.ini`, `reiotest.conf` | CLASS/MUSIC template configs. |
| `pipeline.log`, `xe_history.dat`, `xe_history_full.dat` | generated during a Track A run. |

**Working-directory convention:** most modules (`reion_uvlf_funcs.py`,
`mcmc_uvlf_cmb.py`, `mcmc_uvlf_9params.py`) read their data files through
*relative* paths from `data_files/` (several `os.chdir(DATA_DIR)` at import).
Please run scripts and notebooks **from the repository root** .
---

## Setup & build

Requirements (see `requirements.txt`):

- Linux (the build scripts and compiled binaries are Linux-only)
- Should support Python >= 3.12 (repo developed on 3.14), `pip`
-  `gcc` and `make` required for CLASS and MUSIC

```bash
# 1) install python deps
pip install -r requirements.txt

# 2) clone + build CLASS, MUSIC, and the 'script' package
./build.sh

# 3) install the Planck 2018 likelihood data into cobaya_packages/
#    (first run is the slow one; it downloads the .clik files)
cobaya-install planck_2018 planck_2018_lowl.TT planck_2018_lowl.EE \
    planck_2018_highl_plik.TTTEEE_lite -p cobaya_packages
```

What `build.sh` does:

1. Clones `class_public` (lesgourg), `script` (rctirthankar/bitbucket),
   `music` (ohahn/bitbucket) into the repo root if absent.
2. Patches CLASS's `include/parser.h`: `_LINE_LENGTH_MAX_` and
   `_ARGUMENT_LENGTH_MAX_` 1024 -> 8192 (the `reio_inter_z`/`reio_inter_xe`
   tables are long comma-string lines).
3. `make clean && make -j` in `class_public`, then `pip install .` in
   `class_public/python` (installs `classy`), then copies the `external/`
   runtime data into the installed `classy` package directory.
4. `make` in `music`.
5. `pip install .` in `script`.
6. Clones the repo's template configs into the build trees:
   `reiotest.ini`/`reiotest_1.ini` -> `class_public/`,
   `reiotest.conf` -> `music/`.

`setup_pipeline.sh` is an older, simpler variant of the same steps (clones,
`make -j` in CLASS, `make` in MUSIC, pip-installs `script`, and **moves** the
ini files into place). Prefer `build.sh`.



---

## Config files reference

### `global.ini`

The single user-facing input for the numerical track.

```ini
[cosmology]
h = 0.6736
omega_b = 0.0493          # Omega_b (not omega_b = Omega_b*h^2)
omega_cdm = 0.2645
omega_m = 0.3138
n_s = 0.9649
a_s = 2.1e-9
tau_reio = 0.0544
t_cmb = 2.7255
sigma_8 = 0.8111
pk_ini_type = analytic_Pk      # 'analytic_Pk' or 'external_pk'

[script]
zeta = 14.929192           # ionizing efficiency (photons per (collapsed H / M))
log10_mmin = 8.9733792     # log10 min halo mass for star formation
ngrid = 32                 # grid cells per side for the reionization boxes
helium_factor = 1.0789     # x_e = Q_HII * f_He for z >= 3
helium_factor_lowz = 1.1578# x_e = Q_HII * f_He for z < 3
```

Notes:

- `omega_b`/`omega_cdm` *here* are densities, not `Omega*h^2`. `run_pipeline.py`
  converts them (`Omega_b*h^2`, `Omega_cdm*h^2`) when writing the CLASS ini.
- The MCMC track uses its own **fixed** cosmology (see below), independent of
  `global.ini`.
- `[script]` values are re-written by `test_cobaya_planck.py` (`set_global`)
  when it drives the SCRIPT model from Cobaya.

### CLASS / MUSIC templates

- `reiotest.ini` -> `class_public/reiotest.ini`: CLASS input for the transfer
  function (P(k)) builds. `run_pipeline.py` writes a per-run `reiotest.ini_run.ini`
  appending `h`, `omega_b`, `omega_cdm`, `T_cmb`, and either `Pk_ini_type`,
  `command` (external) or `n_s`, `A_s` (analytic).
- `reiotest_1.ini` -> `class_public/reiotest_1.ini`: CLASS input for the **final**
  CMB computation. `run_pipeline.py` / the MCMC infra append/refresh the
  `reio_inter` block in this file.
- `reiotest.conf` -> `music/reiotest.conf`: MUSIC parameter file. A per-run
  `reiotest.conf_run.conf` is generated with `[cosmology]` and per-z `zstart`.

The `reio_inter` block looks like:

```
# --- BEGIN reio_inter ---
reio_parametrization = reio_inter
reio_inter_num = <n>
reio_inter_z = 0, 0.4, ..., 4.8, 5, 5.4, ...
reio_inter_xe = 1.157800, 1.157800, ..., 1.078900, 0.500000, ...
# --- END reio_inter ---
```

---

## Track A: numerical pipeline

`run_pipeline.py` computes reionization directly from simulated density fields.
Run it with:

```bash
python run_pipeline.py
```

The redshift ladder is `z = 5.0, 5.4, ..., 20.0` (`zmin=5`, `zmax=20`,
`step=0.4`), controlled by module constants at the top of the file.

### Main flow (`main()`)

1. **`make_class`** - copy `reiotest.ini`, append the cosmology from
   `global.ini` (`h`, `omega_b`, `omega_cdm`, `T_cmb`, and `n_s`/`A_s` for
   `analytic_Pk`, or `Pk_ini_type`+`command` for `external_pk`).
2. **`make_music`** - copy `reiotest.conf`, append `[cosmology]` (`Omega_m`,
   `Omega_b`, `Omega_L = 1-Omega_m`, `H0`, `sigma_8`, `nspec`).
3. **`run_pipeline(class_ini, music_conf)`** - loop over the z-ladder:
   - run `./class` with `z_pk = z`;
   - convert the transfer-function table `reiotest.ini_run_tk.dat` into
     `output_camb13.dat` (MUSIC format) with `reformat_tk`;
   - run `./MUSIC` with `zstart = z` and the freshly written transfer file;
   - rename the output to `ics.gadget_<z>.dat`.
4. **`run_script_parallel()`** - for each snapshot, use the `script` package:
   - `script.default_simulation_data(...)` (options from `global.ini`);
   - `matter_fields` -> `ionization_map`;
   - collapsed fraction `fcoll = get_fcoll_for_Mmin(log10_Mmin)`;
   - ionized fraction `qi = get_qi(zeta * fcoll)`;
   - `Q_HII = mean(qi * (1 + delta))` (mass-weighted);
   - `x_e = Q_HII * helium_factor` (z >= 3) or `* helium_factor_lowz` (z < 3).
   Sanity checks: reionization must **end** by `z=5` (`Q_HII >= 0.99`) and must
   **start** at `z=20` (Q_HII drops to >= 0). Writes `xe_history.dat`
   (columns `z Q_HII x_e`).
5. **`make_reio_inter()`** - build the `reio_inter` table for CLASS:
   - take `xe_history.dat`, drop the fractional low-z tail to zero and append a
     zero anchor at `z_last + step`;
   - prepend the low-z plateau `z = 0..4.8` with `x_e = helium_factor[_lowz]`;
   - write `xe_history_full.dat` (columns `z x_e`);
   - insert/refresh the `reio_inter` block in `reiotest_1.ini`.
6. **`run_class_final()`** - run `./class reiotest_1.ini`, parse and print
   `tau_reio` from the CLASS output (`optical depth = ...`).

### Helper APIs (for scripts/notebooks)

- `reionization_history(zeta, log10_Mmin, ...)` - per-z `(z, Q_HII, x_e)` list
  via a worker pool.
- `get_class_cl(history)` - recompute C_l (TT/EE/TE/BB) for a given history by
  refreshing the `reio_inter` block in `reiotest_1.ini` and running CLASS;
  returns `ell, cl_tt, cl_ee, cl_te, cl_bb`.
- `run_pipeline_parallel` / `_gen_one_z` - parallel CLASS+MUSIC snapshot
  generation across the z-ladder.

Outputs: `pipeline.log` (full run log), `xe_history.dat`,
`xe_history_full.dat`, `script_files/`, `class_public/output/`,
`music/ics.gadget_<z>.dat`.

---

## Track B: analytic model & MCMC

### The model

The astrophysical model is implemented in `reion_uvlf_funcs.py`. Given a
z=0 linear power spectrum it computes:

- the **halo mass function** via `massfunction.py` / `transferfunction.py`
  (Sheth-Tormen/Jenkins style, Eisenstein-Hu transfer, ellipsoidal collapse);
- the **star-formation efficiency** redshift evolution
  `f_star(z)/c_star` (ionizing-photon escape normalised) and the **stellar
  spectral slope** `alpha_star(z)`;
- the **UV luminosity function** `phi(M_UV)` at UVLF observation redshifts
  (`z = 5, 6, 7, 8, 9, 10, 11, 12.5, 13.2`) and the **ionized-fraction
  evolution** `Q_HII(z)` from the ionizing budget;
- derived **`tau_e`** by integrating `Q_HII(z)` (with helium corrections).

The redshift evolutions are polynomials in `z`:

- `log10(f_star,10/c_star)  = l0 + l1*z + l2*z^2 + l3*z^3`
- `alpha_star              = a0 + a1*z + a2*z^2 + a3*z^3`

The MCMC parameters are the **sum/difference** combinations:

```
l0 = (lsum + ldiff)/2,   l1 = (lsum - ldiff)/2
a0 = (asum + adiff)/2,   a1 = (asum - adiff)/2
```

with `l2,l3` shared as `a2=l2`, `a3=l3`. `2*l1` is the total jump in
`log10(f_star/c_star)`; `2*a1` in `alpha_star`. Fixed astrophysical constants
live at the top of `reion_uvlf_funcs.py` .

### Parameter sets

| symbol | prior (min, max) | ref | meaning |
|---|---|---|---|
| `lsum` | (-2, 2) | -0.2 | l0+l1 combination |
| `ldiff` | (-2, 1) | -0.75 | l0-l1 combination |
| `l2` | (8, 18) | 13 | z^2 coeff of log10(f_star/c_star) |
| `l3` | (-3, 6) | 2.16 | z^3 coeff of log10(f_star/c_star) |
| `asum` *(9p only)* | (0, 2) | 0.944 | a0+a1 combination |
| `adiff` *(9p only)* | (-2, 2) | 0.289 | a0-a1 combination |
| `log10_fesc10` *(9p only)* | (-2, 2) | -0.812 | log10 escape fraction at z=10 |
| `alpha_esc` *(9p only)* | (-2, 2) | -0.0782 | escape-fraction redshift scaling |
| `log10Mcrit` *(9p only)* | (8, 15) | 10.17 | log10 critical halo mass |

In the **4-param** model (`reion_uvlf_4params.py`) the last five are **fixed**
to `asum=0.94376, adiff=0.28934, log10_fesc10=-0.8122, alpha_esc=-0.078249,
log10Mcrit=10.174`.

Cosmology: the MCMC samples only `n_s` (uniform 0.92-1.00) and
`logA = ln(10^10 A_s)` (uniform 2.5-3.5, dropped/derived). Everything else is
**fixed**: `H0 = 67.4`, `omega_b = 0.0224`, `omega_cdm = 0.120`,
`A_s = 1e-10 * exp(logA)`, `N_ur = 3.044`. Derived: `tau_e` (from the model),
`A_s`, `l0/l1/a0/a1`.

### Log-likelihood

Both `reion_uvlf_4params.py` and `reion_uvlf_9params.py` expose

```python
log_likelihood(..., lnk, lnpk, omega_m, omega_l, h, omega_b, YHe=0.24)
    -> (logL, derived)
```

with `lnk = log(k / h)` (h/Mpc), `lnpk = log(P(k) * h^3)` at z=0.



### Running an MCMC

```bash
python mcmc_uvlf_9params.py   # 9-param run, output chains/uvlf_cmb_9params
python mcmc_uvlf_cmb.py       # 4-param run,  output chains/uvlf_cmb_without_uvlf
```

Each module, run as `__main__`, calls `from cobaya.run import run; run(info,
allow_changes=True)`. Chains (with `.covmat`, `.checkpoint`, `.input.yaml`,
`.progress`) accumulate under `chains/` and `resume=True` allows continuing.
`test_cobaya_planck.py` is a separate, self-contained Cobaya example whose
theory (`ScriptReio`) drives the *numerical* `script` model from `zeta` /
`log10_Mmin` inside the MCMC (output `chains/classy_planck_TTEE`).

---

## Cobaya wiring

Both `mcmc_uvlf_*.py` files define an `info` dict with the same three pieces:

- **theory `uvlf_reio`** (`UVLFReio`): builds the CLASS `P(k)` (output `mPk`,
  `P_k_max_1/Mpc = 100`, `k_per_decade_for_pk = 20`, `z_max_pk = 1`, uniform
  `kk = logspace(-4, log10(100), 500)`), calls the UVLF `log_likelihood`, then
  turns the model `QHII(z)` into an `x_e(z)` history:
  - sort ascending in z; require `Q_HII(z=5) >= 0.99` else reject the step;
  - helium-correct (`* 1.0789` for z >= 3, `* 1.1578` below);
  - clamp the tiny high-z tail to zero and append exactly one zero anchor
    (`z_last + 0.4`) so z and x_e stay equal length;
  - provides `reio_inter_z`, `reio_inter_xe`, `reio_inter_num`, `uvlf_logl`.
- **theory `classy_reio`**: subclass of Cobaya's `classy` that
  `get_requirements()` the `reio_inter_*` entries, injects them into CLASS's
  `extra_args` with `reio_parametrization = reio_inter` each step, and delegates
  to the normal classy calculation. This recomputes the CMB spectra
  (`l_c_l`) with the step's reionization history.
- **likelihoods**:
  - `uvlf` (`UVLFLike`) - returns `uvlf_logl`, i.e. `-chi2/2` from the
    UVLF + Q_HI model;
  - `planck_2018_lowl.TT`, `planck_2018_lowl.EE`,
    `planck_2018_highl_plik.TTTEEE_lite` - Planck 2018 (native Cobaya,
    resolved from `cobaya_packages/`).

`sampler: mcmc` (`Rminus1_stop 0.01`, `learn_proposal True`,
`max_samples 200000`) and `output: chains/...` are set in `info`.

The `plot_cls_reio.py` script demonstrates the same machinery outside an MCMC:
it calls `mcmc_uvlf_cmb`'s model with `model.loglikes(point, as_dict=True,
return_derived=True)` for selected galaxy points (the point must include the
sampled params `n_s` and `logA`).

---

## Notebooks

### `check.ipynb` - 4-param plumbing walkthrough

Runs from the repo root, section by section:

1. **Path setup** - `PROJECT`/`DATA_DIR` from `os.getcwd()`.
2. **P(k) grid resolution & sigma8** - builds CLASS P(k); evaluates
   `massfunction` sigma(M) on uniform vs non-uniform log-k grids to check grid
   convergence; reports `sigma_8`.
3. **UVLF model checks** - `os.chdir(DATA_DIR)`, then:
   - grid choice: non-uniform vs uniform log-k `lnk/lnpk`;
   - **split the UVLF and QHI chi2** for a parameter choice with `classy`;
   - parameter scans over `log10Mcrit` and the `alpha_star` normalization
     (`a0 = (asum+adiff)/2`).
4. **Cobaya Planck likelihoods** - `os.chdir(PROJECT)`, builds the 4-param
   Cobaya model, evaluates it for two "modes" (mode 1: UVLF-only best fit
   `lsum=-0.458, ldiff=-0.883, l2=10.78, l3=1.148`; mode 2: joint fit
   `lsum=2.0068, ldiff=-0.9335, l2=16.42, l3=4.510`), each with the fixed
   sampled cosmology `n_s=0.965, logA=3.043`, and reports per-likelihood chi2 +
   `tau_e`.
5. **Reionization history -> C_l** - `os.chdir(DATA_DIR)`, refreshes
   `reio_inter` in `class_public/reiotest_1.ini`, runs CLASS, plots the power
   spectra (`ell < LMAX`) vs Planck EE (unbinned) with fractional differences,
   plots the reionization histories, and checks the truncation of the high-z
   tail.
6. Append block, **`check_9params_point.ipynb` content** - the 9-param
   single-point evaluation (see below) is also embedded verbatim here.

### `check_9params_point.ipynb` - single-point 9-param evaluation

Edits the `POINT` cell, then runs all cells to evaluate **exactly** what
`mcmc_uvlf_9params.py` computes for one set of parameters:

- builds the Cobaya model from `mcmcmod.info` (stripped of
  `sampler`/`output`/`resume`);
- calls `model.logposterior(POINT, return_derived=True)` and prints the
  full chi2 breakdown (UVLF, Planck lowl TT/EE, highl TTTEEE lite), plus
  derived `tau_e`, `A_s`;
- plots the `x_e(z)` history that was fed into CLASS;
- rebuilds P(k) once and splits the astro model chi2 into the UVLF and
  `Q_HI` contributions, plotting `Q_HII(z)`/`Q_HI(z)`.

---

## Utility & test scripts

| script | what it does |
|---|---|
| `test.py` | smoke test of the `script` package on a MUSIC snapshot (`ics.gadget_5.0.dat`); computes fcoll, qi, and an (R)SD 21-cm power spectrum to `output_pipeline/`. |
| `test_cobaya_planck.py` | self-contained Cobaya example: `ScriptReio` theory drives `run_pipeline`'s SCRIPT machinery from `zeta`/`log10_Mmin`; a `reio_complete_veto` likelihood enforces `Q_HI(z->0) >= 0.99`; classic planck lowl TT/EE compare; output `chains/classy_planck_TTEE`. |
| `plot_cls_reio.py` | compares CLASS C_l (EE) for two galaxy parameter sets against `data_files/Planck_EE_unbinned.txt`, using the UVLF model for the reio history and the Cobaya model for per-likelihood chi2 (`get_like`). |
| `plot_data_compare.py` | overlays the UVLF model against the UVLF data (3x3, one panel per UVLF redshift) and the `Q_HI(z)` model against `QHI_datafiles/fullQHIdata.txt` for two parameter sets. |
| `plot_planckTTEE.py` | GetDist triangle plot of `chains/classy_planck_TTEE` (`zeta`, `log10_Mmin`, `tau_reio`) with the Planck `tau = 0.0544 +/- 0.0073` band highlighted; exports `<chain>_bestfit_triangle.png`. |
| `get_cl.py` | loads `chains/classy_planck_TTEE`, randomly draws 200 accepted samples, recomputes the C_l for each with `run_pipeline.get_class_cl` and plots TT/EE coloured by chi2 against the Planck TT/EE data. |
| `check_tau.py` | deprecated helper; reads `chains/tau_mcmc.1.txt` and prints best-fit rows. |

---

## File reference

| file | role |
|---|---|
| `run_pipeline.py` | Track A orchestration: CLASS transfer files, MUSIC IC generation, `script` Q_HII/x_e computation, `reio_inter` injection, final CLASS C_l/tau. |
| `reion_uvlf_funcs.py` | core analytic model: f_star/c_star and alpha_star redshift evolution, UVLF and Q_HI observables, ionizing budget, `tau_e`, chi2 helpers. |
| `massfunction.py` | halo mass function (`ST_Jenkins`, ellipsoidal collapse), `begin_step()` cache reset, `USE_CACHED_HMF` flag. |
| `transferfunction.py` | sigma(M), dlnsigma/dlnM, growth factor, Eisenstein-Hu transfer; per-step vectorized HMF cache (`cached_sigma`, `cached_dlnsigmadlnm`, `cached_growth_factor`, `_build_step_cache`). |
| `reion_uvlf_4params.py` | 4-param `log_likelihood` (lsum/ldiff/l2/l3 free, rest fixed) returning `(logL, derived)` incl. `tau_e`. |
| `reion_uvlf_9params.py` | 9-param `log_likelihood` (all astrophysics free). |
| `mcmc_uvlf_cmb.py` | Cobaya info + `UVLFReio`/`UVLFLike`/`classy_reio` for the 4-param run (Planck 2018 lowl TT/EE + highl TTTEEE lite). |
| `mcmc_uvlf_9params.py` | same wiring for the 9-param run. |
| `test_cobaya_planck.py` | Cobaya example coupling `zeta`/`log10_Mmin` to the numerical SCRIPT track. |
| `plot_cls_reio.py`, `plot_data_compare.py`, `plot_planckTTEE.py`, `get_cl.py`, `check_tau.py`, `test.py` | visualization / diagnostics (see above). |
| `global.ini` | cosmology + `[script]` inputs for Track A. |
| `reiotest.ini`, `reiotest_1.ini`, `reiotest.conf` | CLASS / MUSIC templates (synced into the build trees by `build.sh`). |
| `requirements.txt` | python dependencies. |
| `build.sh`, `setup_pipeline.sh` | clone + compile + pip-install + config relocation. |
| `check.ipynb`, `check_9params_point.ipynb` | walkthrough/evaluation notebooks. |
| `OPTIMISATION_NOTES_9PARAMS.md` | detailed write-up of the HMF cache + CLASS-resolution optimisations and their verification. |
| `data_files/UVLF_datafiles/` | UVLF observational LF per redshift (`UVLF_z5p0.txt` ... `UVLF_z13p2.txt`). |
| `data_files/QHI_datafiles/fullQHIdata.txt` | observed neutral-fraction curve `Q_HI(z)`. |
| `data_files/Planck_EE_unbinned.txt` | Planck EE unbinned spectrum used for the low-l EE comparison plots. |

