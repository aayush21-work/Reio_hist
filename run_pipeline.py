import os
import configparser
import datetime
import re
import shutil
import subprocess
from pathlib import Path
import numpy as np
import script
import sys
from multiprocessing import Pool

PROJECT_DIR     = Path(__file__).resolve().parent
 
CLASS_DIR       = PROJECT_DIR/"class_public"
CLASS_INI       = CLASS_DIR/"reiotest.ini"          
CLASS_OUT       = CLASS_DIR/"output"
CLASS_FIN       = CLASS_DIR/"reiotest_1.ini"
 
MUSIC_DIR       = PROJECT_DIR/"music"
MUSIC_CONF      = MUSIC_DIR/"reiotest.conf"       
 
SNAP_OUTDIR     = PROJECT_DIR/"script/examples/snapshots"
LOG_PATH        = PROJECT_DIR/"pipeline.log"
GLOBAL_INI      = PROJECT_DIR/"global.ini"

SCRIPT_FILES = PROJECT_DIR/"script_files"    
 
zmin=5
zmax=20
step=0.4
ZLADDER = ZLADDER = np.round(np.arange(zmin , zmax+step, step), 1)

def log(msg, path=LOG_PATH):
    with open(path, "a") as f:
        f.write(msg + "\n")
    # print(msg)

def load_global(path=GLOBAL_INI):
    cfg = configparser.ConfigParser()
    cfg.read(path)
    return cfg["cosmology"]


def make_class(c, src=CLASS_INI):
    dst = Path(str(src) + "_run.ini")
    shutil.copy(src, dst)
    # print(c['Pk_ini_type'])
    if(c['Pk_ini_type']=='external_pk'):
        block = f"""
h = {c['h']}
omega_b = {float(c['Omega_b']) * float(c['h'])**2}
omega_cdm = {float(c['Omega_cdm']) * float(c['h'])**2}
T_cmb = {c['T_cmb']}
Pk_ini_type = {c['Pk_ini_type']}
command = {c['command']}
"""

    elif(c['Pk_ini_type']=='analytic_Pk'):
        block = f"""
h = {c['h']}
omega_b = {float(c['Omega_b']) * float(c['h'])**2}
omega_cdm = {float(c['Omega_cdm']) * float(c['h'])**2}
T_cmb = {c['T_cmb']}
Pk_ini_type = {c['Pk_ini_type']}
n_s = {float(c['n_s'])}
A_s = {float(c['A_s'])} """

    else:
        sys.exit('Check the inputs! and the class explanatory.ini file for reference')

    with open(dst, "a") as f:
        f.write(block)
    return dst

def make_music(c, src=MUSIC_CONF):
    dst = Path(str(src) + "_run.conf")
    shutil.copy(src, dst)
    block = f"""
[cosmology]
Omega_m = {c['Omega_m']}
Omega_b = {c['Omega_b']}
Omega_L = {1 - float(c['Omega_m'])}
H0 = {float(c['h']) * 100}
sigma_8 = {c['sigma_8']}
nspec = {c['n_s']}
"""
    with open(dst, "a") as f:
        f.write(block)
    return dst


def log_cosmology(c):
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log(f"\n========== RUN {stamp} ==========")
    log("cosmology used:")
    for k in c:
        log(f"    {k} = {c[k]}")

def set_key(path, pattern, replacement):
    """In-place substitution on a config file."""
    text = path.read_text()
    text = re.sub(pattern, replacement, text, flags=re.MULTILINE)
    path.write_text(text)

def reformat_tk(src, dst):
    """awk replacement: non-comment rows with >=7 fields -> 13 columns."""
    with open(src) as f, open(dst, "w") as out:
        for line in f:
            if line.lstrip().startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 7:
                continue
            c = [float(x) for x in fields]
            vals = [c[0], c[1], c[2], c[3], c[4], c[5],
                    c[6], c[6], c[6], c[6], c[1], c[2], 0.0]
            out.write(" ".join(f"{v:.12e}" for v in vals) + "\n") 


def run_tool(cmd, cwd, tag):
    log(f"--- {tag}: {' '.join(cmd)} (cwd={cwd}) ---")
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True) 
    with open(LOG_PATH, "a") as f:                                        
        if proc.stdout:
            f.write(proc.stdout)
        if proc.stderr:
            f.write("[stderr]\n" + proc.stderr)
    proc.check_returncode()
    return proc
 
 
def run_pipeline(class_ini, music_conf):
    """Loop over the z-ladder running CLASS then MUSIC. Takes the run-config paths."""
    SNAP_OUTDIR.mkdir(parents=True, exist_ok=True)
    class_ini_name = class_ini.name
    music_conf_name = music_conf.name
 
    for z in ZLADDER:
        log("===================================================================")
        log(f">>> Redshift z = {z}")
        log("===================================================================")
 
        # CLASS
        set_key(class_ini, r"^\s*z_pk\s*=.*", f"z_pk = {z}")
        run_tool(["./class", class_ini_name], cwd=CLASS_DIR, tag=f"CLASS z={z}")
        reformat_tk(CLASS_OUT / "reiotest.ini_run_tk.dat",
                    CLASS_OUT / "output_camb13.dat")
 
        # MUSIC
        set_key(music_conf, r"^\s*zstart\s*=.*", f"zstart\t\t= {z}")
        run_tool(["./MUSIC", music_conf_name], cwd=MUSIC_DIR, tag=f"MUSIC z={z}")
        (MUSIC_DIR / "ics_gadget.dat").rename(MUSIC_DIR / f"ics.gadget_{z}.dat")

def log_run(c, path="pipeline.log"):
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a") as f:
        f.write(f"\n[{stamp}] cosmology used:\n")
        for k in c:
            f.write(f"    {k} = {c[k]}\n")



def load_script_cfg(path=GLOBAL_INI):
    cfg = configparser.ConfigParser()
    cfg.read(path)
    return cfg["script"]


def snapshot_path(z):
    """Path of the MUSIC snapshot for redshift z (matches run_pipeline's naming)."""
    return MUSIC_DIR / f"ics.gadget_{z}.dat"


def run_script(zladder=ZLADDER, out_csv=PROJECT_DIR / "xe_history.dat"):
    
    c = load_global()
    s = load_script_cfg()

    zeta       = float(s["zeta"])
    log10_Mmin = float(s["log10_Mmin"])
    ngrid      = int(s["ngrid"])
    he_factor  = float(s["helium_factor"])
    he_factor_low = float(s['helium_factor_lowz'])

    
    sigma_8 = float(c["sigma_8"])
    ns      = float(c["n_s"])
    omega_b = float(c["Omega_b"])

    os.makedirs(SCRIPT_FILES, exist_ok=True)

    history = []  # (z, Q_HII, x_e)
    global add_flag
    add_flag=False

    for z in zladder:
        snap = snapshot_path(z)
        log(f"SCRIPT z={z}: {snap}")
        if not snap.exists():
            log(f"[skip] snapshot missing: {snap}")
            continue

        # scaledist = 1 since music gives output in Mpc**3 h-3  
        sim = script.default_simulation_data(
            str(snap), str(SCRIPT_FILES),
            sigma_8=sigma_8, ns=ns, omega_b=omega_b,
            scaledist=1, #note
        )

        # print(f"  z={sim.z:.3f}  box={sim.box:.3f}") #printing for verification


        
        mf    = script.matter_fields(sim, ngrid, str(SCRIPT_FILES),overwrite_files=True)
        ion   = script.ionization_map(mf)
        fcoll = mf.get_fcoll_for_Mmin(log10_Mmin)
        qi    = ion.get_qi(zeta * fcoll)
        QHII  = float(np.mean(qi * (1 + mf.densitycontr_arr))) #mass_avg

        if((z-zmin)<step/2 and QHII <=0.99):
            print(f'Reionisation does not end by {zmin} rejected!')
        
        if((z-zmax)<step/2):
            if QHII >= 1e-3:
                print(f'Reionisation does not start at {zmax}, rejected!')
            add_flag = (QHII != 0)


        if(QHII==0):
            print('Encountered 0 QHII, exiting...')
            break
            

        if(z>=3.0):
            x_e = QHII * he_factor # muntiply with 1.08 >3 
        if(z<3.0):
            x_e = QHII * he_factor_low #muntiply with 1.16 < 3
        
        log(f"    z = {sim.z}   Q_HII = {QHII}   x_e = {x_e}")
        history.append((float(sim.z), QHII, x_e))



def run_script_multi(z):
    c = load_global()
    s = load_script_cfg()
 
    zeta          = float(s["zeta"])
    log10_Mmin    = float(s["log10_Mmin"])
    ngrid         = int(s["ngrid"])
    he_factor     = float(s["helium_factor"])
    he_factor_low = float(s["helium_factor_lowz"])
 
    sigma_8 = float(c["sigma_8"])
    ns      = float(c["n_s"])
    omega_b = float(c["Omega_b"])
 
    snap = snapshot_path(z)
    if not snap.exists():
        log(f"[skip] snapshot missing: {snap}")
        return None
 
    
    out_z = SCRIPT_FILES / f"z{z}"
    os.makedirs(out_z, exist_ok=True)
 
    sim = script.default_simulation_data(str(snap), str(out_z),sigma_8=sigma_8, ns=ns, omega_b=omega_b,scaledist=1)                       # MUSIC coords are Mpc/h)
    mf    = script.matter_fields(sim, ngrid, str(out_z), overwrite_files=True)
    ion   = script.ionization_map(mf)
    fcoll = mf.get_fcoll_for_Mmin(log10_Mmin)
    qi    = ion.get_qi(zeta * fcoll)
    QHII  = float(np.mean(qi * (1 + mf.densitycontr_arr)))

    
    if QHII < 1e-6:
        log(f"    z = {sim.z}   Q_HII almost 0, skipping")
        return None

   
    if z >= 3.0:    
        x_e = QHII * he_factor
    else:
        x_e = QHII * he_factor_low
 
    log(f"    z = {sim.z}   Q_HII = {QHII}   x_e = {x_e}")
    return (float(sim.z), QHII, x_e)
 
 
def run_script_parallel(zladder=ZLADDER, out_csv=PROJECT_DIR / "xe_history.dat", nproc=None):                                
    os.makedirs(SCRIPT_FILES, exist_ok=True)
 
    
    if nproc is None:
        nproc = max(1, (os.cpu_count() or 2) - 1)
 
    
    with Pool(processes=nproc) as pool:
        results = pool.map(run_script_multi, list(zladder))
 
   
    history = [r for r in results if r is not None]
 
    
    hist_by_z = {round(zz, 4): (zz, q, xe) for zz, q, xe in history}
    z_lo = min(hist_by_z) if hist_by_z else None
    z_hi = max(hist_by_z) if hist_by_z else None

    if z_lo is None or hist_by_z[z_lo][1] <= 0.99:
        q = hist_by_z[z_lo][1] if z_lo is not None else float("nan")
        print(f"reionization incomplete: Q_HII({z_lo}) = {q} < 0.99 Rejecting!.")
        return None                       

    if z_hi is not None and hist_by_z[z_hi][1] >= 1e-3:
        print(f"highest z={z_hi} has Q_HII={hist_by_z[z_hi][1]} > 0 not fully neutral, Rejecting!")
        return None
 
    
    history.sort(key=lambda r: r[0])
    with open(out_csv, "w") as f:
        f.write("# z   Q_HII   x_e\n")
        for zz, q, xe in history:
            f.write(f"{zz} {q} {xe}\n")
    log(f"SCRIPT history written to {out_csv} ({len(history)} points)")
    return True

    


def make_reio_inter(history_file=PROJECT_DIR/"xe_history.dat", ini_file=CLASS_FIN):

    _s = load_script_cfg()
    he_factor = float(_s["helium_factor"])
    he_factor_low = float(_s["helium_factor_lowz"])


    START = "# --- BEGIN reio_inter ---"
    END   = "# --- END reio_inter ---"

    data = np.loadtxt(history_file, comments="#")
    z, xe = data[:, 0], data[:, 2]
    if(True):
        z = np.append(z, z[-1] + step)  
        xe = np.append(xe, 0.0)           

    
    order = np.argsort(z)
    z, xe = z[order], xe[order]

    z_vals  = [f"{v:g}" for v in z]
    xe_vals = [f"{v:.6f}" for v in xe]

    z_val_low = np.round(np.arange(0, zmin, step), 1)
    xe_vals_low = [(he_factor) if i > 3 else (he_factor_low) for i in z_val_low]

    z_vals_low_str  = [f"{v:g}" for v in z_val_low]
    xe_vals_low_str = [f"{v:.6f}" for v in xe_vals_low]

    z_full  = np.concatenate([z_val_low, z])
    xe_full = np.concatenate([xe_vals_low, xe])

    full_out = PROJECT_DIR/"xe_history_full.dat"
    with open(full_out, "w") as f:
        f.write("# z   x_e   (full)\n")
        for zz, xx in zip(z_full, xe_full):
            f.write(f"{zz:.6f} {xx:.6f}\n")
    log(f"full x_e curve written to {full_out} ({len(z_full)} points)")

    z_str  = ", ".join(z_vals_low_str + z_vals)    
    xe_str = ", ".join(xe_vals_low_str + xe_vals)



    block = (
        f"{START}\n"
        "reio_parametrization = reio_inter\n"
        f"reio_inter_num = {len(z_full)}\n"
        f"reio_inter_z = {z_str}\n"
        f"reio_inter_xe = {xe_str}\n"
        f"{END}\n"
    )

    text = Path(ini_file).read_text()
    # strip a previously written block, if present
    start = text.find(START)
    if start != -1:
        end = text.find(END, start)
        if end != -1:
            text = text[:start] + text[end + len(END):]
        else:
            text = text[:start]
    text = text.rstrip() + "\n\n" + block

    Path(ini_file).write_text(text)
    log(f"reio_inter block written to {ini_file}")
    # print(block)
    return block

TAU_RE = re.compile(r"optical depth\s*=\s*([0-9.]+)")

def run_class_final(ini_file=CLASS_DIR/"reiotest_1.ini"):
    ini_name = Path(ini_file).name
    proc = run_tool(["./class", ini_name], cwd=CLASS_DIR, tag="CLASS final (reio_inter)")

    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    m = TAU_RE.search(combined)
    if m:
        tau = float(m.group(1))
        print(f">>> tau_reio = {tau}")
        log(f">>> tau_reio = {tau:.6f}")
        return tau
    else:
        print(">>> tau not found in CLASS output")
        log(">>> tau not found in CLASS output")
        return None


def get_class_cl(history, ini_file=CLASS_FIN):
    _s = load_script_cfg()
    he_factor     = float(_s["helium_factor"])
    he_factor_low = float(_s["helium_factor_lowz"])

    START = "# --- BEGIN reio_inter ---"
    END   = "# --- END reio_inter ---"

    arr = np.asarray(history, dtype=float)
    z, xe = arr[:, 0], arr[:, 2]

   
    z  = np.append(z,  z[-1] + step)
    xe = np.append(xe, 0.0)

    order = np.argsort(z)
    z, xe = z[order], xe[order]

    z_vals  = [f"{v:g}"   for v in z]
    xe_vals = [f"{v:.6f}" for v in xe]

  
    z_val_low   = np.round(np.arange(0, zmin, step), 1)
    xe_vals_low = [(he_factor) if i > 3 else (he_factor_low) for i in z_val_low]
    z_vals_low_str  = [f"{v:g}"   for v in z_val_low]
    xe_vals_low_str = [f"{v:.6f}" for v in xe_vals_low]

    z_full = np.concatenate([z_val_low, z])  

    z_str  = ", ".join(z_vals_low_str + z_vals)
    xe_str = ", ".join(xe_vals_low_str + xe_vals)

    block = (
        f"{START}\n"
        "reio_parametrization = reio_inter\n"
        f"reio_inter_num = {len(z_full)}\n"
        f"reio_inter_z = {z_str}\n"
        f"reio_inter_xe = {xe_str}\n"
        f"{END}\n"
    )

    text = Path(ini_file).read_text()
    start = text.find(START)
    if start != -1:
        end = text.find(END, start)
        text = text[:start] + (text[end + len(END):] if end != -1 else "")
    text = text.rstrip() + "\n\n" + block
    Path(ini_file).write_text(text)

    ini_name = Path(ini_file).name
    run_tool(["./class", ini_name], cwd=CLASS_DIR, tag="CLASS cl")

    cl_file = CLASS_DIR / "output" / "reiotest_1_cl_lensed.dat"
    if not cl_file.exists():
        cl_file = CLASS_DIR / "output" / "reiotest_1_cl.dat"
    if not cl_file.exists():
        raise FileNotFoundError(f"no C_ell output in {CLASS_DIR/'output'}")

    data = np.loadtxt(cl_file,comments="#")
    ell   = data[:, 0]
    cl_tt = data[:, 1]
    cl_ee = data[:, 2]
    cl_bb = data[:, 3]
    cl_te = data[:, 4]
    return ell, cl_tt, cl_ee, cl_te, cl_bb













def _gen_one_z(args):
    """
    Generate the MUSIC snapshot for a single redshift z, using per-z
    args = (z, base_class_ini, base_music_conf)
    """
    z, base_class_ini, base_music_conf = args

   
    cls_ini  = CLASS_DIR / f"reiotest_z{z}.ini"
    mus_conf = MUSIC_DIR / f"reiotest_z{z}.conf"
    shutil.copy(base_class_ini,  cls_ini)
    shutil.copy(base_music_conf, mus_conf)

    # per-z unique output names 
    cls_root   = f"output/reio_z{z}_"          # CLASS writes reio_z{z}_tk.dat
    tk_file    = CLASS_DIR / f"{cls_root}tk.dat"
    camb13     = CLASS_OUT / f"camb13_z{z}.dat"
    mus_snap   = MUSIC_DIR / f"ics_z{z}.gadget"  

    set_key(cls_ini, r"^\s*z_pk\s*=.*",          f"z_pk = {z}")
    set_key(cls_ini, r"^\s*root\s*=.*",          f"root = {cls_root}")
    txt = cls_ini.read_text()
    if "overwrite_root" not in txt:
        cls_ini.write_text(txt.rstrip() + "\noverwrite_root = yes\n")

    subprocess.run(["./class", cls_ini.name], cwd=CLASS_DIR,
                   capture_output=True, text=True).check_returncode()

    reformat_tk(tk_file, camb13)

    set_key(mus_conf, r"^\s*zstart\s*=.*",        f"zstart\t\t= {z}")
    set_key(mus_conf, r"^\s*transfer_file\s*=.*", f"transfer_file\t= {camb13}")
    set_key(mus_conf, r"^\s*filename\s*=.*",      f"filename\t= {mus_snap.name}")

    subprocess.run(["./MUSIC", mus_conf.name], cwd=MUSIC_DIR,
                   capture_output=True, text=True).check_returncode()

    
    final = MUSIC_DIR / f"ics.gadget_{z}.dat"
    if mus_snap.exists():
        mus_snap.rename(final)

    # cleanup
    for f in (cls_ini, mus_conf):
        if f.exists():
            f.unlink()

    return str(final)


def run_pipeline_parallel(class_ini, music_conf, nproc=None):
    """Parallel CLASS+MUSIC generation across the z-ladder."""
    SNAP_OUTDIR.mkdir(parents=True, exist_ok=True)
    if nproc is None:
        nproc = max(1, (os.cpu_count() or 2) - 1)

    args = [(z, class_ini, music_conf) for z in ZLADDER]
    with Pool(processes=nproc) as pool:
        snaps = pool.map(_gen_one_z, args)
    return snaps


def history_one_z(args):

    z, zeta, log10_Mmin = args
 
    c = load_global()
    s = load_script_cfg()
 
    ngrid         = int(s["ngrid"])
    he_factor     = float(s["helium_factor"])
    he_factor_low = float(s["helium_factor_lowz"])
 
    sigma_8 = float(c["sigma_8"])
    ns      = float(c["n_s"])
    omega_b = float(c["Omega_b"])
 
    snap = snapshot_path(z)
    if not snap.exists():
        log(f"[skip] snapshot missing: {snap}")
        return None
 
    out_z = SCRIPT_FILES / f"z{z}"
    os.makedirs(out_z, exist_ok=True)
 
    sim = script.default_simulation_data(str(snap), str(out_z),sigma_8=sigma_8, ns=ns, omega_b=omega_b,scaledist=1)
    mf    = script.matter_fields(sim, ngrid, str(out_z), overwrite_files=True)
    ion   = script.ionization_map(mf)
    fcoll = mf.get_fcoll_for_Mmin(log10_Mmin)
    qi    = ion.get_qi(zeta * fcoll)
    QHII  = float(np.mean(qi * (1 + mf.densitycontr_arr)))
 
    if QHII < 1e-6:
        return None
 
    x_e = QHII * (he_factor if z >= 3.0 else he_factor_low)
    return (float(sim.z), QHII, x_e)
 



def reionization_history(zeta, log10_Mmin, zladder=ZLADDER,nproc=None, out_csv=None):

    os.makedirs(SCRIPT_FILES, exist_ok=True)
    if nproc is None:
        nproc = max(1, (os.cpu_count() or 2) - 1)
 
    args = [(z, (zeta), (log10_Mmin)) for z in zladder]
    with Pool(processes=nproc) as pool:
        results = pool.map(history_one_z, args)
 
    history = sorted((r for r in results if r is not None), key=lambda r: r[0])
 
    # if out_csv is not None:
    #     with open(out_csv, "w") as f:
    #         f.write("# z   Q_HII   x_e\n")
    #         for zz, q, xe in history:
    #             f.write(f"{zz:} {q:} {xe:}\n")
    #     log(f"history (zeta={zeta}, log10_Mmin={log10_Mmin}) "
    #         f"written to {out_csv} ({len(history)} points)")
 
    return history
 


def main():
    c = load_global()
    log_cosmology(c)
 
    class_ini = make_class(c)
    music_conf = make_music(c)
 
    try:
        run_pipeline(class_ini, music_conf)
    finally:
        for f in (class_ini, music_conf):
            if f.exists():
                f.unlink()
        log("run-config files discarded.")

    if run_script_parallel() is None:
        print("reionization incomplete; aborting this run.")
        return
    make_reio_inter()
    run_class_final()




 
 
if __name__ == "__main__":
    main()
