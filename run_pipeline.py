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

PROJECT_DIR     = Path("/home/aayush/PROJECT_NCRA")
 
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
    print(msg)

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
    """In-place regex substitution on a config file (sed -i equivalent)."""
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
    """Run a subprocess, capture stdout+stderr, append it all to the log."""
    log(f"--- {tag}: {' '.join(cmd)} (cwd={cwd}) ---")
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.stdout:
        log(proc.stdout.rstrip())
    if proc.stderr:
        log("[stderr]\n" + proc.stderr.rstrip())
    log(f"--- {tag} exit code: {proc.returncode} ---")
    proc.check_returncode()  # raise if nonzero
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
        reformat_tk(CLASS_OUT / "reiotest_tk.dat",
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

        if(z==zmin and QHII <=0.99):
            print(f'{zmin} is not small enough, try to reduce it and rerun')
            sys.exit('Exiting...')

        
        if(z == zmax):
            if QHII >= 1e-3:
                print(f'{zmax} is not large enough, try increasing it and rerun')
                sys.exit('Exiting...')
            add_flag = (QHII != 0)
            

        if(z>=3.0):
            x_e = QHII * he_factor # muntiply with 1.08 >3 
        if(z<3.0):
            x_e = QHII * he_factor_low #muntiply with 1.16 < 3
        
        log(f"    z = {sim.z:.3f}   Q_HII = {QHII:.4f}   x_e = {x_e:.4f}")
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
 
    sim = script.default_simulation_data(
        str(snap), str(out_z),
        sigma_8=sigma_8, ns=ns, omega_b=omega_b,
        scaledist=1,                       # MUSIC coords are Mpc/h
    )
    mf    = script.matter_fields(sim, ngrid, str(out_z), overwrite_files=True)
    ion   = script.ionization_map(mf)
    fcoll = mf.get_fcoll_for_Mmin(log10_Mmin)
    qi    = ion.get_qi(zeta * fcoll)
    QHII  = float(np.mean(qi * (1 + mf.densitycontr_arr)))
 
   
    if z >= 3.0:
        x_e = QHII * he_factor
    else:
        x_e = QHII * he_factor_low
 
    log(f"    z = {sim.z:.3f}   Q_HII = {QHII:.4f}   x_e = {x_e:.4f}")
    return (float(sim.z), QHII, x_e)
 
 
def run_script_parallel(zladder=ZLADDER, out_csv=PROJECT_DIR / "xe_history.dat",
               nproc=None):                                
    os.makedirs(SCRIPT_FILES, exist_ok=True)
 
    
    if nproc is None:
        nproc = max(1, (os.cpu_count() or 2) - 1)
 
    
    with Pool(processes=nproc) as pool:
        results = pool.map(run_script_multi, list(zladder))
 
   
    history = [r for r in results if r is not None]
 
    
    hist_by_z = {round(zz, 4): (zz, q, xe) for zz, q, xe in history}
    z_lo = min(hist_by_z) if hist_by_z else None
    z_hi = max(hist_by_z) if hist_by_z else None
    if z_lo is not None and hist_by_z[z_lo][1] <= 0.99:
        log(f"[warn] lowest z={z_lo} has Q_HII={hist_by_z[z_lo][1]:.3f} < ~1; "
            f"reionization may not complete -- consider lowering zmin.")
    if z_hi is not None and hist_by_z[z_hi][1] >= 1e-3:
        log(f"[warn] highest z={z_hi} has Q_HII={hist_by_z[z_hi][1]:.3e} > 0; "
            f"not fully neutral -- consider raising zmax.")
 
    # write z x_e table (sorted by z) for CLASS reio_inter  
    history.sort(key=lambda r: r[0])
    with open(out_csv, "w") as f:
        f.write("# z   Q_HII   x_e\n")
        for zz, q, xe in history:
            f.write(f"{zz:.6f} {q:.6f} {xe:.6f}\n")
    log(f"SCRIPT history written to {out_csv} ({len(history)} points)")
    return history

    # write z x_e table (sorted by z) for CLASS reio_inter
    history.sort(key=lambda r: r[0])
    with open(out_csv, "w") as f:
        f.write("# z   Q_HII   x_e\n")
        for zz, q, xe in history:
            f.write(f"{zz:.6f} {q:.6f} {xe:.6f}\n")
    log(f"SCRIPT history written to {out_csv} ({len(history)} points)")
    return history


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
    print(block)
    return block


def run_class_final(ini_file=CLASS_DIR/"reiotest_1.ini"):
    ini_name = Path(ini_file).name
    proc = run_tool(["./class", ini_name], cwd=CLASS_DIR, tag="CLASS final (reio_inter)")

    # print tau_reio from stdout if CLASS printed it
    for line in proc.stdout.splitlines():
        if "tau" in line.lower() and "reio" in line.lower():
            log(f">>> {line.strip()}")

    return proc


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

    run_script_parallel() 
    make_reio_inter()
    run_class_final()




 
 
if __name__ == "__main__":
    main()