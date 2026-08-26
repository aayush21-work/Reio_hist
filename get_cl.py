import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import run_pipeline as P
from getdist import loadMCSamples

CHAIN = "chains/classy_planck_TTEE"
BURN_IN =0.4
TOTAL_SAMPLES=200

tt = np.loadtxt("COM_PowerSpect_CMB-TT-binned_R3.01.txt",comments='#') ##TT planck obs
ee = np.loadtxt("Planck_EE_unbinned.txt",comments='#') ## EE planck obs
lt, Dlt, dlot, dhit = tt[:,0], tt[:,1], tt[:,2], tt[:,3]
le, Dle, dloe, dhie = ee[:,0], ee[:,1], ee[:,2], ee[:,3]








def main():
    # data = np.loadtxt(CHAIN, comments="#")
    samples=loadMCSamples(CHAIN,settings={"ignore_rows":0.4})
    data=samples.samples

    zeta  = data[:, 0]
    lmmin = data[:, 1]
    chi2  = data[:, 4]

    order = np.random.randint(0,len(data),TOTAL_SAMPLES)

    z_sel = zeta[order]
    m_sel = lmmin[order]
    c_sel = chi2[order]

    # print(z_sel,m_sel)   #check 1
    

    cmap = plt.cm.jet
    norm = mpl.colors.Normalize(vmin=c_sel.min(), vmax=c_sel.max())

    fig, (ax_tt, ax_ee) = plt.subplots(1, 2, figsize=(10, 6))

    draw_order = np.argsort(c_sel)[::-1]
  

    nfail = 0
    for k in draw_order:
        zeta_k, mmin_k, chi2_k = z_sel[k], m_sel[k], c_sel[k]

        history = P.reionization_history(zeta_k, mmin_k)


        if len(history) == 0 or history[0][1] < 0.99:
            nfail += 1
            continue

        try:
            ell, cl_tt, cl_ee, cl_te, cl_bb = P.get_class_cl(history)
        except Exception as e:
            print(f"  CLASS failed zeta={zeta_k:} Mmin={mmin_k:}")
            nfail += 1
            continue

        # print(cl_ee)  #check 2


        color = cmap(norm(chi2_k))
        ax_tt.plot(ell, cl_tt, lw=0.5, alpha=0.6, color=color)
        ax_ee.plot(ell, cl_ee, lw=0.5, alpha=0.6, color=color)

    print(f"done; {nfail} samples skipped (incomplete or CLASS error)")

    ax_tt.errorbar(lt, Dlt, yerr=[dlot, dhit], fmt="o", ms=3, color="black",capsize=2, label="Planck EE")
    ax_tt.set_xlabel(r"$\ell$")
    ax_tt.set_ylabel(r"$\mathcal{D}_\ell^{TT}\ [\mu K^2]$")
    

    ax_ee.errorbar(le, Dle, yerr=[dloe, dhie], fmt="o", ms=2.5, color="black",capsize=1.5, label="Planck EE")
    ax_ee.set_xlabel(r"$\ell$")
    ax_ee.set_ylabel(r"$\mathcal{D}_\ell^{EE}\ [\mu K^2]$")
    ax_ee.set_xlim(2, 30)
    ax_ee.set_xscale("log")
    ax_ee.set_yscale("log")
    

    
    fig.subplots_adjust(right=0.88)                       
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])    
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),cax=cbar_ax, label=r"$\chi^2$")

    plt.savefig(f"cl_ee_tt_chi2.png", dpi=600)
    plt.show()


if __name__ == "__main__":
    main()
