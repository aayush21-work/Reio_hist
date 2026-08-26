import script
import sys
import numpy as np
z = 5.0
output_dir='./output_pipeline'
snap_file='./output_pipeline/ics.gadget_5.0.dat'
print("Init default_simulation_data", flush=True)
default_simulation_data = script.default_simulation_data(snap_file, output_dir, sigma_8=0.811, ns=0.9649, omega_b=0.049, scaledist=1.e-3)
print("Init matter_fields", flush=True)
matter_fields = script.matter_fields(default_simulation_data, 64, output_dir, overwrite_files=False)
print("matter_fields init done", flush=True)
ionization_map = script.ionization_map(matter_fields)
print("ionization_map init done", flush=True)

fcoll_arr = matter_fields.get_fcoll_for_Mmin(9.0)
print("fcoll_arr done", flush=True)
qi_arr = ionization_map.get_qi(15 * fcoll_arr)
print("qi_arr done", flush=True)

qi_mean = np.mean(qi_arr * (1 + matter_fields.densitycontr_arr))
print("Calculated qi_mean", flush=True)

Delta_HI_arr = (1 - qi_arr) * (1 + matter_fields.densitycontr_arr)
Delta_HI_rsd_arr = ionization_map.add_rsd_box(qi_arr)

matter_fields.initialize_powspec()
k_edges, k_bins = matter_fields.set_k_edges(nbins=15, log_bins=True)

powspec_21cm_rsd_binned, kount = ionization_map.get_binned_powspec(Delta_HI_rsd_arr, k_edges, units='mK')
powspec_21cm_binned, kount = ionization_map.get_binned_powspec(Delta_HI_arr, k_edges, units='mK')

out_file = f"{output_dir}/power_spectrum_z{z}.txt"
mask = kount > 0
res = np.column_stack((k_bins[mask], powspec_21cm_binned[mask], powspec_21cm_rsd_binned[mask]))
np.savetxt(out_file, res, header="k[h/Mpc] P(k)_no_RSD[mK^2] P(k)_RSD[mK^2]")
print(f"Power spectrum saved to {out_file}", flush=True)
