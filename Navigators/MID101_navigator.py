'''
@author: Jana Zhang, @date: 28.09.2021
This script generates the image-based self-navigator for the volunteer MID101.
The filepath to the measurement data (.h5 file) needs to be assigned in the beginning.
The script uses PTBPyRecon (more info in main text file).
'''

#imports
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys
# assign filepath to PtbPyRecon folder
sys.path.append('/PtbPyRecon_develop/')
import PTBRecon

# assign filepath to the measurement data (.h5 file)
filepath_data = "/data/zhang18/MRI_Data/Data_PTBMR04_010/meas_MID101_T1resp_13x13x8_FID74799.h5"

rec_im = []
#
# Total number of radial lines (to speed up reconstruction)
nr_tot = 3191
#
# Number of radial lines in each dynamic (i.e. TI image)
nr_dyn = 20

# Read in raw data
mr = PTBRecon.MRScan(filepath_data)
mr.ReadHdr(mr)
mr.ReadData(mr)
mr.SortData(mr)

# Coil compression
mr.Pars.Recon.CoilCompNum = 4
mr.CoilComp(mr)

idx = np.argsort(mr.Pars.Encoding.Idx.TimeMr)
mr.Data.K = mr.Data.K[:, idx, ...]
mr.Pars.Encoding.Idx.TimeMr = mr.Pars.Encoding.Idx.TimeMr[idx]
mr.Pars.Encoding.Idx.Ki[1, :] = np.linspace(1, len(idx), len(idx))

mr.Pars.Encoding.IDims = [256, 256, 1]

# Calculate trajectory
mr.Pars.Recon.KTrajType = 'radial'
mr.CalcTraj(mr)

mr.Pars.Recon.KDcfType = 'zwart'
mr.CalcDcf(mr)

# Calculate coil maps
mr.Pars.Recon.CsmMode = 'inati'
mr.CalcCsm(mr)

# Reconstruction of mask
mr.Pars.ItRecon.MaxIt = 3
mr.Pars.Recon.ReconType = 'cg'
mr.ReconData(mr)

# Calculate mask
idat_mean = np.abs(np.squeeze(mr.Data.I))
idat_mean = idat_mean / idat_mean.max()
mask = np.zeros(idat_mean.shape)
mask[idat_mean > 0.08] = 1

mask = sp.ndimage.morphology.binary_opening(mask.astype(np.int), np.ones((1, 1)).astype(np.int))
mask = sp.ndimage.morphology.binary_closing(mask.astype(np.int), np.ones((9, 9)).astype(np.int))
mask = sp.ndimage.morphology.binary_fill_holes(mask.astype(np.int))
mr.Pars.Fit.Mask = mask[:, :, np.newaxis]

# Calculate parameters for navigator
diff_time_mr = np.diff(mr.Pars.Encoding.Idx.TimeMr)
idx_inv = np.where(diff_time_mr >= 20)
num_inv = len(idx_inv[0]) + 1
Tinv_calc = mr.Pars.Encoding.Idx.TimeMr[idx_inv[0]]

tau = np.round(np.mean(diff_time_mr[idx_inv[0]]))
scan_time = mr.Pars.Encoding.Idx.TimeMr[-1] - mr.Pars.Encoding.Idx.TimeMr[0] + tau
Ti_calc = scan_time / num_inv

# Add inversion prior to first data acquisition
Tinv_calc = np.concatenate(([mr.Pars.Encoding.Idx.TimeMr[0], ], Tinv_calc), axis=0)
idx_acq = np.where(diff_time_mr < 20)
Tr_calc = np.mean(diff_time_mr[idx_acq])

# Select first part of acquisition to speed up reconstruction
mr.Pars.Recon.SelectType = ['ky', ]
mr.Pars.Recon.SelectRange = [np.linspace(0, nr_tot - 1, nr_tot), ]
mr.Pars.Recon.SelectKeepSize = [0, ]
mr.Pars.Recon.SelectKeepTempOrd = [1, ]
mr.SelectData(mr)

# Split dynamics
mr.Pars.Recon.SplitDynSlWnd = 0
mr.Pars.Recon.SplitDynNum = int(np.floor(len(mr.Pars.Encoding.Idx.TimeMr) / nr_dyn))
mr.SplitDyn(mr)

# Calculate timing
acq_time = [np.average(val) for val in mr.Pars.Recon.SplitDynVal]

ti = (np.asarray(acq_time) - Tinv_calc[0]) / 1000
ti = ti.astype(np.float64)
tinv = (Tinv_calc - Tinv_calc[0]) / 1000
tinv = tinv.astype(np.float64)
tr = float(Tr_calc / 1000)
tau = float(tau / 1000)

# # #Remove any frames which covers an inversion pulse
rm_idx = []
for idx, val in enumerate(mr.Pars.Recon.SplitDynVal):
    for cinv in tinv * 1000:
        if len(np.where((val - Tinv_calc[0]) <= cinv)[0]) > 0 and len(np.where((val - Tinv_calc[0]) > cinv)[0]) > 0:
            rm_idx.append(idx)
            break

phase_idx = []
if len(rm_idx) > 0:
    phase_idx = np.arange(0, len(acq_time))
    phase_idx = np.delete(phase_idx, rm_idx)
    mr.Pars.Recon.SelectType = ['phase', ]
    mr.Pars.Recon.SelectRange = [phase_idx, ]
    mr.Pars.Recon.SelectKeepSize = [0, ]
    mr.Pars.Recon.SelectKeepTempOrd = [0, ]
    mr.SelectData(mr)
    ti = np.asarray(ti)[phase_idx]

mr.Pars.Recon.KDcfType = 'zwart'
mr.CalcDcf(mr)


'''
Image reconstruction followed by calculation of navigator for respiratory motion
'''
# Iterative SENSE
mr.Pars.ItRecon.MaxIt = 5
mr.Pars.Recon.ReconType = 'cg'
mr.ReconData(mr)
mr.CombineCoils(mr)
idat_ti = np.squeeze(mr.Data.I)

# parameters for calculating zero crossings of liver
M0 = mr.Pars.Scan.B0
fa = mr.Pars.Scan.Fa * np.pi / 180
t1_liver = 0.783  # T1 time of liver in s
t1_eff = 1 / ((1 / t1_liver) - (1 / (Tr_calc/1000)) * np.log(np.cos(fa)))
M0_eff = (M0 * t1_eff) / t1_liver
# calculate zero crossings and indices
t_zero = []
t_zero_id_K = []
t_k = (mr.Pars.Scan.Ta*60)/nr_tot #time between lines in k-space
idx_inv_K = np.array(idx_inv[0])

for inv_index in range(num_inv):
    if (inv_index == 0):
        t_zero_curr = (-np.log(M0_eff / (M0 + M0_eff))) * t1_eff
        t_zero.append(t_zero_curr)
        id_zerocr_K_curr = t_zero_curr/t_k #lines in k-space between 0 and current zero crossing
        t_zero_id_K.append(int(id_zerocr_K_curr))
    else:
        curr_inv = tinv[inv_index]
        t_zero_curr = (-np.log(1 / 2)) * t1_eff + curr_inv
        t_zero.append(t_zero_curr)
        id_zerocr_K_curr = t_zero_curr / t_k  #lines in k-space between 0 and current zero crossing
        t_zero_id_K.append(int(id_zerocr_K_curr))

# set parameters for navigator
mr.Pars.Nav.RNav_idat_ti = idat_ti
mr.Pars.Nav.RNav_ti = ti
mr.Pars.Nav.RNav_t_zero = t_zero
# user defined
mr.Pars.Nav.RNavType= 'im_sagittal'
mr.Pars.Nav.x_line = 75
mr.Pars.Nav.y_min = 60
mr.Pars.Nav.y_max = 90
mr.Pars.Nav.smooth_fac = 50

mr.CalcRNav(mr)
r_nav_sig = mr.Pars.Nav.RNavSig
r_nav_sig_mm = (r_nav_sig * (320/256)) - np.mean(r_nav_sig*(320/256))
nav_sig_time = mr.Pars.Nav.RNavSigT
nav_sig_points = mr.Pars.Nav.RNavSigPoints
nav_sig_points_mm = (nav_sig_points * (320/256)) - np.mean(nav_sig_points*(320/256))
time_mr_corr = ((mr.Pars.Encoding.Idx.TimeMr - mr.Pars.Encoding.Idx.TimeMr[0])/1000)
plt.figure()
plt.plot(nav_sig_time, nav_sig_points_mm, 'm.', time_mr_corr, r_nav_sig_mm, '--k')
plt.plot(nav_sig_time,nav_sig_points_mm,'m.')
plt.xticks(fontsize='large')
plt.yticks(fontsize='large')
plt.xlabel('time in s', fontsize='large')
plt.ylabel('displacement of the lungs in mm', fontsize='large')
plt.title('image-based self-navigator MID101')