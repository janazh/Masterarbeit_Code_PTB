'''
@author: Jana Zhang, @date: 28.09.2021
This script generates the breathhold T1 map for the volunteer MID236.
The measurement data for the breathhold is MID237 (the number is the number of the original data plus 1).
The filepath to the measurement data (.h5 file) needs to be assigned in the beginning.
The script uses PTBPyRecon (more info in main text file).
'''

import numpy as np
import scipy as sp
import TestScripts.helper as helper
import matplotlib.pyplot as plt
import sys
sys.path.append('../../../')
import PTBRecon
import PTBRecon.Subpackages.parametric as par

def r1_to_t1_rgb(fit_result):
    map_t1 = np.squeeze(fit_result)
    map_t1_r1_2_t1 = 1000. / map_t1[:, :, 2]
    map_t1_r1_2_t1[map_t1[:, :, 2] == 0] = 0
    map_t1_rgb = helper.im2rgb([map_t1[:, :, 0], map_t1[:, :, 1], map_t1_r1_2_t1], 'magma',
                               [map_t1[:, :, 0].max() * 0.8, 6, 2000])
    return (map_t1_rgb)


praw = "/data/zhang18/MRI_Data/Data_PTBMR04_012/meas_MID237_T1resp_13x13x8_FID74935.h5"
rec_im = []

# Total number of radial lines (to speed up reconstruction)
nr_tot = 3191

# Number of radial lines in each dynamic (i.e. TI image)
nr_dyn = 40

# Read in raw data
mr = PTBRecon.MRScan(praw)
mr.ReadHdr(mr)
mr.ReadData(mr)
mr.SortData(mr)

# Coil compression
mr.Pars.Recon.CoilCompNum = 4
mr.CoilComp(mr)

# Angular index is not correctly set in sequence and needs to be defined here
num_ang = mr.Pars.Encoding.Idx.Ki.shape[1]
mr.Pars.Encoding.Idx.Ki[1, :] = np.linspace(1, num_ang, num_ang)

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
mask[idat_mean > 0] = 1

mask = sp.ndimage.morphology.binary_opening(mask.astype(np.int), np.ones((1, 1)).astype(np.int))
mask = sp.ndimage.morphology.binary_closing(mask.astype(np.int), np.ones((9, 9)).astype(np.int))
mask = sp.ndimage.morphology.binary_fill_holes(mask.astype(np.int))
mr.Pars.Fit.Mask = mask[:, :, np.newaxis]

# Calculate parameters for fit
diff_time_mr = np.diff(mr.Pars.Encoding.Idx.TimeMr)
idx_inv = np.where(diff_time_mr >= 20)
num_inv = len(idx_inv[0]) + 1
Tinv_calc = mr.Pars.Encoding.Idx.TimeMr[idx_inv[0]]

tau = np.round(np.mean(diff_time_mr[idx_inv[0]]))
scan_time = mr.Pars.Encoding.Idx.TimeMr[-1] - mr.Pars.Encoding.Idx.TimeMr[0] + tau
Ti_calc = scan_time / num_inv

# Add inversion prior to first data acquisition
Tinv_calc = np.concatenate(([mr.Pars.Encoding.Idx.TimeMr[0], ], Tinv_calc), axis=0) - tau

idx_acq = np.where(diff_time_mr < 20)
Tr_calc = np.mean(diff_time_mr[idx_acq])

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

# Remove any frames which covers an inversion pulse
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

mr.Pars.Recon.KDcfType = 'voronoi'
mr.CalcDcf(mr)

# Set lower and upper bounds and starting values (rho, alpha, R1)
mr.Pars.Fit.LowBounds = [0.0, 1.0, 0.0]
mr.Pars.Fit.UpBounds = [np.inf, 5.0, np.inf]
mr.Pars.Fit.StartingValues = [1.0, 3.0, 1.0]
mr.Pars.Fit.VaryPars = [True, True, True]

# Set fit parameters
mr.Pars.Fit.FitPerformance = 'multi'
mr.Pars.Fit.Dim = 6
mr.Pars.Fit.MinPars = {'xtol': 1e-1}
mr.Pars.Fit.SigModel = par.models.model_cont_acq_3par_r1
mr.Pars.Fit.DiffFitModel = par.fit.diff_abs_model_data
mr.Pars.Fit.SigModelPars = (ti, tinv, tr, tau, 0)

'''
A) Image reconstruction followed by data fit
'''
# Iterative SENSE
mr.Pars.ItRecon.MaxIt = 4
mr.Pars.Recon.ReconType = 'cg'
mr.ReconData(mr)
mr.CombineCoils(mr)
idat_ti = np.squeeze(mr.Data.I)

# T1 fit of continuous radial acq - multiple CPU
mr.Pars.Fit.FitPerformance = 'multi'
mr.FitImages(mr)
idat_qpar = np.squeeze(mr.Data.Fit)
rec_im.append(np.concatenate(r1_to_t1_rgb(mr.Data.Fit), axis=0))
fig1 = plt.figure()
im1 = plt.imshow((1/abs(idat_qpar[:,:,2]))*1000, cmap = 'magma', vmin = 0, vmax = 2500)
fig1.colorbar(im1)