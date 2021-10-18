'''
@author: Jana Zhang, @date: 28.09.2021
In this script, the image-based self-navigator is compared with the head-feet radar signal for volunteer MID182.
The filepaths to the measurement data and the transversal radar shifts need to be assigned in the beginning.
A figure is created comparing both signals, the variable corr_fac gives the correlation coefficient between both signals.
'''

#imports
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.signal as si
import sys
# assign filepath to PtbPyRecon folder
sys.path.append('/PtbPyRecon_develop/')
import PTBRecon

# assign filepath to the measurement data (.h5 file)
filepath_data = "/data/zhang18/MRI_Data/Data_PTBMR04_011/meas_MID182_T1resp_13x13x8_FID74880.h5"
# assign filepath to the transversal radar shifts (.txt file)
filepath_radar = "/data/zhang18/MRI_Data/Data_PTBMR04_011/Radar_trashift/trashift_182.txt"

rec_im = []
# Total number of radial lines (to speed up reconstruction)
nr_tot = 3191
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

#Remove any frames which covers an inversion pulse
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
mr.Pars.Nav.x_line = 60
mr.Pars.Nav.y_min = 30
mr.Pars.Nav.y_max = 120
mr.Pars.Nav.smooth_fac = 100

mr.CalcRNav(mr)
r_nav_sig = mr.Pars.Nav.RNavSig
nav_sig_time = ((mr.Pars.Encoding.Idx.TimeMr - mr.Pars.Encoding.Idx.TimeMr[0])/1000)
rnav_sig_noOffset = r_nav_sig - np.mean(r_nav_sig)

# read in radar signal
radar_shift_f = open(filepath_radar, 'r')
radar_shifts = radar_shift_f.readlines()

radar_shifts = [float(num) for num in radar_shifts] # radar shifts in mm
radar_shifts = radar_shifts[:-1]
radar_shifts_voxel = np.array(radar_shifts) * 320/300* 256/320 #radar shifts in voxel (radar shift in mm were calculated with FOV = 300, but FOV is actually 320)
radar_shifts_voxel_noOffset = radar_shifts_voxel - np.mean(radar_shifts_voxel)

'''eliminate time shift'''
n = len(rnav_sig_noOffset)
# calculate cross correlation
nav_radar_corr = si.correlate(rnav_sig_noOffset, radar_shifts_voxel_noOffset)
# delay of second to first signal is the index of maximum of cross correlation minus the length of the signal
delay = np.argmax(nav_radar_corr) - len(rnav_sig_noOffset)
# new indices for radar signal without delay
radar_idx_noDelay = np.linspace(delay,delay+len(rnav_sig_noOffset),num= len(rnav_sig_noOffset))

# plot with time axis, starting from zero going to end of radar signal
if delay < 0:
    rnav_time_new = nav_sig_time[0:(delay+len(nav_sig_time))]
    # plot in mm
    radar_shifts_new = radar_shifts_voxel_noOffset[-delay:len(radar_shifts_voxel_noOffset)]*(320/256)
    rnav_sig_new = rnav_sig_noOffset[0:(len(r_nav_sig)+delay)]*(320/256)
    fig,axs = plt.subplots()
    axs.plot(rnav_time_new, radar_shifts_new, color = 'orange',label='shifted radar signal')
    axs.plot(rnav_time_new, rnav_sig_new, 'k--', label='image-based navigator')
    plt.xlabel('time in s', fontsize='large')
    plt.ylabel('displacement of the lungs in mm', fontsize='large')
    handles, labels = axs.get_legend_handles_labels()
    lgd = axs.legend(handles, labels,
                     fontsize='large',
                     loc='upper left',
                     ncol=3,
                     scatterpoints=1)
    lgd.legendHandles[0]._sizes = [80]
    lgd.legendHandles[1]._sizes = [80]
    lgd.legendHandles[2]._sizes = [80]
    fig.show()

if delay > 0:
    rnav_time_new = nav_sig_time[delay:-1]
    # plot in mm
    radar_shifts_new = radar_shifts_voxel_noOffset[0:(len(radar_shifts_voxel_noOffset)-delay-1)]*(320/256)
    rnav_sig_new = rnav_sig_noOffset[delay:-1]*(320/256)
    fig,axs = plt.subplots()
    axs.plot(rnav_time_new, radar_shifts_new, color = 'orange',label='shifted radar signal')
    axs.plot(rnav_time_new, rnav_sig_new, 'k--', label='image-based navigator')
    plt.xlabel('time in s', fontsize='large')
    plt.ylabel('displacement of the lungs in mm', fontsize='large')
    handles, labels = axs.get_legend_handles_labels()
    lgd = axs.legend(handles, labels,
                     fontsize='large',
                     loc='upper left',
                     ncol=3,
                     scatterpoints=1)
    lgd.legendHandles[0]._sizes = [80]
    lgd.legendHandles[1]._sizes = [80]
    lgd.legendHandles[2]._sizes = [80]
    fig.show()

# get correlation factor
rnav_sig_new_adapt = (rnav_sig_new - np.mean(rnav_sig_new)) / (np.std(rnav_sig_new)*len(rnav_sig_new))
radar_shifts_new_adapt = (radar_shifts_new - np.mean(radar_shifts_new)) / (np.std(radar_shifts_new))
corr_fac = np.correlate(rnav_sig_new_adapt, radar_shifts_new_adapt)