'''
@author: Jana Zhang, @date: 30.09.2021
Master thesis section "Quantitative Evaluation of Motion-Corrected T1 Maps" Fig. 4.4.16
This script takes the measurement data of one volunteer and plots the averaged T1 value and the corresponding standard
deviation (over the region) for the scan time 16, 8, 6, 4, 3 and 2 seconds. There is a plot for each region of interest.
The p values are calculated between the 16 seconds scan and all other scan times respectively. P values smaller than 0.05
are marked with a bracket with a *.
As input, the folderpath to the folder containing the motion-corrected T1 maps created with different scan times (16,8,6,4,3,2 seconds)
is given. The filepath to the 16 seconds T1 map is given. The filepath to the ROI masks of the volunteer is given, these
masks are then used for all scan times.

For the results in the master thesis, the measurement data MID101 was used for the evaluation of the scan time.
'''

import pandas as pd
import numpy as np
import numpy.ma as ma
import scipy.stats as stats
import matplotlib.pyplot as plt
import pickle


folderpath_scans = '/data/zhang18/evalScanTime/Data_16sEstimation/'
filepath_16s_refScan =  "/data/zhang18/evalScanTime/Data_16sEstimation/16_idat_qpar_moCo.npy"
# filepath to moCo mask for MID101, will be used for all scans
filepath_masks = "/data/zhang18/eval_T1maps_data/roi_masks_new/MID101_masks_fb.npy"

filenumber_list = [16,8,6,4,3,2]

df_pValues = pd.DataFrame()
df_meanValues = pd.DataFrame()
df_stdValues = pd.DataFrame()

idatqpar_16s = np.load(filepath_16s_refScan)
t1map_16s = 1/abs(idatqpar_16s[:,:,2])*1000
masks = np.load(filepath_masks)

for i_iter,i_file in enumerate(filenumber_list):
    # get T1 maps
    curr_idatqpar_scantime = np.load(folderpath_scans + str(i_file)+'_idat_qpar_moCo.npy')
    curr_t1map_scantime = 1/abs(curr_idatqpar_scantime[:,:,2])*1000

    # get average T1 values for each ROI
    for i_mask in range(np.shape(masks)[0]):
        curr_mask = masks[i_mask,:,:]
        # get masked T1 maps
        curr_t1map_scantime_roi = ma.masked_array(curr_t1map_scantime, np.logical_not(curr_mask))
        t1map_16s_roi = ma.masked_array(t1map_16s, np.logical_not(curr_mask))

        # put all nonzero values into array
        curr_t1map_scantime_roi_arr = curr_t1map_scantime_roi[np.nonzero(curr_t1map_scantime_roi)]
        t1map_16s_roi_arr = t1map_16s_roi[np.nonzero(t1map_16s_roi)]

        # calculate mean value and standard deviation of T1 values in ROI
        curr_mean = curr_t1map_scantime_roi_arr.mean()
        curr_std = curr_t1map_scantime_roi_arr.std()
        # store mean value and std into data frames
        df_meanValues.loc[i_mask,i_iter] = curr_mean
        df_stdValues.loc[i_mask,i_iter] = curr_std
        #calculate pvalue
        # check if data is normally distributed
        w_shapiro_16s, p_shapiro_16s = stats.shapiro(t1map_16s_roi)
        w_shapiro_currTime, p_shapiro_currTime = stats.shapiro(curr_t1map_scantime_roi_arr)
        if(p_shapiro_16s > 0.05 or p_shapiro_currTime > 0.05):
            w_16sCurrTime, p_16sCurrTime = stats.ttest_rel(curr_t1map_scantime_roi_arr,t1map_16s_roi_arr)
        else:
            w_16sCurrTime, p_16sCurrTime = stats.wilcoxon(curr_t1map_scantime_roi_arr, t1map_16s_roi_arr)

        #store pvalue into data frame
        df_pValues.loc[i_mask,i_iter] = p_16sCurrTime

df_meanValues.columns = ['mean(16s)','mean(8s)','mean(6s)','mean(4s)','mean(3s)','mean(2s)']
df_stdValues.columns = ['std(16s)','std(8s)','std(6s)','std(4s)','std(3s)','std(2s)']
df_pValues.columns = ['p(16s)','p(8s)','p(6s)','p(4s)','p(3s)','p(2s)']

# use data frames to create plots
# iterate over ROIs
for i_roi in range(np.shape(masks)[0]):
    if i_roi == 0:
        curr_title = 'Liver Edge'
    if i_roi == 1:
        curr_title = 'Kidney Medulla'
    if i_roi == 2:
        curr_title = 'Kidney Cortex'
    if i_roi == 3:
        curr_title = 'Blood Vessel'
    if i_roi == 4:
        curr_title = 'Below Blood Vessel'

    fig, ax = plt.subplots()
    bar_width = 0.2
    i_p = 10
    # iterate over scan times
    for i_time in range(len(filenumber_list)):
        currMean = df_meanValues.iloc[i_roi,i_time]
        currStd = df_stdValues.iloc[i_roi,i_time]
        rect = plt.bar(i_time + bar_width, currMean, bar_width, alpha=0.8, color='b',yerr=currStd)
        if(i_time > 0):
            currPValue = df_pValues.iloc[i_roi,i_time]
            if(currPValue < 0.05):
                y, h, col = max(currMean,df_meanValues.iloc[i_roi,0])+max(currStd,df_stdValues.iloc[i_roi,0]) + i_p, 4, 'k'
                plt.plot([bar_width,bar_width, i_time+bar_width, i_time+bar_width], [y, y + h, y + h, y], lw=1.5, c=col)
                plt.text((i_time + 2*bar_width) / 2, y, '*', ha='center', va='bottom', color=col,
                         fontsize='large')
                i_p = i_p+40
    index = np.arange(np.shape(masks)[0]+1)
    plt.xlabel('Scan duration', fontsize='x-large')
    plt.ylabel('T1 values', fontsize='x-large')
    plt.xticks(index + bar_width, ('16s', '8s', '6s', '4s', '3s', '2s'), fontsize='large')
    plt.title(curr_title, fontsize='x-large', fontweight='bold')



