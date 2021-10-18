'''
@author: Jana Zhang, @date: 30.09.2021
Master thesis section "Quantitative Evaluation of Motion-Corrected T1 Maps", Fig. 4.4.14

This script creates a plot that shows the standard deviation of the T1 value over all volunteers for each region of interest.
For each region, the standard deviation is shown for the uncorrected, the motion-corrected and the breathhold case.

As input, the filepaths to the dataframes containing information about averaged T1 value and standard deviation for each
region of interest are given.
'''

# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# assign filepaths
filepath_df_roi1 = '/data/zhang18/eval_T1maps_data/roi_dataframes_new/df_roi1.pkl'
filepath_df_roi2 = '/data/zhang18/eval_T1maps_data/roi_dataframes_new/df_roi2.pkl'
filepath_df_roi3 = '/data/zhang18/eval_T1maps_data/roi_dataframes_new/df_roi3.pkl'
filepath_df_roi4 = '/data/zhang18/eval_T1maps_data/roi_dataframes_new/df_roi4.pkl'
filepath_df_roi5 = '/data/zhang18/eval_T1maps_data/roi_dataframes_new/df_roi5.pkl'
# load dataframes
df_roi1 = pd.read_pickle(filepath_df_roi1)
df_roi2 = pd.read_pickle(filepath_df_roi2)
df_roi3 = pd.read_pickle(filepath_df_roi3)
df_roi4 = pd.read_pickle(filepath_df_roi4)
df_roi5 = pd.read_pickle(filepath_df_roi5)
# iterate over all ROIs
#amount of ROIs
amount_roi = 5
# create plot
fig, ax = plt.subplots()
bar_width = 0.2
# iterate over every ROI
for i_roi in range(amount_roi):
    if i_roi == 0:
        curr_df = df_roi1
    if i_roi == 1:
        curr_df = df_roi2
    if i_roi == 2:
        curr_df = df_roi3
    if i_roi == 3:
        curr_df = df_roi4
    if i_roi == 4:
        curr_df = df_roi5

    # arrays contain T1 value of ROI of every volunteer (in this case 5 values in total)
    currMean_moCo_arr = curr_df['moCo_mean']
    currMean_bh_arr = curr_df['breathhold_mean']

    rects2 = plt.bar(i_roi + bar_width, currMean_moCo_arr.mean(), bar_width, alpha = 0.8, color = 'b', label = 'moCo', yerr = currMean_moCo_arr.std())
    rects3 = plt.bar(i_roi + 2*bar_width, currMean_bh_arr.mean(), bar_width, alpha = 0.8, color = 'g', label = 'breath hold', yerr = currMean_bh_arr.std())

    # check if data is distributed normally
    w_shapwilk_bh, p_shapwilk_bh = stats.shapiro(currMean_bh_arr)
    w_shapwilk_moCo, p_shapwilk_moCo =stats.shapiro(currMean_moCo_arr)
    # statistical significance between moCo and breathhold
    if(p_shapwilk_bh > 0.05 or p_shapwilk_moCo > 0.05):
        #for normally distributed data, use student t-test
        stat_moCo_bh, p_moCo_bh = stats.ttest_rel(currMean_moCo_arr,currMean_bh_arr)
    else:
        #for not normally distributed data, use wilcoxon test
        stat_moCo_bh, p_moCo_bh = stats.wilcoxon(currMean_moCo_arr,currMean_bh_arr)
    if(p_moCo_bh > 0.05):
        y, h, col = max([currMean_bh_arr.mean(),currMean_moCo_arr.mean()])+ max([currMean_bh_arr.std(),currMean_moCo_arr.std()])+20, 2, 'k'
        plt.plot([i_roi+bar_width,i_roi+bar_width,i_roi+2*bar_width,i_roi+2*bar_width],[y,y+h,y+h,y], lw=1.5, c=col)
        plt.text(i_roi+1.5*bar_width,y+h, 'ns',ha='center',va='bottom',color=col,fontsize='large')

index = np.arange(amount_roi)
plt.xlabel('ROIs',fontsize='x-large')
plt.ylabel('T1 values in ms',fontsize='x-large')
plt.title('Averaged T1 value over all volunteers for each ROI',fontsize='x-large')
plt.xticks(index+1.5*bar_width, ('liver edge','kidney medulla','kidney cortex','blood vessel','below blood vessel'),fontsize='large')
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles[0:2],labels[0:2],fontsize='large')

plt.tight_layout()
plt.show()
