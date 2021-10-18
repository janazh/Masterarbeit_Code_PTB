'''
@author: Jana Zhang, @date: 30.09.2021
Master thesis section

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
    currMean_noMoCo_arr = curr_df['noMoCo_mean']
    currMean_moCo_arr = curr_df['moCo_mean']
    currMean_bh_arr = curr_df['breathhold_mean']

    rects1 = plt.bar(i_roi, currMean_noMoCo_arr.std(), bar_width, alpha = 0.8, color = 'y', label = 'noMoCo')
    rects2 = plt.bar(i_roi + bar_width, currMean_moCo_arr.std(), bar_width, alpha = 0.8, color = 'b', label = 'moCo')
    rects3 = plt.bar(i_roi + 2*bar_width, currMean_bh_arr.std(), bar_width, alpha = 0.8, color = 'g', label = 'breath hold')

index = np.arange(amount_roi)
plt.xlabel('ROIs',fontsize='x-large')
plt.ylabel('Standard deviation of T1 value in ms',fontsize='x-large')
plt.title('Standard deviation of T1 value over all volunteers for each ROI',fontsize='x-large')
plt.xticks(index+bar_width, ('liver edge','kidney medulla','kidney cortex','blood vessel','below blood vessel'),fontsize='large')
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles[0:3],labels[0:3],fontsize='large')

plt.tight_layout()
plt.show()
