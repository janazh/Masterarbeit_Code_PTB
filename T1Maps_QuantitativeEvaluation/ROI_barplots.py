'''
@author: Jana Zhang, @date: 28.09.2021
Masterthesis Section "Quantitative Evluation of Motion-Corrected T1 Maps", Fig. 4.4.7 - 4.4.11
This script generates for every region of interest a figure. The figure contains the averaged (over the region)
uncorrected, motion-corrected and breath hold T1 values with the corresponding standard deviation for each volunteer.

As input, the filepaths to the dataframes containing information about averaged T1 value and standard deviation for each
region of interest are given.
'''

# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

#amount of ROIs
amount_roi = 5
# amount of subjects (volunteers)
amount_subjects = 5
# iterate over every ROI
for i_roi in range(amount_roi):
    if i_roi == 0:
        curr_df = df_roi1
        curr_title = 'Liver Edge'
    if i_roi == 1:
        curr_df = df_roi2
        curr_title = 'Kidney Medulla'
    if i_roi == 2:
        curr_df = df_roi3
        curr_title = 'Kidney Cortex'
    if i_roi == 3:
        curr_df = df_roi4
        curr_title = 'Blood Vessel'
    if i_roi == 4:
        curr_df = df_roi5
        curr_title = 'Below Blood Vessel'

    # create plot
    fig, ax = plt.subplots(figsize=(8,6))
    bar_width = 0.2
    # iterate over every subject
    for i_subj in range(amount_subjects):
        curr_noMoCo_mean = curr_df['noMoCo_mean'][i_subj]
        curr_moCo_mean = curr_df['moCo_mean'][i_subj]
        curr_bh_mean = curr_df['breathhold_mean'][i_subj]
        curr_noMoCo_std = curr_df['noMoCo_std'][i_subj]
        curr_moCo_std = curr_df['moCo_std'][i_subj]
        curr_bh_std = curr_df['breathhold_std'][i_subj]

        rects1 = plt.bar(i_subj, curr_noMoCo_mean, bar_width, alpha = 0.8, color = 'orange', label = 'noMoCo', yerr = curr_noMoCo_std)
        rects2 = plt.bar(i_subj + bar_width, curr_moCo_mean, bar_width, alpha = 0.8, color = 'g', label = 'moCo', yerr = curr_moCo_std)
        rects3 = plt.bar(i_subj + 2*bar_width, curr_bh_mean, bar_width, alpha = 0.8, color = 'b', label = 'breath hold', yerr = curr_bh_std)

    index = np.arange(amount_subjects)
    plt.ylabel('Average T1 values in ms', fontsize='x-large')
    plt.title(curr_title, fontsize='x-large', fontweight='bold')
    plt.xticks(index+bar_width, ('Volunteer 1','Volunteer 2','Volunteer 3','Volunteer 4','Volunteer 5'), fontsize='large')
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[0:3], labels[0:3], loc = 'upper left',bbox_to_anchor = (0,1.1), fontsize='large')

    plt.tight_layout()
    plt.show()