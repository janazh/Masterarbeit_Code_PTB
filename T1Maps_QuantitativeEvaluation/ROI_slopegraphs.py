'''
@date:16.06.2021, @author:JZ
Master thesis "Quantitative Evluation of Motion-Corrected T1 Maps" (Fig. 4.4.12)
With this script, slopegraphs are created for each region of interest. The slopegraphs connect the noMoCo and moCo values.
Values are relative differences to breathhold T1 values. P values are calculated and p values smaller than 0.05 are
marked with *.

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

#amount of ROIs
amount_roi = 5
# amount of subjects (volunteers)
amount_subjects = 5
#create graph
red = '#C25539'
blue = '#3F7F93'
fig,ax = plt.subplots()
i = 1
# iterate over every ROI
for i_roi in range(amount_roi):
    if i_roi == 0:
        curr_df = df_roi1
        curr_title = 'ROI1: liver edge'
    if i_roi == 1:
        curr_df = df_roi2
        curr_title = 'ROI2: kidney medulla'
    if i_roi == 2:
        curr_df = df_roi3
        curr_title = 'ROI3: kidney cortex'
    if i_roi == 3:
        curr_df = df_roi4
        curr_title = 'ROI4: blood vessel'
    if i_roi == 4:
        curr_df = df_roi5
        curr_title = 'ROI5: below blood vessel'

    # relative difference noMoCo and breathhold as well as moCo and breathhold (in percent)
    curr_noMoCo = np.array(curr_df['noMoCo_mean'])
    curr_moCo = np.array(curr_df['moCo_mean'])
    curr_breathhold = np.array(curr_df['breathhold_mean'])

    curr_noMoCo_diff = (abs(curr_noMoCo - curr_breathhold)*100/curr_breathhold)
    curr_moCo_diff = (abs(curr_moCo - curr_breathhold)*100/curr_breathhold)

    #mean of relative differences
    mean_noMoCo_diff = np.mean(curr_noMoCo_diff)
    mean_moCo_diff = np.mean(curr_moCo_diff)

    # set up x-axis values
    x1 = i-0.2
    x2 = i+0.2

    # set up line color
    line_colors = (curr_noMoCo_diff - curr_moCo_diff) > 0
    line_colors = [blue if j else red for j in line_colors]

    #set up alpha values for slightly transparent lines
    alphas = [0.5]*len(line_colors)
    # plot the lines connecting the dots
    for i_noMoCo, i_moCo, ci, ai in zip(curr_noMoCo_diff, curr_moCo_diff,line_colors,alphas):
        ax.plot([x1, x2],[i_noMoCo, i_moCo], c=ci, alpha=ai)

    #plot mean
    ax.plot([x1,x2],[mean_noMoCo_diff,mean_moCo_diff], c='magenta',label='average value')

    # plot the points
    ax.scatter(len(curr_noMoCo_diff)*[x1-0.01], curr_noMoCo_diff, c='orange', s=25, lw=0.5, label='without motion correction')
    ax.scatter(len(curr_moCo_diff)*[x2+0.01], curr_moCo_diff, c='g', s=25, lw=0.5, label='with motion correction')
    ax.scatter([x1-0.01],mean_noMoCo_diff, c='orange', s=25, lw=0.5,)
    ax.scatter([x2+0.01],mean_moCo_diff, c='g', s=25, lw=0.5,)
    # update x-axis
    i += 1

    #check if data is normally distributed
    w_shapiro_noMoCo, p_shapiro_noMoCo = stats.shapiro(curr_noMoCo_diff)
    w_shapiro_moCo, p_shapiro_moCo = stats.shapiro(curr_moCo_diff)

    # calculate p-Value
    if(p_shapiro_noMoCo > 0.05 or p_shapiro_moCo > 0.05):
        stat_noMoCo_moCo, p_noMoCo_moCo = stats.ttest_rel(curr_noMoCo_diff,curr_moCo_diff)
    else:
        stat_noMoCo_moCo, p_noMoCo_moCo = stats.wilcoxon(curr_noMoCo_diff,curr_moCo_diff)

    # divide p by 2, because it is a one-sided test
    if (p_noMoCo_moCo/2 <= 0.05):
        y, h, col = max(curr_noMoCo_diff) + 2, 2, 'k'
        plt.plot([x1,x1,x2,x2], [y, y + h, y + h, y], lw=1.5, c=col)
        plt.text((x1+x2)/2, y + h, '*', ha='center', va='bottom', color=col,
            fontsize='large')

# Fix axes
ax.set_xticks([1,2,3,4,5])
_ = ax.set_xticklabels(['Liver Edge','Kidney Medulla','Kidney Cortex','Blood Vessel','Below Blood Vessel'], fontsize='x-large')

plt.ylabel('Relative Difference of T1 values to Breath Hold in %',fontsize='x-large')

# add legend
handles, labels = ax.get_legend_handles_labels()
handles_adapt = list(handles[i] for i in [0,5,6])
labels_adapt = list(labels[i] for i in [0,5,6])
lgd = ax.legend(handles_adapt, labels_adapt,
              fontsize='x-large',
              loc = 'upper center',
              ncol=1,
              scatterpoints=1)
lgd.legendHandles[0]._sizes = [80]
lgd.legendHandles[1]._sizes = [80]
lgd.legendHandles[2]._sizes = [80]
plt.show()