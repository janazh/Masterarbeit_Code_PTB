'''
@date 30.09.2021, @author JZ

This script calculates for every volunteer the average T1 values and the standard deviation of T1 over the ROI
of each ROI and stores them into a table for each ROI. The table for each ROI then contains all volunteers with their
according averaged T1 value and standard deviation of the ROI.

As input, the folderpaths to the motion-corrected, uncorrected and breathhold T1 maps have to be assigned. The folderpath
to the ROI masks (masks for free breathing and breathhold) has to be given as well. For storing the calculated tables
(data frames), filepaths to each ROI table have to be assigned (the ending for storing a dataframe is .pkl)

For the masterthesis, the dataframes have already been created and stored. The folderpath where the dataframes are stored
is given in the main text file.
'''

import pandas as pd
import numpy as np
import numpy.ma as mas

# folderpath to the motion-corrected T1 maps
folderpath_T1maps_moCo = '/data/zhang18/eval_T1maps_data/moCo/'
# folderpath to the uncorrected T1 maps
folderpath_T1maps_noMoCo = '/data/zhang18/eval_T1maps_data/noMoCo/'
# folderpath to the breathhold T1 maps
folderpath_T1maps_bh = '/data/zhang18/eval_T1maps_data/breathhold/'
# folderpath to the masks that describe the chosen regions of interest
folderpath_roi_masks = '/data/zhang18/eval_T1maps_data/roi_masks_new/'
# filepaths to where the dataframes for the ROIs shall be stored (choose ending .pkl)
filepath_df_roi1 = '/data/zhang18/eval_T1maps_data/roi_dataframes_new/df_roi1.pkl'
filepath_df_roi2 = '/data/zhang18/eval_T1maps_data/roi_dataframes_new/df_roi2.pkl'
filepath_df_roi3 = '/data/zhang18/eval_T1maps_data/roi_dataframes_new/df_roi3.pkl'
filepath_df_roi4 = '/data/zhang18/eval_T1maps_data/roi_dataframes_new/df_roi4.pkl'
filepath_df_roi5 = '/data/zhang18/eval_T1maps_data/roi_dataframes_new/df_roi5.pkl'

filenumber_list = [122,29,101,182,236]


column_names = ['noMoCo_mean','noMoCo_std','moCo_mean','moCo_std','breathhold_mean','breathhold_std']
df_roi1 = pd.DataFrame(columns = column_names)
df_roi2 = pd.DataFrame(columns = column_names)
df_roi3 = pd.DataFrame(columns = column_names)
df_roi4 = pd.DataFrame(columns = column_names)
df_roi5 = pd.DataFrame(columns = column_names)

for i_iter,i_file in enumerate(filenumber_list):
    # get T1 maps
    curr_idatqpar_noMoCo = np.load(folderpath_T1maps_noMoCo + 'MID'+ str(i_file) + '_idat_qpar_noMoCo.npy')
    curr_idatqpar_moCo   = np.load(folderpath_T1maps_moCo + 'MID'+ str(i_file) + '_idat_qpar_moCo.npy')
    curr_idatqpar_bh     = np.load(folderpath_T1maps_bh + 'MID'+ str(i_file) + '_idat_qpar_bh.npy')
    curr_t1map_noMoCo = 1/abs(curr_idatqpar_noMoCo[:,:,2])*1000
    curr_t1map_moCo =  1/abs(curr_idatqpar_moCo[:,:,2])*1000
    curr_t1map_bh = 1/abs(curr_idatqpar_bh[:,:,2])*1000
    # get stack of masks
    curr_masks_fb = np.load(folderpath_roi_masks + 'MID' + str(i_file) + '_masks_fb.npy')
    curr_masks_bh = np.load(folderpath_roi_masks + 'MID' + str(i_file) + '_masks_bh.npy')

    # get average T1 values for each ROI
    for i_mask in range(np.shape(curr_masks_fb)[0]):
        curr_mask_fb = curr_masks_fb[i_mask,:,:]
        curr_mask_bh = curr_masks_bh[i_mask,:,:]
        # get masked T1 maps
        curr_t1map_noMoCo_roi = ma.masked_array(curr_t1map_noMoCo, np.logical_not(curr_mask_fb))
        curr_t1map_moCo_roi   = ma.masked_array(curr_t1map_moCo, np.logical_not(curr_mask_fb))
        curr_t1map_bh_roi     = ma.masked_array(curr_t1map_bh, np.logical_not(curr_mask_bh))
        # get mean values of each roi and put into list
        curr_roi_meanStdVals = [curr_t1map_noMoCo_roi.mean(),curr_t1map_noMoCo_roi.std(),curr_t1map_moCo_roi.mean(),curr_t1map_moCo_roi.std(),curr_t1map_bh_roi.mean(),curr_t1map_bh_roi.std()]

        if(i_mask == 0):
            df_roi1.loc[i_iter] = curr_roi_meanStdVals
        if(i_mask == 1):
            df_roi2.loc[i_iter] = curr_roi_meanStdVals
        if(i_mask == 2):
            df_roi3.loc[i_iter] = curr_roi_meanStdVals
        if(i_mask == 3):
            df_roi4.loc[i_iter] = curr_roi_meanStdVals
        if(i_mask == 4):
            df_roi5.loc[i_iter] = curr_roi_meanStdVals

# save data frames
df_roi1.to_pickle(filepath_df_roi1)
df_roi2.to_pickle(filepath_df_roi2)
df_roi3.to_pickle(filepath_df_roi3)
df_roi4.to_pickle(filepath_df_roi4)
df_roi5.to_pickle(filepath_df_roi5)
