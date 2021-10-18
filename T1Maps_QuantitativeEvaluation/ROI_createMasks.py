'''
@author: Jana Zhang, @date: 30.09.2021

With this script, regions of interests can defined on a T1 map. For this, the according T1 map has to be loaded.
Afterwards, an interactive ROI-tool is created and five (number can be adapted) regions of interest have to be chosen.
The masks and patches are then stored according to the pre-defined filepaths. The storing has to be executed separately
after using the ROI tool (mark the lines in the code and execute with ALT+Shift+E.

For the master thesis, this script has already been executed and the ROIS have been chosen. ROIs have been chosen
for the breathhold T1 maps as well as for the motion-corrected T1 maps. For the uncorrected T1 maps, the motion-corrected
masks were used. The folderpath to the stored masks is given in the main text file.
'''


import roitool
import numpy as np
import matplotlib.pyplot as plt
import pickle

# define filepath to file where T1 map is stored
filepath_T1map = "/data/zhang18/eval_T1maps_data/breathhold/MID101_idat_qpar_bh.npy"
# define filepath to file where ROI masks shall be stored
filepath_ROImasks = '/data/zhang18/eval_T1maps_data/roi_masks_new/MID101_masks_bh.npy'
# define filepath to file where ROI patches shall be stored
filepath_ROIpatches = '/data/zhang18/eval_T1maps_data/roi_patches_new/MID101_patches_bh.pkl'


# load image dat
idat_qpar = np.load(filepath_T1map)
idat_map = 1/abs(idat_qpar[:,:,2])*1000

# start interactive ROI-tool
number_of_rois = 5 # can be adapted if more ROIs are needed
shape_of_roi = 'ellipse'  # Supported shapes: 'ellipse', 'rectangle', 'lasso', 'polygon'

roi_obj = roitool.roitool(img=idat_map, num_roi=number_of_rois, roi_shape=shape_of_roi)
roi_masks = roi_obj.roi_stack
roi_patches = roi_obj.roi_patches

# # store masks # has to be executed separately after interactive ROI-tool!!!
np.save(filepath_ROImasks, roi_masks)
#store patches # has to be executed separately after interactive ROI-tool!!!
open_file = open(filepath_ROIpatches, 'wb')
pickle.dump(roi_patches, open_file)
open_file.close()


# # plot masks
# if roi_masks.shape[0] == 1:
#     plt.imshow(roi_masks[0])
# else:
#     fig, axs = plt.subplots(1, roi_masks.shape[0])
#     for i, ax in enumerate(axs.flatten()):
#         ax.imshow(roi_masks[i,])
#
#
# # plot image with ROIs
# fig, ax = plt.subplots()
# ax.imshow(idat_map,cmap='magma',vmin = 0, vmax = 2500)
# for patch in roi_patches:
#     ax.add_patch(patch)




