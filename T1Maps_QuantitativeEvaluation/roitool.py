import numpy as np
import matplotlib.pyplot as plt

from matplotlib.widgets import RectangleSelector
from matplotlib.widgets import EllipseSelector
from matplotlib.widgets import LassoSelector
from matplotlib.widgets import PolygonSelector
from matplotlib.widgets import Button
from matplotlib.patches import PathPatch
from matplotlib.path import Path

class roitool():
    def __init__(self, img, num_roi=1, roi_shape='ellipse'):
        self.img = img
        self.n_roi = num_roi
        self.shape = roi_shape.lower()
        self.counter = 0
        self.roi_stack = np.zeros([num_roi, img.shape[1], img.shape[0]])
        self.roi_patches = []

        # create meshgrid of current image
        x, y = np.meshgrid(np.arange(img.shape[1], dtype=int),
                           np.arange(img.shape[0], dtype=int))

        # convert meshgrid to array of tuples
        self.pixel_array = np.vstack((x.flatten(), y.flatten())).T

        self.plot()
        self.start_selector()

    def onselect(self,vertices,unused=None):
        # extract vertices for 'ellipse' and 'rectangle' case
        if self.shape in ['ellipse', 'rectangle']:
            vertices = list(zip(self.selector.geometry[0],
                                self.selector.geometry[1]))

        # create path from vertices
        self.roi_path = Path(vertices)

        # get indices of pixels inside the path
        idx = self.roi_path.contains_points(self.pixel_array, radius=1)

        # create empty mask and set values inside path/mask to True
        self.roi_mask = np.zeros_like(self.img, bool)
        self.roi_mask.flat[idx] = True

        # draw the current path of 'lasso' selection
        if self.shape == 'lasso':
            self.ax.add_patch(PathPatch(self.roi_path))

        self.fig.canvas.draw_idle()

    def toggle_selector(self, event):
        # write current mask to roi_stack
        self.roi_stack[self.counter, :, :] = self.roi_mask
        # self.roi_patches.append(PathPatch(self.roi_path))
        self.roi_patches.append(PathPatch(self.roi_path, alpha = 0.5, color = 'w' ))

        # remove patches from figure
        [p.remove() for p in reversed(self.ax.patches)]

        # increase counter and update title
        self.counter += 1
        if self.counter < self.n_roi:
            self.ax.set_title(f'DRAW AND CONFIRM ROI #{self.counter + 1} of {self.n_roi}',
                              fontdict={'fontsize': 12, 'color': 'red'})
            self.fig.canvas.draw_idle()
        else:
            plt.close()

    def plot(self):
        # plot image and set title
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.img,cmap='magma',vmin = 0, vmax = 2500)
        self.ax.set_title(f'DRAW AND CONFIRM ROI #{self.counter + 1} of {self.n_roi}',
                          fontdict={'fontsize': 12, 'color': 'red'})

        # add button to confirm selection
        axbtn = plt.axes([0.01, 0.94, 0.15, 0.05])
        self.btn = Button(axbtn, 'Confirm ROI')
        self.btn.on_clicked(self.toggle_selector)

    def start_selector(self):
        if self.shape == 'ellipse':
            self.selector = EllipseSelector(self.ax, onselect = self.onselect, drawtype="line", interactive=True)

        elif self.shape == 'rectangle':
            self.selector = RectangleSelector(self.ax, onselect=self.onselect, drawtype="box", interactive=True)

        elif self.shape == 'lasso':
            self.selector = LassoSelector(self.ax, onselect=self.onselect,
                                          lineprops=dict(color='r', linestyle='-', linewidth=2))

        elif self.shape == 'polygon':
            self.selector = PolygonSelector(self.ax, onselect=self.onselect,
                                            lineprops=dict(color='r', linestyle='-', linewidth=2))
