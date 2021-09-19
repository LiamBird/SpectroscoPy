"""
SpectrumMap
Version: 25/05/2021

Modification: 
05/05/2021: Version started

"""

import numpy as np
import os

def _slide_viewer(self, x, y, xmin=None, xmax=None):
    from ipywidgets import interact, IntSlider
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)

    def update(xi, yi):
        ax.cla()
        ax.plot(x, y[xi, yi, :])
        ax.set_xlim([xmin, xmax])


    interact(update, xi=IntSlider(min=0, max=y.shape[0]-1, step=1),
                     yi=IntSlider(min=0, max=y.shape[1]-1, step=1))

class SpectrumMap(object):
    """
    A class to load and aid navigation of Raman map data.
    ...
    
    Input arguments
    -----
    filename: str
        name and path of the four-column text file of Raman map data, including file extension

    Attributes
    -----
    filename: str
        filename of input file without path or extension (for user reference)
    imported_data: numpy array
        4 column input data (x position | y position | shift | intensity)
    x_coords: numpy array
        x positions corresponding to spectra positions
    y_coords: numpy array
        y positions corresponding to spectra positions
    shift: numpy array
        the Raman shift axis corresponding to the map data
    x_extent: int
        number of x positions at which spectra were collected
    y_extent: int
        number of y positions at which spectra were collected
    shift_extent: int
        number of shift values
    map: numpy array
        3d array containing the map data (dimensions for x, y, and shift axes)
    slide_viewer(xmin=None, ymin=None):
        returns an interactive (ipywidgets) plot of the signal.

    Methods
    -----
    normalise(norm_peak, norm_peak_tolerance=5):
        returns (as an Attribute) the normalised intensity of the whole shift range for a given peak shift value
    
    roi_select(roi_min, roi_max, norm_peak=None):
        returns the shift and (normalised) intensity arrays for a region of interest (roi) defined by shift
        
    slide_viewer(xmin=None, ymin=None):
        returns an interactive (ipywidgets) plot of the signal.

    data_clip(start_shift):
        clips the lower wavenumbers from a map of Raman spectra (designed to remove abrupt step in background)
        returns new attribute ("clipped_data")

    ccd_pixel(pixel_pos, smooth_width=5):
        replaces portion of each spectrum in a map (2*smooth width, centered on pixel_pos) with random noise to account for
        assumes that same pixel in each spectrum is affected (not suitable for random cosmic rays)
        
    cosmic_ray_remove(step=2, threshold=3, smooth_width=5):
        
        
    """
    def __init__(self, filename):
        self.filename = os.path.split(filename)[-1][:-4]
        self.imported_data = np.loadtxt(filename, delimiter="\t")
        self.x_coords = np.unique(self.imported_data[:, 0])
        self.y_coords = np.unique(self.imported_data[:, 1])
        self.shift = np.unique(self.imported_data[:, 2])
        self.x_extent = self.x_coords.shape[0]
        self.y_extent = self.y_coords.shape[0]
        self.shift_extent = self.shift.shape[0]

        self.map = np.zeros((self.x_extent, self.y_extent, self.shift_extent))
        for x in range(self.x_extent):
            for y in range(self.y_extent):
                self.map[x, y, :] = self._spectrum_pos(self.imported_data, x, y)[::-1]
                ## [::-1] to reverse data because file format records high to low shift as default

        
    def _spectrum_pos(self, input_data, x_pos, y_pos):
        return input_data[np.argwhere((input_data[:, 0]==self.x_coords[x_pos]) \
                                      & (input_data[:, 1]==self.y_coords[y_pos])), -1].flatten()

    def normalise(self, norm_peak, norm_peak_tolerance=5):
        """
        Returns (as an Attribute of SpectrumMap) the instensity of the full shift range for a specified peak shift position.
        ...

        Input arguments:
        -----
        norm_peak: float
            The shift value to which the data will be normalised
        norm_peak_tolerance: float (optional, default=5)
            The data are normalised to the maximum value within the range of norm_peak +/- norm_peak_tolerance

        Returns:
        -----
        SpectrumMap.norm_#: attribute labelled with #=norm_peak value
            Array of normalised data. Can be utilised directly by roi_select method without recalculating
                
        """
        norm_range_min = np.argmin(abs(self.shift-(norm_peak-norm_peak_tolerance)))
        norm_range_max = np.argmin(abs(self.shift-(norm_peak+norm_peak_tolerance)))

        norm_data = np.zeros((self.x_extent, self.y_extent, self.shift_extent))
        for x in range(self.x_extent):
            for y in range(self.y_extent):
                spectrum_max = max(self.map[x, y, norm_range_min:norm_range_max])
                norm_data[x, y, :] = self.map[x, y, :]/spectrum_max

        setattr(self, "norm_"+str(norm_peak), norm_data)

    def roi_select(self, roi_min, roi_max, norm_peak=None):
        """
        Returns an ROI object with shift and intensity attributes for a region of interest (roi) selected according to shift limits
        ...

        Input arguments:
        -----
        roi_min: float
            Lower shift limit of region of interest (cm-1)
        roi_max: float
            Upper shift limit of region of interest (cm-1)

            
        norm_peak: float (optional)
            shift value for which the data should be normalised (cm-1)
            if defined, the data is normalised to the intensity at the specified shift value
            NB: CAN be outside of region of interest shift values - normalisation completed before roi selected

        Returns:
        -----
        ROI class with attributes:
            shift: array
                Shift axis relevant to the region of interest
            intensity:
                3d array (x position, y position, shift) relevant to the region of interest
        """
        
        idx_min = np.argmin(abs(self.shift-roi_min))
        idx_max = np.argmin(abs(self.shift-roi_max))

        if norm_peak == None:
            data = self.map
        elif "norm_"+str(norm_peak) != None and "norm_"+str(norm_peak) in vars(self):
            data = vars(self)["norm_"+str(norm_peak)]
        elif "norm_"+str(norm_peak) != None and "norm_"+str(norm_peak) not in vars(self):
            self.normalise(norm_peak)
            data = vars(self)["norm_"+str(norm_peak)]
        
        class ROI(object):
            def __init__(roi_self):
                roi_self.shift = self.shift[idx_min:idx_max]
                roi_self.intensity = data[:, :, idx_min:idx_max]
            def slide_viewer(roi_self, xmin=None, xmax=None):
                _slide_viewer(roi_self, x=roi_self.shift, y=roi_self.intensity, xmin=xmin, xmax=xmax)
                
        return ROI()
    
    def slide_viewer(self, xmin=None, xmax=None):
        """
        Displays widget for viewing spectra according to x and y position in jupyter notebook
        """
        _slide_viewer(self, x=self.shift, y=self.map, xmin=xmin, xmax=xmax)

    def data_clip(self, start_shift=None, end_shift=None):
        if start_shift != None:
            start_idx = np.argmin(abs(self.shift-start_shift))
        else:
            start_idx = None
        if end_shift != None:
            end_idx = np.argmin(abs(self.shift-end_shift))
        else:
            end_idx = None
            
        class ClippedData(object):
            def __init__(clipped_self):
                clipped_self.shift = self.shift[start_idx:end_idx]
                clipped_self.map = self.map[:, :, start_idx:end_idx]
                clipped_self.x_extent = self.x_extent
                clipped_self.y_extent = self.y_extent
                clipped_self.shift_extent = self.shift[start_idx:end_idx].shape[0]

        setattr(self, "clipped_data", ClippedData())

    def ccd_pixel(self, pixel_pos, smooth_width=5, use_clipped_data=True):
        if use_clipped_data == True and "clipped_data" in vars(self):
#             print("Using clipped data range")
            pixel_idx = np.argmin(abs(self.clipped_data.shift-pixel_pos))
#             print("Pixel idx: ", pixel_idx)
#             print("Pixel shift (clipped): ", self.clipped_data.shift[pixel_idx])
            map_data = self.clipped_data.map
        else:
            print("Using full range")
            pixel_idx = np.argmin(abs(self.shift-pixel_pos))
            map_data = self.map

        CCD_map = np.zeros((map_data.shape))
        for x in range(CCD_map.shape[0]):
            for y in range(CCD_map.shape[1]):
                surround_min = min([min(map_data[x, y, pixel_idx-2*smooth_width:pixel_idx-smooth_width]),
                                    min(map_data[x, y, pixel_idx+smooth_width:pixel_idx+2*smooth_width])])
                surround_max = max([max(map_data[x, y, pixel_idx-2*smooth_width:pixel_idx-smooth_width]),
                                    max(map_data[x, y, pixel_idx+smooth_width:pixel_idx+2*smooth_width])])
                noisy_smooth = surround_min+(surround_max-surround_min)*np.random.rand(2*smooth_width)

                CCD_map[x, y, :pixel_idx-smooth_width] = map_data[x, y, :pixel_idx-smooth_width]
                CCD_map[x, y, pixel_idx-smooth_width:pixel_idx+smooth_width] = noisy_smooth
                CCD_map[x, y, pixel_idx+smooth_width:] = map_data[x, y, pixel_idx+smooth_width:]

        setattr(self, "map_CCD", CCD_map)                                  
