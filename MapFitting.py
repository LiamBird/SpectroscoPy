"""
Version 1
Created 26/10/2021
Remember to include snv.py function in same directory! 
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import glob

from snv import snv                     ### Requires snv (available in same repository)
from lmfit import Model, Parameters
from lmfit.models import LorentzianModel, LinearModel

try:
    from tqdm import notebook
    from notebook import tqdm as notebook
except:
    from tqdm import tqdm_notebook as notebook

class _FitDict(object):
    """
    A hidden class to contain fit results for an individual peak
    """
    def __init__(self, center, sigma, amplitude, intensity, best_fit, shift):
        keys = ["center", "sigma", "amplitude", "intensity", "best_fit", "shift"]
        values = [center, sigma, amplitude, intensity, best_fit, shift]
        self.__dict__.update([(key, values[nkey]) for nkey, key in enumerate(keys)])
        
class FitResults(object):
    """
    A class to store, retrieve and display the results of processing using FitMap.
    Returned as an attribute in local FitMap class, or can be used to conveniently re-load previously fitted model data.

    Input arguments
    ----------
    FitMap_object: FitMap object, default=None
        a locally defined FitMap object

    savedir: str, default=None:
        the directory to save the array-formatted results to

    loaddir: str, default=None:
        the top directory containing subdirectories of array-formatted results for all relevant peaks

    Attributes
    ----------
    For each peak in either peaks_to_fit of FitMap_object, or each subdirectory of loaddir, a _FitDict object containing arrays for:
        - center
        - sigma
        - amplitude
        - intensity (calculated from amplitude and sigma)
        - best fit (3d array with shift dimension)
        - shift  

    Methods
    ----------
    heatmap(peaks, quantity="intensity"):
        returns heatmaps showing spatial variation of fitted values
    """
    def __init__(self, FitMap_object=None, savedir=None, loaddir=None):
        if FitMap_object != None:
            if savedir != None:
                if os.path.isdir(savedir) == False:
                    os.mkdir(savedir)

            peak_outputs = {}

            for peak in FitMap_object.peaks_to_fit.keys():                   
                ROI_group = [key for key, value in FitMap_object.ROI_groups.items() if peak in value][0]
                center = np.full((FitMap_object.x_extent, FitMap_object.y_extent), np.nan)
                sigma = np.full((FitMap_object.x_extent, FitMap_object.y_extent), np.nan)
                amplitude = np.full((FitMap_object.x_extent, FitMap_object.y_extent), np.nan)
                intensity = np.full((FitMap_object.x_extent, FitMap_object.y_extent), np.nan)

                shift_extent = FitMap_object.ROI[ROI_group]["shift"].shape[0]
                best_fit = np.full((FitMap_object.x_extent, FitMap_object.y_extent, shift_extent), np.nan)

                for x in range(FitMap_object.x_extent):
                    for y in range(FitMap_object.y_extent):
                        center[x, y] = FitMap_object.ROI_fits[ROI_group][x, y].best_values[peak+"center"]
                        sigma[x, y] = FitMap_object.ROI_fits[ROI_group][x, y].best_values[peak+"sigma"]
                        amplitude[x, y] = FitMap_object.ROI_fits[ROI_group][x, y].best_values[peak+"amplitude"]
                        intensity = amplitude/np.pi/2/sigma

                        best_fit[x, y, :] = FitMap_object.ROI_fits[ROI_group][x, y].best_fit

                setattr(self, peak, _FitDict(center=center, 
                                            sigma=sigma,
                                            amplitude=amplitude,
                                            intensity=intensity,
                                            best_fit=best_fit,
                                            shift=FitMap_object.ROI[ROI_group]["shift"]))
                if savedir != None:
                    if peak not in os.listdir(os.path.join(savedir)):
                        os.mkdir(os.path.join(savedir, peak))
                    [np.save(os.path.join(savedir, peak, "{}.npy".format(key)),
                             value,
                             allow_pickle=True) for key, value in vars(vars(self)[peak]).items()];

        elif loaddir != None:
            for peak in os.listdir(loaddir):
                setattr(self, peak, _FitDict(center=np.load(os.path.join(loaddir, peak, "center.npy"), allow_pickle=True),
                                             sigma=np.load(os.path.join(loaddir, peak, "sigma.npy"), allow_pickle=True),
                                             amplitude=np.load(os.path.join(loaddir, peak, "amplitude.npy"), allow_pickle=True),
                                             intensity=np.load(os.path.join(loaddir, peak, "intensity.npy"), allow_pickle=True),
                                             best_fit=np.load(os.path.join(loaddir, peak, "best_fit.npy"), allow_pickle=True),
                                             shift=np.load(os.path.join(loaddir, peak, "shift.npy"), allow_pickle=True)                        
                                            ))
    def heatmap(self, peaks, quantity="intensity", cmap="viridis"):
        """
        Returns heatmap-style graphs of the magnitude of fitted parameters. 
        NB: if multiple peaks are to be displayed, the same colorbar scale (same min/ max) values will be used for all heatmaps for cross-comparison purposes. 

            Parameters:
                peaks (str or list): the label(s) of peaks to show. If single string, returns single heatmap, if list of peaks, returns 1xlen(list) heatmaps
                quantity (str, default="intensity"): the fit result value to display as a heatmap. Allowed values: "center", "sigma", "amplitude", "intensity" 
                cmap (default="viridis"): the matplotlib color map to use.
        """

        if type(peaks) == str:
            f, ax = plt.subplots()
            ax.imshow(vars(vars(self)[peaks])[quantity])

        elif type(peaks) == list:
            f, (axes) = plt.subplots(1, len(peaks))
            f.suptitle(quantity)
            vmax = np.nanmax([np.max(vars(vars(self)[peak])[quantity]) for peak in peaks])
            vmin = np.nanmin([np.min(vars(vars(self)[peak])[quantity]) for peak in peaks])
            for n_peak, peak in enumerate(peaks):
                im = axes[n_peak].imshow(vars(vars(self)[peak])[quantity], 
                                    vmax=vmax,
                                    vmin=vmin,
                                    cmap=cmap)
                axes[n_peak].set_title(peak)
        f.subplots_adjust(right=0.8)   
        cbar_ax = f.add_axes([0.85, 0.35, 0.01, 0.3])
        cbar = f.colorbar(im, cax=cbar_ax)
        
class FitMap(object):
    """
    A class to pre-process and fit map arrays of data
    ...
    Input arguments 
    ----------
    map_array: numpy array
        a 3d numpy array (x, y, shift) of (pre-processed) spectral data to fit (pre-processing may include background removal, etc)

    shift_axis: numpy array
        a 1d numpy array of the shift values associated with the map_array. Must have the same size as second axis dimension of map_array

    Attributes
    ----------
    map: numpy array
        input map_array data

    shift: numpy array
        input shift_axis data

    x_extent: int
        number of x positions at which spectra were collected (0 axis shape of map_array)

    y_extent: int
        number of y positions at which spectra were collected (1 axis shape of map_array)

    peaks_to_fit: dict
        empty when instantiated (see add_peak_to_fit method)

    Methods
    ----------
    add_peak_to_fit(label, peak_center):
        appends peak to be fit by model with user-defined label at user-defined shift/ wavenumber

    make_rois(overlap_range=50):
        splits input data into regions of interest (ROIs) to speed up fitting process
        returns ROI-related attributes

    fit_model:
        fits peaks defined in peaks_to_fit dict
        returns FitResults object
    """

    def __init__(self, map_array, shift_axis):
        self.map = map_array
        self.shift = shift_axis
        self.x_extent = self.map.shape[0]
        self.y_extent = self.map.shape[1]

        self.peaks_to_fit = {}

    def add_peak_to_fit(self, label, peak_center):
        """
        Appends entry to peaks_to_fit dictionary. 

            Parameters:
                label (str): User defined label to aid results identification (eg: "G")
                peak_center (float): The center value of the peak, cm-1 (eg: 1580)

            Returns:
                None
        """
        self.peaks_to_fit.update([(label, peak_center)])

    def make_rois(self, overlap_range=50):
        """
        Optional: can be called by user if overlap_range other than 50 is required.
        Splits the input data into regions of interest (ROI) to speed up peak fitting. 
        Peaks are sorted into groups (ROIs) such that all peaks have center values within +/- overlap_range of each other. 

            Parameters:
                overlap_range (float, default=50): the maximum difference (cm-1) between centers of different peaks in each ROI

            Returns: 
                self.ROI_groups: dict of peak labels listed by ROI
                self.ROI_shifts: dict of indices of start and end points of each ROI (indices corresponding to input shift_axis)
                self.ROI: dict with...
                    keys: ROI labels
                    values: dicts containing shift and map data corresponding to each ROI
                            NB: map_data is normalised using snv to expedite fitting and produce consistent values across each map
        """

        self._overlap_range = overlap_range
        self.ROI_groups = {}
        for names, wns in self.peaks_to_fit.items():
            if len(self.ROI_groups) == 0:
                self.ROI_groups.update([(1, [names])])
            else:
                for groups, members in self.ROI_groups.items():
                    for item in members:
                        if abs(wns-self.peaks_to_fit[item]) < overlap_range:
                            members.append(names)
                            new_line_needed = False
                            break
                        else:
                            new_line_needed = True
                if new_line_needed == True:
                    self.ROI_groups.update([(max(self.ROI_groups.keys())+1, [names])])

            self.ROI_shifts = {}

            for group_id, group in self.ROI_groups.items():
                self.ROI_shifts.update([(group_id, [np.min([self.peaks_to_fit[key] for key in group])-self._overlap_range,
                                                    np.max([self.peaks_to_fit[key] for key in group])+self._overlap_range])])

            self.ROI = {}
            for group_id, shift_range in self.ROI_shifts.items():
                min_idx = np.argmin(abs(shift-shift_range[0]))
                max_idx = np.argmin(abs(shift-shift_range[1]))
                self.ROI.update([(group_id, {"shift": shift[min_idx:max_idx],
                                             "map": sp.snv(self.map[:, :, min_idx:max_idx])})])

    def fit_model(self, center_tolerance=10, groups_to_fit="all", savedir=None):
        """
        Creates and fits lmfit models corresponding to identified ROIs. 
        Creates ROIs if not explicitely set by user. 

        Parameters:
            center_tolerance (float, default=10): maximum deviation of fitted center from user-defined center, cm-1
            groups_to_fit (list or "all", default="all"): select which ROI groups to fit by key from self.ROI (eg. groups_to_fit=[1] would fit only first ROI). Fits all ROIs by default.
            savedir (str, default=None): name of directory to save fitted results to (will be created if does not exist).

        Returns:
            self.ROI_fits (attr): an array of model results
            self.fit_results (attr, FitResults object): a class with easily accessible fitted results and methods for displaying data


        """

        if "ROI" not in vars(self).keys():
            self.make_rois()

        if groups_to_fit == "all":
            groups_to_fit = self.ROI.keys()

        self.ROI_fits = {}
        for roi_id in groups_to_fit:
            m = LinearModel()
            for key in self.ROI_groups[roi_id]:
                m+=LorentzianModel(prefix=key)

            p = m.make_params()

            center_tolerance = 10

            for name in p:
                if "center" in name:
                    p[name].value = peaks_to_fit[name[:-len("center")]]
                    p[name].min =  p[name].value-center_tolerance
                    p[name].max =  p[name].value+center_tolerance
                if "amplitude" in name:
                    ## Ensures all peaks have positive amplitude
                    p[name].min = 0     
                if "sigma" in name:
                    ## Sigma has non-zero minimum value to avoid very narrow peaks.
                    ## Since intensity=amplitude/2/pi/sigma, sigma ~ 0 returns very high intensity peaks
                    ## even where peak intensity is negligble. 
                    p[name].min = 1     
                    p[name].value = 2

            p["slope"].value = 0

            fit_temp = []
            print("Fitting ROI {}".format(roi_id))
            for x in notebook(range(self.ROI[roi_id]["map"].shape[0])):
                for y in range(self.ROI[roi_id]["map"].shape[1]):
                    fit_results = m.fit(x=self.ROI[roi_id]["shift"],
                                        data=self.ROI[roi_id]["map"][x, y, :],
                                        params=p)
                    fit_temp.append(fit_results)

            self.ROI_fits.update([(roi_id, np.array(fit_temp).reshape(self.ROI[roi_id]["map"].shape[0],
                                                                     self.ROI[roi_id]["map"].shape[1]))])  

        setattr(self, "fit_results", FitResults(self, savedir=savedir))
