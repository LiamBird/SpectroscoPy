import numpy as np
import os

class MapSelect(object):
    """
    Interactive widget for selecting positions in Raman maps with relevant signal/ high noise to keep/ exclude from future analysis.
    ...
    
    Parameters:
    ----------
    fname (str): 
        EITHER previously created MapSelect file OR name for new MapSelect file
        If previously created MapSelect file, the previously-defined Boolean values are re-loaded into the grid
    xy_extent (tuple):
        the (x, y) size of the grid to generate for selecting positions
        not required if re-loading previous MapSelect file because (x, y) size is already defined
        
    Attributes:
    ----------
    grid (widget): 
        grid of checkboxes at each (x, y) location
        
    Methods:
    ----------
    get_positions():
        Returns Boolean array of True/ False values for the checkboxes selected by the user
        
    save(fname=None):
        If fname is None, uses the fname value entered when class instantiated
        Saves Boolean array of True/ False values to fname.npy   
    
    """
    def __init__(self, fname, xy_extent=None):
        import ipywidgets as wg
        
        self._fname = fname
        
        if fname!=None and os.path.isfile(fname):
            reload = np.load(fname, allow_pickle=True)
            self._x_extent = reload.shape[0]
            self._y_extent = reload.shape[1]
            checkbox_grid = {}
            for x in range(self._x_extent):
                for y in range(self._y_extent):
                    if np.array(reload, dtype=bool)[x, y] == False:
                        check_value = False
                    elif np.array(reload, dtype=bool)[x, y] == True:
                        check_value = True
                    checkbox_grid.update([(",".join((str(x), str(y))),
                                           wg.Checkbox(indent=False,
                                                      value=check_value,
                                                      layout=wg.Layout(flex="1 1 0%", width="auto")
                                                      ))])
            
        else:
            self._x_extent = xy_extent[0]
            self._y_extent = xy_extent[1]
            checkbox_grid = dict([(",".join((str(x), str(y))),
                                    wg.Checkbox(indent=False, layout=wg.Layout(flex='1 1 0%', width='auto')))\
                                   for x in range(self._x_extent) for y in range(self._y_extent)])
            
        x_labels = wg.VBox([wg.Label(value=" ")]+[wg.Label(value=str(x)) for x in range(self._x_extent)])
        x_list = [x_labels]
        for x in range(self._x_extent):
            x_list.append(wg.VBox([wg.Label(value=str(x))]+\
                                  [values for keys, values in checkbox_grid.items() if keys.split(",")[0]==str(x)]))
        self.grid = wg.HBox([x for x in x_list])
        
    def get_positions(self):
        return np.array([[child.value for child in self.grid.children[x].children[1:]] for x in range(1, self._x_extent+1)])
        
    def save(self, fname=None):
        if fname == None:
            fname = self._fname
            
        np.save(fname, self.get_positions(), allow_pickle=True)
            
        
