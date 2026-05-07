# This module sets a BaseClass for objects with Prob data and coordinates.
# I could have tried the Xarr package, which provides arrays with labels
# with both numpy-like and pandas-like label referencing. 
# It looks rather well maintained, but there are still a lot of open issues and pull requests. 
# So I opted for something a bit more simple. 

from . import validators as valid
import numpy as np
from abc import ABC, abstractmethod
import h5py

class GridDimensionError(ValueError):
    """Exception raised when the data shape doesn't match the coordinate grids."""
    pass

class BaseBayesObject(ABC):
    '''
    Base class for objects that will contain probability data. 
    This class should not be really used as is in the science codes --
    it is mean to be inherited by specific classes for the science objects. 
    '''

    # This list allows the science classes to provide a specific
    # list of coordinate needed for each science bayes object. 
    # For example, ['Bpole', 'beta', 'phi'].
    # This is to make science functions for a given BayesObject 
    # (the LH, the posterior, etc) more clear. 
    # The subclasses MUST define it.
    
    IS_LOG = False # Default to Linear

    @property
    @abstractmethod
    def REQUIRED_COORDS(self):
        pass
    
    def __init__(self, prob, **coords):
        # Use the validator to make sure that the coords keywords are
        # List-like objects, convert to numpy, and make sure that they are numerical.
        self.prob = valid.convert_to_numpy_and_validate_numerical('prob', prob)
        self.coords = {k: valid.convert_to_numpy_and_validate_numerical(k, v) for k, v in coords.items()}
        self._validate_names()
        self._validate()    

    def _validate_names(self):
        """Checks if the keys in self.coords match REQUIRED_COORDS exactly."""
        provided_names = list(self.coords.keys())
        
        if provided_names != self.REQUIRED_COORDS:
            missing = set(self.REQUIRED_COORDS) - set(provided_names)
            extra = set(provided_names) - set(self.REQUIRED_COORDS)
            
            error_msg = f"Initialization error for {type(self).__name__}: "
            if missing: error_msg += f"Missing: {missing}. "
            if extra: error_msg += f"Unexpected: {extra}. "
            if not missing and not extra:
                error_msg += f"Incorrect coordinate order. Expected {self.REQUIRED_COORDS}, got {provided_names}."
            
            raise KeyError(error_msg)

    def _validate(self):
        # Universal validation: Do the lengths match the Prob data dimensions?
        
        # 1. Dimension count check
        if np.atleast_1d(self.prob).ndim != len(self.coords):
            raise GridDimensionError(
                f"Bayes Object Dimension mismatch: Prob is {self.prob.ndim}D, "
                f"but {len(self.coords)} coordinates were provided."
            )

        # 2. Individual coordinate validations
        for i, (name, arr) in enumerate(self.coords.items()):
            # Check for 0D or 1D:
            if arr.ndim > 1:
                raise GridDimensionError(
                    f"Bayes Object Coordinate '{name}' must be 0D or 1D. "
                    f"Received a {arr.ndim}D array."
                )            
            if len(np.atleast_1d(arr)) != np.atleast_1d(self.prob).shape[i]:
                raise GridDimensionError(f"Bayes object Probability data dimension {i} ({np.atleast_1d(self.prob).shape[i]} elements) does not match the lenght of the '{name}' coordinate array ({len(np.atleast_1d(arr))} elements)")
            
    def __getattr__(self, name):
        # This allows obj.coord_name to look inside the self.coords dict
        if name in self.coords:
            return self.coords[name]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
    
    def __getitem__(self, key):
        """
        Slicing of objects. Dimensions do not collapse automatically,
        because the number of dimension of the object is fixed by the REQUIRED_COORDS
        """
        # 1. Normalize the key to a tuple to handle single vs multi-dim indexing
        print(key)
        if not isinstance(key, tuple):
            key = (key,)
        print(key)
        # 2. Convert any integers in the key to slices to preserve dimensions
        # This turns obj[0] into obj[0:1]
        new_key = tuple(
            slice(k, k + 1) if isinstance(k, (int, np.integer)) else k for k in key
            )
        print(new_key)

        # 3. Slice the probability data
        new_prob = self.prob[new_key]

        # 4. Slice each coordinate array using the corresponding part of the key
        new_coords = {}
        for i, (name, arr) in enumerate(self.coords.items()):
            # Use the specific slice for this axis, or a full slice if not provided
            coord_key = new_key[i] if i < len(new_key) else slice(None)
            new_coords[name] = arr[coord_key]                
        
        return type(self)(new_prob, **new_coords)

    def __repr__(self):
        coord_info = ", ".join([f"{k}={v.shape}" for k, v in self.coords.items()])
        return f"{type(self).__name__}(prob={self.prob.shape}, {coord_info})"

    def _is_compatible(self, other):
        """Checks if another object has identical coordinates and shapes."""
        if not isinstance(other, BaseBayesObject):
            return False
        
        # 1. Check if coordinate names match
        if self.REQUIRED_COORDS != other.REQUIRED_COORDS:   
            return False
            
        # 2. Check if the coordinate arrays themselves are identical
        for name in self.REQUIRED_COORDS:
            if not np.array_equal(self.coords[name], other.coords[name]):
                return False
                
        return True

    def __add__(self, other):
        # Case 1: Adding another Bayes Object
        if isinstance(other, BaseBayesObject):
            if not self._is_compatible(other):
                raise ValueError("Objects are incompatible: Coordinates must match exactly to add.")
            # Create a new instance of the same class (e.g., Mar1D + Mar1D = Mar1D)
            return type(self)(self.prob + other.prob, **self.coords)

        # Case 2: Adding a scalar
        if isinstance(other, (int, float, np.number)):
            return type(self)(self.prob + other, **self.coords)
        
        return NotImplemented
    
    # Allow scalar + obj
    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        # Case 1: Adding another Bayes Object
        if isinstance(other, BaseBayesObject):
            if not self._is_compatible(other):
                raise ValueError("Objects are incompatible: Coordinates must match exactly to multiply.")    
            return type(self)(self.prob * other.prob, **self.coords)
        
        # Case 2: Multiplying by a scalar
        if isinstance(other, (int, float, np.number)):
            return type(self)(self.prob * other, **self.coords)
        
        return NotImplemented
    
    # Allow scalar * obj
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __sub__(self, other):
        if isinstance(other, BaseBayesObject):
            if not self._is_compatible(other):
                raise ValueError("Incompatible coordinates for subtraction.")
            return type(self)(self.prob - other.prob, **self.coords)
        
        if isinstance(other, (int, float, np.number)):
            return type(self)(self.prob - other, **self.coords)
        
        return NotImplemented

    def __rsub__(self, other):
        # Handles: scalar - obj
        if isinstance(other, (int, float, np.number)):
            return type(self)(other - self.prob, **self.coords)
        return NotImplemented
    
    def __truediv__(self, other):
        if isinstance(other, BaseBayesObject):
            if not self._is_compatible(other):
                raise ValueError("Incompatible coordinates for division.")
            return type(self)(self.prob / other.prob, **self.coords)
        
        if isinstance(other, (int, float, np.number)):
            return type(self)(self.prob / other, **self.coords)
        
        return NotImplemented

    def __rtruediv__(self, other):
        # Handles: scalar / obj
        if isinstance(other, (int, float, np.number)):
            return type(self)(other / self.prob, **self.coords)
        return NotImplemented
    
    def _log_sum_exp(self, axis):
        """Robust Log-Sum-Exp implementation."""
        norm = np.nanmax(self.prob)
        # Handle the case where all values are -inf
        if not np.isfinite(norm):
            # If axis is None, return a simple scalar -inf
            if axis is None:
                return -np.inf
            # Otherwise, return an array of -inf with reduced dimensions
            return np.full(np.delete(self.prob.shape, axis), -np.inf) 
                   
        p_array = np.exp(self.prob - norm)
        return np.log(np.nansum(p_array, axis=axis)) + norm        

    def _log_integrate_uniform(self, axis=None):
        """
        Integrates log(probability) over specified axes.
        Standard: Coordinate values represent the LEFT EDGES of the bins.
        Total Volume = Product of (Number of bins * bin_width) for each axis.
        """
        # 1. Normalize axes to a list of indices
        if axis is None:
            axis_indices = list(range(len(self.REQUIRED_COORDS)))
        elif isinstance(axis, (int, np.integer)):
            axis_indices = [axis]
        else:
            axis_indices = list(axis)

        # 2. Basic Log-Sum-Exp (The sum of the 'heights')
        total_ln_prob = self._log_sum_exp(axis=axis)

        # 3. Add the log-volume (The 'widths')
        ln_total_volume = 0
        for idx in axis_indices:
            coord_name = self.REQUIRED_COORDS[idx]
            vals = self.coords[coord_name]
            
            if len(vals) > 1:
                # Width of one bin
                dx = vals[1] - vals[0]
                # Total width for this axis = (Number of Left Edges) * dx
                ln_total_volume += np.log(len(vals) * dx)
            else:
                # If it's a single point, we treat it as a delta function (width=1)
                ln_total_volume += 0

        return total_ln_prob + ln_total_volume
     

    def writef(self, f):
        """
        Helper function to create datasets in the passed h5 file object.
        Works for any number of dimensions and coordinate names.
        """
        # Save the main probability data
        f.create_dataset('prob', data=self.prob)
        
        # Dynamically save all coordinates found in self.coords
        for name, arr in self.coords.items():
            f.create_dataset(name, data=arr)
            
        # Optional: Save the class name so you know what object to 
        # recreate when reading the file back
        f.attrs['class_name'] = type(self).__name__

    def write(self, fname):
        """
        Standard entry point to write the object to an HDF5 file.
        """
        import h5py
        with h5py.File(fname, 'w') as f:
            self.writef(f)
            ## Here, could add a **metadata to the call, and do
            #for key, value in metadata.items():
            #    f.attrs[key] = value
            # if I need to be able to add some addition stuff on the fly

    def marginalize(self, axis_name):
        """
        Marginalizes over a specific coordinate by name.
        Returns the necessary elements to create a DIFFERENT class (e.g., Grid3D -> Grid2D).
        """
        if axis_name not in self.REQUIRED_COORDS:
            raise ValueError(f"Cannot marginalize over '{axis_name}'. Not in coords.")
        
        axis_idx = self.REQUIRED_COORDS.index(axis_name)
        coord_vals = self.coords[axis_name]   

        # 1. Perform the math based on representation
        if self.IS_LOG:
            new_prob = self._log_sum_exp(axis=axis_idx)
        else:
            new_prob = np.sum(self.prob, axis=axis_idx)
            
        # 2. Prepare new coordinates (drop the marginalized one)
        new_coords = self.coords.copy()
        new_coords.pop(axis_name)    

        # Note: You'll likely need a 'Factory' or a specific target class here
        # because the REQUIRED_COORDS of the current class won't match 
        # the new 2D data.
        return new_prob, new_coords