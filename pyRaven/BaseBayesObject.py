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

    Standard Convention: Coordinates in this package are treated as Left Edges.
    * For Physical Ranges: If you want to cover $0$ to $10$ Gauss with $10$ bins, 
      your coordinate array should be `np.linspace(0, 9, 10)`.
    * For Periodic Angles: To avoid double-counting, do not include the endpoint. 
      Use `np.arange(0, 360, 5)` (which stops at $355$) rather than `np.linspace(0, 360, 73)`.

    '''

    #-------------------------------------
    # 1. Properties & Initialization
    #-------------------------------------

    # Properties used by the sum and integration to determine
    # whether the probability is stores in P or ln(P)    
    PROB_IS_LOG = False # Default to Linear

    @property
    @abstractmethod
    def REQUIRED_COORDS(self):
        '''This list allows the science classes to provide a specific
        list of coordinate needed for each science bayes object. 
        For example, ['Bpole', 'beta', 'phi'].
        This is to make science functions for a given BayesObject 
        (the LH, the posterior, etc) more clear. 
        The subclasses MUST define it.'''
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

    @classmethod
    def empty(cls, **kwargs):
        """
        Universal factory method in the base class.
        Automatically constructs a zeroed array matching the subclass REQUIRED_COORDS.
        """
        shape = []
        for name in cls.REQUIRED_COORDS:
            if name not in kwargs:
                raise ValueError(f"Missing required coordinate array: '{name}'")
            
            # Safely measure length whether given an array, list, or tuple
            # or other unallowed types -- the __init__ will validate
            array_size = np.atleast_1d(kwargs[name]).size
            shape.append(array_size)
            
        # Allocate the core probability array shell
        data = np.zeros(shape)
        
        # Instantiate the subclass, passing the new array as the 'prob' keyword argument
        return cls(prob=data, **kwargs)

    #-------------------------------------
    # 2. Magic Methods & Overloads
    #-------------------------------------

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
        
        # 5. Create a raw instance of the current class without invoking __init__
        new_obj = type(self).__new__(type(self))
        
        # 6. Manually assign the freshly sliced core arrays
        new_obj.prob = new_prob
        new_obj.coords = new_coords

        # 7. Automatically copy over all remaining science metadata and properties
        core_attrs = {'prob', 'coords'}
        for attr, value in self.__dict__.items():
            if attr not in core_attrs:
                setattr(new_obj, attr, value)
                
        return new_obj

    def __repr__(self):
        coord_info = ", ".join([f"{k}={v.shape}" for k, v in self.coords.items()])
        return f"{type(self).__name__}(prob={self.prob.shape}, {coord_info})"

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
    
    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, BaseBayesObject):
            if not self._is_compatible(other):
                raise ValueError("Objects are incompatible: Coordinates must match exactly to subtract.")
            return type(self)(self.prob - other.prob, **self.coords)
        
        if isinstance(other, (int, float, np.number)):
            return type(self)(self.prob - other, **self.coords)
        
        return NotImplemented
    
    def __rsub__(self, other):
        # Handles: scalar - obj
        if isinstance(other, (int, float, np.number)):
            return type(self)(other - self.prob, **self.coords)
        return NotImplemented
    
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
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if isinstance(other, BaseBayesObject):
            if not self._is_compatible(other):
                raise ValueError("Objects are incompatible: Coordinates must match exactly to divide.")
            return type(self)(self.prob / other.prob, **self.coords)
        
        if isinstance(other, (int, float, np.number)):
            return type(self)(self.prob / other, **self.coords)
        
        return NotImplemented

    def __rtruediv__(self, other):
        # Handles: scalar / obj
        if isinstance(other, (int, float, np.number)):
            return type(self)(other / self.prob, **self.coords)
        return NotImplemented

    #-------------------------------------
    # 3. User-Facing Public API
    #-------------------------------------

    def marginalize(self, axis=None):
        """
        Marginalizes over a specific coordinate by name or by index.
        Returns the necessary elements to create a DIFFERENT class (e.g., Grid3D -> Grid2D).
        """
        
        # 1. Validate the axis input
        axis_idx = self._get_validated_axes_indexes(axis)  

        # 2. Run the integral
        if self.PROB_IS_LOG:
            new_prob = self._log_integrate_uniform(axis_idx)
        else:
            dv = np.exp(self._get_axis_log_dv(axis_idx))
            new_prob = np.sum(self.prob, axis=tuple(axis_idx)) * dv
            
        # 3. Handle structural metadata metadata
        new_coords = self.coords.copy()
        for idx in axis_idx:
            new_coords.pop(self.REQUIRED_COORDS[idx])

        # Note: You'll likely need a 'Factory' or a specific target class here
        # because the REQUIRED_COORDS of the current class won't match 
        # the new 2D data.
        return new_prob, new_coords
    
    def write(self, fname):
        """
        Standard entry point to write the object to an HDF5 file.
        """
        import h5py
        with h5py.File(fname, 'w') as f:
            self._writef(f)
            ## Here, could add a **metadata to the call, and do
            #for key, value in metadata.items():
            #    f.attrs[key] = value
            # if I need to be able to add some addition stuff on the fly

    @classmethod
    def read(cls, filename: str, **kwargs):
        """
        Universal HDF5 reader factory in the base class.
        Dynamically extracts required datasets and file attributes.
        """
        with h5py.File(filename, 'r') as f:
            # SAFETY CHECK: Verify that the file matches the class calling it
            file_class_type = f.attrs.get('class_name')
            # Look at an attribute on the class itself, like cls.__name__ or a custom CLASS_ID
            expected_type = getattr(cls, 'CLASS_ID', cls.__name__) 
            
            if file_class_type and file_class_type != expected_type:
                raise TypeError(
                    f"Mismatched file type! File '{filename}' contains a '{file_class_type}' object, "
                    f"but you are trying to read it using the '{expected_type}' class."
                )            
            
            # 1. Dynamically read the core probability matrix and coordinates
            loaded_args = {
                'prob': f['prob'][:]
            }
            for coord_name in cls.REQUIRED_COORDS:
                loaded_args[coord_name] = f[coord_name][:]
            
            # 2. Automatically harvest every HDF5 attribute into a metadata dict
            metadata = dict(f.attrs)
            
            # 3. Merge them together, prioritizing any runtime overrides passed in via **kwargs
            constructor_inputs = {**loaded_args, **metadata, **kwargs}
            
            # 4. Remove internal identifiers like 'class_type' so they don't break __init__
            constructor_inputs.pop('class_name', None)
            
            # 5. Dynamically instantiate the subclass
            return cls(**constructor_inputs)

    #-------------------------------------
    # 4. Input Processing / Validation Helpers
    #-------------------------------------

    def _get_validated_axes_indexes(self, axis=None):
        """
        Standardize any input (None, int, str, or sequence of ints/strs) for listing axes to use in sum or marginalize
        into a list of integer axis indices.
        """
        # Handle the global option
        if axis is None:
            return list(range(len(self.REQUIRED_COORDS)))
        
        # Coerce single items (strings or integers) into an iterable list
        # If it is a basic scalar type, wrap it in a list.
        if isinstance(axis, (str, int, np.integer)):
            axis_list = [axis]
        # If it's already a sequence/iterable (excluding strings), convert to a list
        elif isinstance(axis, (list, tuple, set, np.ndarray)):
            axis_list = list(axis)
            axis_list = list(axis)
        else:
            raise TypeError(f"Invalid axis identifier type: {type(axis)}")
            
        axis_indices = []
        for item in axis_list:
            if isinstance(item, str):
                if item not in self.REQUIRED_COORDS:
                    raise ValueError(f"Coordinate '{item}' is not valid for this object.")
                axis_indices.append(self.REQUIRED_COORDS.index(item))
            elif isinstance(item, (int, np.integer)):
                if item < 0 or item >= len(self.REQUIRED_COORDS):
                    raise IndexError(f"Axis index {item} is out of bounds.")
                axis_indices.append(int(item))
            else:
                raise TypeError(f"Invalid axis identifier type: {type(item)}")
                
        # Return sorted unique indices (collapsing dimensions in order prevents shape bugs)
        return sorted(list(set(axis_indices)))  

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

    #-------------------------------------
    # 5. IO helpers
    #-------------------------------------

    def _writef(self, f):
        """
        Helper function to create datasets in the passed h5 file object.
        Works for any number of dimensions and coordinate names.
        """
        # Save the main probability data
        f.create_dataset('prob', data=self.prob)
        
        # Dynamically save all coordinates found in self.coords
        for name, arr in self.coords.items():
            f.create_dataset(name, data=arr)
            
        # Automatically record class name and ALL extra science metadata
        f.attrs['class_name'] = type(self).__name__

        core_attrs = {'prob', 'coords'}
        for attr, value in self.__dict__.items():
            if attr not in core_attrs and np.isscalar(value):
                f.attrs[attr] = value        

    #-------------------------------------
    # 6. Core Mathematical Backends
    #-------------------------------------

    def _log_sum_exp(self, validated_axis_indices):
        """Robust Log-Sum-Exp implementation."""
        norm = np.nanmax(self.prob)
        # Handle the case where all values are -inf
        if not np.isfinite(norm):
            # If we integrated over ALL available dimensions, return a scalar
            if len(validated_axis_indices) == len(self.prob.shape):
                return -np.inf
            # Otherwise, calculate the exact shape of the remaining uncollapsed dimensions
            remaining_shape = tuple(dim for i, dim in enumerate(self.prob.shape) if i not in validated_axis_indices)
            return np.full(remaining_shape, -np.inf)
                   
        p_array = np.exp(self.prob - norm)
        return np.log(np.nansum(p_array, axis=tuple(validated_axis_indices))) + norm        

    def _get_axis_log_dv(self, validated_axis_indices):
        '''Helper function to get the log(dV) of the grid for integration over the given axis'''

        # Assumes that the axis have already been validates with _standardize_axes?        
        ln_total_dx_volume = 0
        
        for idx in validated_axis_indices:
            coord_name = self.REQUIRED_COORDS[idx]
            vals = self.coords[coord_name]
            
            if len(vals) > 1:
                # Width of one bin
                dx = vals[1] - vals[0]
                # Total width for this axis = (Number of Left Edges) * dx
                ln_total_dx_volume += np.log(dx)
            else:
                # If it's a single point, we treat it as a delta function (width=1)
                ln_total_dx_volume += 0

        return ln_total_dx_volume

    def _log_integrate_uniform(self, validated_axis_indices):
        """
        Integrates log(probability) over specified axes.
        Standard: Coordinate values represent the LEFT EDGES of the bins.
        Total Volume = Product of (Number of bins * bin_width) for each axis.
        """

        # 2. Basic Log-Sum-Exp (The sum of the 'heights')
        total_ln_prob = self._log_sum_exp(validated_axis_indices)

        # 3. The log-volume of one grid point (The 'widths')
        ln_total_dx_volume = self._get_axis_log_dv(validated_axis_indices)

        return total_ln_prob + ln_total_dx_volume
    
