import numpy as np
from matplotlib.axes import Axes

def validate_array_like_container(name, value):
    """Utility to ensure a variable is a list, ndarray, or scalar."""
    allowed_types = (list, tuple, np.ndarray, np.number, float, int)
    if not isinstance(value, allowed_types):
        raise TypeError(
            f"{name} must be a list, numpy array, or float. "
            f"Got {type(value).__name__} instead."
        )
    
def convert_to_numpy_and_validate_numerical(name, value):
            
    validate_array_like_container(name, value)

    arr_value = np.asanyarray(value)
    
    # Now check the types of the RESULTING array's elements
    # (Checking .dtype is often better than checking the container type)
    if not np.issubdtype(arr_value.dtype, np.number):
        raise TypeError(f"{name} must contain numeric data, got {arr_value.dtype}")
    
    return np.atleast_1d(arr_value)

def validate_matplotlib_axes(axes) -> np.ndarray:
    """
    Validates that the input is a matplotlib Axes object, a list of Axes, 
    or a NumPy array of Axes. 

    Converts the input into a flattened NumPy array of objects for easy, 
    uniform iteration, or raises a clear TypeError if validation fails.

    Parameters
    ----------
    axes : matplotlib.axes.Axes, list, or np.ndarray
        The axes structure passed by the user.

    Returns
    -------
    np.ndarray
        A ND NumPy array containing the matplotlib Axes objects.

    Raises
    ------
    TypeError
        If the input or any sub-element is not an instance of matplotlib.axes.Axes.
    """
    if axes is None:
        raise TypeError("Validation Error: Expected axes input, but received None.")

    # 1. Normalize the input into a standard NumPy array
    if isinstance(axes, Axes):
        # Wrap standalone single axis into an array wrapper
        axes_array = np.array([axes], dtype=object)
    elif isinstance(axes, (list, tuple)):
        axes_array = np.array(axes, dtype=object)
    elif isinstance(axes, np.ndarray):
        axes_array = axes.astype(object)
    else:
        raise TypeError(
            f"Validation Error: Input must be a matplotlib Axes instance, "
            f"a list of Axes, or a NumPy array of Axes. Received type: {type(axes)}"
        )

    # 2. Iterate and verify every single nested element is a true Axes instance
    # using .flat to handle 1D lists, 2D subplot grids, or N-D layouts transparently
    for idx, item in enumerate(axes_array.flat):
        if not isinstance(item, Axes):
            raise TypeError(
                f"Validation Error: Element at flat index {idx} inside the axes "
                f"structure is not a valid matplotlib Axes object. "
                f"Received type: {type(item).__name__}"
            )

    return axes_array