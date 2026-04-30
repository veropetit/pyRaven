import numpy as np

def validate_array_like_container(name, value):
    """Utility to ensure a variable is a list, ndarray, or scalar."""
    allowed_types = (list, np.ndarray, np.number, float, int)
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
    
    return arr_value