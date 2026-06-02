import pytest
import numpy as np
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from pyRaven import validators as v

class Test_validate_array_like_container:
    def test_validate_numeric_success(self):
        """Test that valid inputs do NOT raise an error."""
        # No error should be raised for these
        v.validate_array_like_container("test", [1, 2, 3])
        v.validate_array_like_container("test", (1, 2, 3))
        v.validate_array_like_container("test", np.array([1, 2]))
        v.validate_array_like_container("test", np.zeros((2,3,4)))
        v.validate_array_like_container("test", 10.5)
        v.validate_array_like_container("test", 100)
        v.validate_array_like_container("test", np.array(["a", "b"]))
        v.validate_array_like_container("test", np.array(100))
        v.validate_array_like_container("test", np.array(10.5))

    @pytest.mark.parametrize("bad_val", [
        "string", 
        {"a": 1}, 
        None, 
        ])
    def test_validate_numeric_failure(self, bad_val):
        """Test that invalid inputs raise a TypeError."""
        with pytest.raises(TypeError) as excinfo:
            v.validate_array_like_container("my_var", bad_val)
        # Check if the error message correctly names the variable
        assert "my_var" in str(excinfo.value)
        assert "must be a list, numpy array, or float" in str(excinfo.value)

class Test_convert_container_to_numpy_and_validate:

    def test_check_call_validate_array_like_container(self):
        # Only need to test the call to validate_array_like_container
        # so I can do that with passing one bad type (the others are already tested)
        with pytest.raises(TypeError) as excinfo:
            v.convert_to_numpy_and_validate_numerical("my_var", "string")
        # Check if the error message correctly names the variable
        assert "my_var" in str(excinfo.value)
        assert "must be a list, numpy array, or float" in str(excinfo.value)

    @pytest.mark.parametrize("bad_val", [
        ['a', 'b'], 
        ('a', 'b'),
        [{'a':2}]*2,
        np.array(['a', 'b']),
        np.array([{'a':2}])
        ])
    def test_validate_numeric_failure(self, bad_val):
        """Test that invalid inputs raise a TypeError."""
        with pytest.raises(TypeError) as excinfo:
            result = v.convert_to_numpy_and_validate_numerical("my_var", bad_val)
        # Check if the error message correctly names the variable
        assert "my_var" in str(excinfo.value)
        assert "must contain numeric data" in str(excinfo.value)

    def test_validate_sucess(self):
        """Test that valid inputs do NOT raise an error."""
        res = v.convert_to_numpy_and_validate_numerical("test", [1, 2, 3])
        # Assert that it was converted to a numpy array internally
        assert isinstance(res, np.ndarray)
        assert res.shape == (3,)
        np.testing.assert_array_equal(res, np.array([1,2,3]))        

        res = v.convert_to_numpy_and_validate_numerical("test", np.array([1, 2]))
        assert isinstance(res, np.ndarray)
        assert res.shape == (2,)
        np.testing.assert_array_equal(res, np.array([1,2]))            
        
        res = v.convert_to_numpy_and_validate_numerical("test", np.zeros((2,3,4)))
        assert isinstance(res, np.ndarray)
        assert res.shape == (2,3,4)        

        res = v.convert_to_numpy_and_validate_numerical("test", 10.5)
        assert isinstance(res, np.ndarray)
        assert res.shape == (1,)
        np.testing.assert_array_equal(res, np.array([10.5]))            
       
        res = v.convert_to_numpy_and_validate_numerical("test", [100])
        assert isinstance(res, np.ndarray)
        assert res.shape == (1,)
        np.testing.assert_array_equal(res, np.array([100]))            

class TestValidateMatplotlibAxes:
    """
    Unit test suite covering successful conversions, structural handling, 
    and defensive exceptions for the validate_matplotlib_axes utility.
    """

    @pytest.fixture(autouse=True)
    def setup_canvas(self):
        """
        Fixture that automatically initializes and tears down a clean, 
        headless matplotlib canvas context for every test loop.
        """
        # Close any lingering global figures before starting
        plt.close('all')
        yield
        # Clear out test figure memory immediately after the test runs
        plt.close('all')

    def test_accepts_single_raw_axis_object(self):
        """Verifies that a standalone raw Axes instance is safely wrapped into a 1D object array."""
        fig, raw_ax = plt.subplots()
        
        result = v.validate_matplotlib_axes(raw_ax)
        
        assert isinstance(result, np.ndarray)
        assert result.dtype == object
        assert result.size == 1
        assert result[0] is raw_ax

    def test_accepts_standard_python_list_of_axes(self):
        """Verifies that a standard Python 1D list of multiple axes is normalized into a flat array."""
        fig, (ax1, ax2) = plt.subplots(1, 2)
        axes_list = [ax1, ax2]
        
        result = v.validate_matplotlib_axes(axes_list)
        
        assert isinstance(result, np.ndarray)
        assert result.size == 2
        assert result[0] is ax1
        assert result[1] is ax2

    def test_flattens_multidimensional_numpy_grids(self):
        """Verifies that a 2D grid matrix of subplots is successfully unrolled into a flat 1D sequence."""
        fig, ax_grid = plt.subplots(2, 2)  # Creates a 2x2 shape numpy array
        
        result = v.validate_matplotlib_axes(ax_grid)
        
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2  # Must unroll multi-dimensional bounds
        assert result.size == 4
        assert result.shape == (2,2)
        # Assert the ordering matches sequential flat matrix mapping (.flat)
        assert result.flat[0] is ax_grid[0, 0]
        assert result.flat[3] is ax_grid[1, 1]

    def test_raises_type_error_for_none_input(self):
        """Verifies passing None triggers a clear TypeError guard rails."""
        with pytest.raises(TypeError, match="Expected axes input, but received None"):
            v.validate_matplotlib_axes(None)

    def test_raises_type_error_for_non_matplotlib_types(self):
        """Verifies passing arbitrary strings or integers fails early."""
        with pytest.raises(TypeError, match="Input must be a matplotlib Axes instance"):
            v.validate_matplotlib_axes('bad_input')

    def test_raises_type_error_for_mixed_corrupted_lists(self):
        """Verifies that if a list contains axes but is corrupted by a rogue object, it fails explicitly."""
        fig, raw_ax = plt.subplots()
        corrupted_list = [raw_ax, "NotAnAxisObject", raw_ax]
        
        # Match pattern checks against our descriptive nested loop error message
        with pytest.raises(TypeError, match="Element at flat index 1 inside the axes structure is not a valid"):
            v.validate_matplotlib_axes(corrupted_list)
            
    def test_subclass_compatibility(self):
        """Verifies that specialized polar/3D projections pass because they inherit from Axes."""
        fig = plt.figure()
        # Instantiate a specialized projection subclass axis
        polar_ax = fig.add_subplot(111, projection='polar')
        
        result = v.validate_matplotlib_axes(polar_ax)
        
        assert isinstance(result[0], Axes)
        assert result[0] is polar_ax