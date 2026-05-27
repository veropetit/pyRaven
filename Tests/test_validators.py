import pytest
import numpy as np

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
