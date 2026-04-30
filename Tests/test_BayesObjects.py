import pytest
import numpy as np

from pyRaven import BayesObjects as bo

NON_NUMERIC_AND_NONE_INPUTS = ["a_string", {"a": 1}, None]

class Test_lnP_odds:

    @pytest.mark.parametrize("bad_data, expected_exception", [
        ([1, 2, 3], TypeError),                      # Not a numpy array
        (np.zeros((5, 3, 4)), bo.GridDimensionError),   # 3D instead of 4D
        (np.zeros((5, 3, 4, 10)), bo.GridDimensionError) # Wrong size in 4th dimension
    ])
    def test_init_invalid_data_input(self, bad_data, expected_exception):
        # Setup dummy grid arrays
        beta = np.linspace(0, 10, 5)   # Length 5
        bp   = np.linspace(0, 100, 3)  # Length 3
        phi  = np.linspace(0, 360, 4)  # Length 4
        incl = np.linspace(0, 90, 2)   # Length 2 
        
        # Verify that GridDimensionError is raised
        with pytest.raises(expected_exception):
            bo.lnP_odds(bad_data, beta, bp, phi, incl, "obs001", 1.2)

    @pytest.mark.parametrize("bad_type", NON_NUMERIC_AND_NONE_INPUTS) 
    def test_coordinate_type_validation(self, bad_type):
        # Valid dummy data
        data = np.zeros((1, 1, 1, 1))
        valid_arr = [1.0]
        
        # Test 1: Passing a string instead of a list/array/float
        with pytest.raises(TypeError) as excinfo:
            bo.lnP_odds(data, bad_type, valid_arr, valid_arr, valid_arr, "ID", 0.0)
        assert "must be a list, numpy array, or float" in str(excinfo.value)

    def test_coordinate_numeric_scalar(self):
        """Verify that passing a single float/int works without crashing."""
        data = np.zeros((1, 1, 1, 1))
        # All coords are single floats
        try:
            model=bo.lnP_odds(data, 1.0, 2500.0, 45.5, 90, "obs001", 1.2)
            assert model.phi_arr == 45.5
        except (TypeError, ValueError) as e:
            pytest.fail(f"Initialization with floats failed unexpectedly: {e}")


class Test_mar1D:

    @pytest.mark.parametrize("bad_type", NON_NUMERIC_AND_NONE_INPUTS)
    def test_init_invalid_types(self, bad_type):
        """Verify the coordinate type"""
        # Valid dummy data
        valid_arr = [1.0]
        # Test 1: Pass an bad input for "x"
        with pytest.raises(TypeError) as excinfo:
            bo.mar1D(bad_type, valid_arr)
        assert "must be a list, numpy array, or float" in str(excinfo.value)

        # Test 2: Pass a bad input for "mar"
        with pytest.raises(TypeError):
            bo.mar1D(valid_arr, bad_type)
        assert "must be a list, numpy array, or float" in str(excinfo.value)

    def test_init_not_1D(self):
        good = np.zeros((3,3))
        bad = np.zeros(4)
        # Verify that GridDimensionError is raised
        with pytest.raises(bo.GridDimensionError) as excinfo:
            bo.mar1D(good, bad )
        assert "a mar1D array must be 0D or 1D" in str(excinfo.value)
        with pytest.raises(bo.GridDimensionError) as excinfo:
            bo.mar1D(bad, good )
        assert "a mar1D array must be 0D or 1D" in str(excinfo.value)

    def test_init_mismatched_sizes(self):
        # setup dummy x
        good = np.zeros(3)
        bad = np.zeros(4)
        # Verify that GridDimensionError is raised
        with pytest.raises(bo.GridDimensionError) as excinfo:
            bo.mar1D(good, bad )
        assert "x and mar must have the same size" in str(excinfo.value)
        with pytest.raises(bo.GridDimensionError) as excinfo:
            bo.mar1D(bad, good)
        assert "x and mar must have the same size" in str(excinfo.value)

    def test_getitem_success(self):
        # check that the slicing of array on array works
        data = bo.mar1D([1,2,3], [4,5,6])
        data = data[0:2]
        assert len(data.x) == 2
        np.testing.assert_array_equal(data.x, np.array([1,2]))
        np.testing.assert_array_equal(data.mar, np.array([4,5]))

        # check that the slicing of a single index on array works
        data = bo.mar1D([1,2,3], [4,5,6])
        data = data[0]
        assert data.x == 1
        assert data.mar == 4

        # check that the slicing of a single-item on a single-item works
        data = bo.mar1D(1, 4)
        data = data[0]
        assert data.x == 1
        assert data.mar == 4

    @pytest.mark.parametrize('bad_index',[20, 'string'])
    def test_getitem_failure(self, bad_index):
        data = bo.mar1D([1,2,3], [4,5,6])
        with pytest.raises(IndexError) as excinfo:
            data[bad_index]
        assert "is out of bounds for mar1D object" in str(excinfo.value)
