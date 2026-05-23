import pytest
import numpy as np
import h5py
from pyRaven import BaseBayesObject as bo

class Test_Base_init:
    
    @pytest.fixture
    def MockSciObject3D(self):
        """A helper to create a 'temporary' 3D science subclass for testing."""
        class MockSciObject3Dclass(bo.BaseBayesObject):
            REQUIRED_COORDS = ['x','y','z']
        return MockSciObject3Dclass 

    @pytest.fixture
    def MockSciObject1D(self):
        """A helper to create a 'temporary' 1D science subclass for testing."""
        class MockSciObject1Dclass(bo.BaseBayesObject):
            REQUIRED_COORDS = ['x']
        return MockSciObject1Dclass 

    @pytest.fixture
    def correct_obj3D(self, MockSciObject3D):
        """Returns a valid BaseBayesObject for use in tests."""
        prob = np.ones( (3,4,5) )
        x = np.array([30, 31, 32])
        y = np.array([40, 41, 42, 43])
        z = np.array([50, 51, 52, 53, 54])
        return MockSciObject3D(prob, x=x, y=y, z=z)
        
    def test_no_base_instantiate(self):
        '''Check that a user cannot instantiate a Base Class directly'''
        with pytest.raises(TypeError, match="Can't instantiate abstract class BaseBayesObject"):
            bo.BaseBayesObject()

    def test_no_REQUIRED_COORDS_defined_in_subclass(self):
        '''Check that an error is returned if the child class does not define the coordinates in REQUIRED_COORDS'''
        class SciObj(bo.BaseBayesObject):
            pass
        with pytest.raises(TypeError, match="Can't instantiate abstract class SciObj without an"):
            obj = SciObj()

    def test_getattr_access(self, correct_obj3D):
        """Verify that __getattr__ works for coordinates."""
        # Using getattr should match the internal dict
        assert np.array_equal(correct_obj3D.x, correct_obj3D.coords['x'])
        assert np.array_equal(correct_obj3D.y, correct_obj3D.coords['y'])
        assert np.array_equal(correct_obj3D.z, correct_obj3D.coords['z'])

    def test_getattr_access_failure(self, correct_obj3D):
        '''Verify that calling an unexisting coords fail'''
        with pytest.raises(AttributeError, match="'MockSciObject3Dclass' has no attribute 'noexist'"):
            correct_obj3D.noexist

    def test_initialization_shapes(self, correct_obj3D):
        """Test that prob and coords are stored with correct shapes."""
        assert correct_obj3D.prob.shape == (3,4,5)
        assert len(correct_obj3D.x) == 3
        assert len(correct_obj3D.y) == 4
        assert len(correct_obj3D.z) == 5

    def test_missing_required_coord(self, MockSciObject3D):
        """Should fail if a required name is missing."""
        prob = np.zeros((5, 5, 5))
        arr = np.zeros((5))
        # Missing 'z'
        with pytest.raises(KeyError, match="Missing: {'z'}"):
            MockSciObject3D(prob, x=arr, y=arr)

    def test_more_than_required_coord(self, MockSciObject3D):
        """Should fail if an extra coordinate name is given."""
        prob = np.zeros((5, 5, 5))
        arr = np.zeros((5))
        with pytest.raises(KeyError, match="Unexpected: {'w'}"):
            MockSciObject3D(prob, x=arr, y=arr, z=arr, w=arr)

    def test_wrong_coord_order(self, MockSciObject3D):
        """Should fail if coordinates are provided in the wrong order."""            
        prob = np.zeros((5, 5, 5))
        arr = np.zeros((5))
        # Provided as beta then Bpole
        with pytest.raises(KeyError, match="Incorrect coordinate order"):
            MockSciObject3D(prob, x=arr, z=arr, y=arr)

    def test_invalid_coordinate_dimensions(self, MockSciObject3D):
        """Test that passing a 2D array as a coordinate raises ValueError."""
        prob = np.zeros((5, 5, 5))
        goodcoord = np.zeros(5)
        badcoord = np.zeros((5, 2)) 
        
        with pytest.raises(bo.GridDimensionError, match="Coordinate 'x' must be 0D or 1D. Received a 2D array."):
            MockSciObject3D(prob, x=badcoord, y=goodcoord, z=goodcoord)

    def test_coordinate_dimension_mismatch(self, MockSciObject3D):
        """Test that it raises GridDimensionError when dims don't match coord count."""
        prob_2d = np.zeros((5, 5))
        arr = np.zeros(5)
        # Only providing one coord for a 2D array
        with pytest.raises(bo.GridDimensionError, match="Dimension mismatch: Prob is 2D, "
                f"but 3 coordinates were provided."):
            MockSciObject3D(prob_2d, x=arr, y=arr, z=arr)

    def test_coordinate_length_mismatch(self, MockSciObject3D):
        """Test that it raises GridDimensionError when coord length doesn't match data shape."""
        prob = np.zeros((5,5,5))
        goodcoord = np.zeros(5)
        badcoord = np.zeros((7)) 
        with pytest.raises(bo.GridDimensionError, match=r"Bayes object Probability data dimension 1 \(5 elements\) does not match the lenght of the 'y' coordinate array \(7 elements\)"):
            MockSciObject3D(prob, x=goodcoord, y=badcoord, z=goodcoord)

    @pytest.mark.parametrize("coord, coord_dim, prob, prob_dim", [
        (10, 0, 0.5, 0), # A 0D prob array with a single 0D coordinate
        (10, 0, [0.5], 1), # A 1D prob array with a single 0D coordinate
        ([10], 1, 0.5, 0), # A 0D prob array with a single 1D coordinate
        ([10], 1, [0.5], 1) # A 1D prob array with a single 1D coordinate
    ])
    def test_scalar_coordinate_success(self, prob, prob_dim, coord, coord_dim, MockSciObject1D):
        """Test that a 0D (scalar) coordinate is accepted."""        
        obj = MockSciObject1D(np.array(prob), x=np.array(coord))
        assert obj.x == 10.0
        assert obj.x.ndim == coord_dim
        assert obj.prob == 0.5
        assert obj.prob.ndim == prob_dim
    
    def test_slice_3d_to_3d(self, correct_obj3D):
            """Test standard slicing that preserves dimensions (3D -> 3D)."""
            # Slice: x[0:2], y[all], z[0:1]
            sliced = correct_obj3D[0:2, :, 0:1]
            
            assert sliced.prob.shape == (2, 4, 1)
            assert isinstance(sliced, type(correct_obj3D))
            
            # Coordinates should be sliced but still 1D
            assert np.array_equal(sliced.x, [30, 31])
            assert np.array_equal(sliced.y, [40, 41, 42, 43])
            assert np.array_equal(sliced.z, [50])
            assert sliced.z.ndim == 1

    def test_integer_indexing_preserves_ndim(self, correct_obj3D):
        """Verify that obj[0] results in a 3D object, not 2D."""
        # Slicing axis 0 with an integer
        sliced = correct_obj3D[0]
        
        # Validation check: Did it stay 3D?
        assert sliced.prob.ndim == 3
        assert sliced.prob.shape == (1, 4, 5)
        
        # Coordinate check: Is x still 1D?
        assert sliced.x.ndim == 1
        assert len(sliced.x) == 1
        assert sliced.x[0] == 30

    def test_step_slicing(self, correct_obj3D):
        """Test taking every second value: obj[::2]."""
        sliced = correct_obj3D[::2, :, :]
        
        # Axis 0 was length 3, so every 2nd point makes it length 2
        assert sliced.prob.shape == (2, 4, 5)
        assert np.array_equal(sliced.x, [30, 32])

    def test_ellipsis_and_trailing_slices(self, correct_obj3D):
        """Test that implicit slices (like obj[0]) don't break trailing coords."""
        # Only specifying the first dimension
        sliced = correct_obj3D[0:2] 
        
        # Axes 1 and 2 should remain full length
        assert sliced.prob.shape == (2, 4, 5)
        assert len(sliced.y) == 4
        assert len(sliced.z) == 5

    def test_repr_output(self, correct_obj3D):
        """Verify the string representation is accurate and descriptive."""
        representation = repr(correct_obj3D)
        
        # 1. Check that the subclass name is present
        assert "MockSciObject3Dclass" in representation
        
        # 2. Check that the probability shape is correctly reported
        assert "(3, 4, 5)" in representation
        
        # 3. Check that all coordinate names are listed
        assert "x" in representation
        assert "y" in representation
        assert "z" in representation

    def test_repr_after_slicing(self, correct_obj3D):
        """Verify repr updates correctly after the object is sliced."""
        sliced = correct_obj3D[0:1, :, 0:2]
        representation = repr(sliced)
        # 1. Check that the subclass name is present
        assert "MockSciObject3Dclass" in representation        
        # The shape in repr should reflect the slice, not the original
        assert "(1, 4, 2)" in representation
        assert "x" in representation
        assert "y" in representation
        assert "z" in representation

    def test_added_attr_persist_after_slicing(self):
        '''Test that the extra Science class attributes stay after slicing'''
        # 1. Define a temporary mock science class that uses the base class
        
        class MockScienceGrid(bo.BaseBayesObject):
            REQUIRED_COORDS = ['x', 'y']
            PROB_IS_LOG = False

            def __init__(self, prob, x, y, a, b):
                super().__init__(prob=prob, x=x, y=y)
                self.a = a
                self.b = b
        
        # 2. Initialize the mock science object with distinct metadata
        prob_data = np.ones((3, 4))
        x_vals = np.array([10, 20, 30])
        y_vals = np.array([1, 2, 3, 4])     
        
        orig_obj = MockScienceGrid(
            prob=prob_data, 
            x=x_vals, 
            y=y_vals, 
            a='a', 
            b=3
        )

        # Slice a subgrid out of the object
        sliced_obj = orig_obj[0:2, :]   

        # Verify array slicing happened properly
        assert sliced_obj.prob.shape == (2, 4)
        # Verify metadata survived the slice dynamically
        assert hasattr(sliced_obj, 'a')
        assert sliced_obj.a == 'a'
        assert hasattr(sliced_obj, 'b')
        assert sliced_obj.b == 3

    def test_h5_persistence(self, correct_obj3D, tmp_path):
        """Verify that all arrays are correctly saved to HDF5."""
        file_path = tmp_path / "test_data.h5"
        
        # 1. Write the file
        correct_obj3D.write(file_path)
        
        # 2. Read it back and verify
        with h5py.File(file_path, 'r') as f:
            assert np.array_equal(f['prob'][:], correct_obj3D.prob)
            assert np.array_equal(f['x'][:], correct_obj3D.x)
            assert np.array_equal(f['y'][:], correct_obj3D.y)
            assert np.array_equal(f['z'][:], correct_obj3D.z)
            assert f.attrs['class_name'] == "MockSciObject3Dclass"    

    def test_h5_persistance_extra_key(self, tmp_path):
        '''Check that the k5 writer captures the extra science object attirbutes'''
        # 1. Define a temporary mock science class that uses the base class
        class MockScienceGrid(bo.BaseBayesObject):
            REQUIRED_COORDS = ['x', 'y']
            PROB_IS_LOG = False

            def __init__(self, prob, x, y, a, b):
                super().__init__(prob=prob, x=x, y=y)
                self.a = a
                self.b = b
        
        # 2. Initialize the mock science object with distinct metadata
        prob_data = np.ones((3, 4))
        x_vals = np.array([10, 20, 30])
        y_vals = np.array([1, 2, 3, 4])     
        
        obj = MockScienceGrid(
            prob=prob_data, 
            x=x_vals, 
            y=y_vals, 
            a='a', 
            b=3
        )   

        # 1. Write the file
        file_path = tmp_path / "test_data.h5"
        obj.write(file_path)
        
        # 2. Read it back and verify
        with h5py.File(file_path, 'r') as f:
            assert np.array_equal(f['prob'][:], obj.prob)
            assert np.array_equal(f['x'][:], obj.x)
            assert np.array_equal(f['y'][:], obj.y)
            assert f.attrs['class_name'] == "MockScienceGrid"
            assert f.attrs['a'] == 'a'
            assert f.attrs['b'] == 3    
     
    @pytest.mark.parametrize("axis_input, expected_indices",[
            (None, [0, 1, 2]),
            (1, [1]),
            ((0, 1), [0, 1]),
            ([0, 1], [0, 1]),
            ('x', [0]),
            (['x', 'y'], [0, 1]),
            (['x', 'z'], [0, 2]),
            (['z', 'x'], [0, 2]),
            (['z', 0, 0], [0, 2]),
        ]
    )
    def test_get_validated_axes_indexes(self, correct_obj3D, axis_input, expected_indices):
        """Verify that _get_validated_axes_indexes standardizes various inputs correctly."""
        result = correct_obj3D._get_validated_axes_indexes(axis=axis_input)
        assert result == expected_indices

    @pytest.mark.parametrize("invalid_axis, expected_exception, match_msg",[
            # Test out-of-bounds indices
            (3, IndexError, "Axis index 3 is out of bounds."),
            (-1, IndexError, "Axis index -1 is out of bounds."),
            # Test invalid coordinate string names
            ('a', ValueError, "Coordinate 'a' is not valid for this object."),
            (['x', 'invalid_name'], ValueError, "Coordinate 'invalid_name' is not valid for this object."),
            # Test completely unsupported data types
            (2.3, TypeError, "Invalid axis identifier type: <class 'float'>"),
            ({'x': 1}, TypeError, "Invalid axis identifier type: <class 'dict'>"),
            ([0, 1.5], TypeError, "Invalid axis identifier type: <class 'float'>"),
        ]
    )
    def test_get_validated_axes_indexes_exceptions(self, correct_obj3D, invalid_axis, expected_exception, match_msg):
        """Verify that _get_validated_axes_indexes raises correct exceptions for bad inputs."""
        with pytest.raises(expected_exception, match=match_msg):
            correct_obj3D._get_validated_axes_indexes(axis=invalid_axis)                   

class Test_Base_Math:
    
    @pytest.fixture
    def MockSciObject3D(self):
        """A helper to create a 'temporary' 3D science subclass for testing."""
        class MockSciObject3Dclass(bo.BaseBayesObject):
            REQUIRED_COORDS = ['x','y','z']
        return MockSciObject3Dclass 

    @pytest.fixture
    def obj3D(self, MockSciObject3D):
        """Returns a valid BaseBayesObject for use in tests."""
        prob = np.ones( (3,4,5) )
        x = np.array([30, 31, 32])
        y = np.array([40, 41, 42, 43])
        z = np.array([50, 51, 52, 53, 54])
        return MockSciObject3D(prob, x=x, y=y, z=z)

    @pytest.fixture
    def obj3D_not_compatible(self, MockSciObject3D):
        """Returns a valid BaseBayesObject that has different coordinate values than obj3D."""
        prob = np.ones( (3,4,5) )
        x = np.array([60, 61, 62])
        y = np.array([40, 41, 42, 43])
        z = np.array([50, 51, 52, 53, 54])
        return MockSciObject3D(prob, x=x, y=y, z=z)

    def test_multiplication_success(self, obj3D):
        """Test multiplying two identical objects (e.g., Likelihood * Prior)."""
        # Create a second object with the same coords but different data
        other = type(obj3D)(obj3D.prob * 2, **obj3D.coords)
        
        result = obj3D * other
        
        assert np.all(result.prob == 2.0)
        assert isinstance(result, type(obj3D))

    def test_scalar_multiplication_sucess(self, obj3D):
        """Test obj * 2 and 2 * obj."""
        # obj * scalar
        result = obj3D * 10
        assert np.all(result.prob == 10.0) # Since obj3D is all ones
        assert result.prob.shape == obj3D.prob.shape
        
        # scalar * obj (handled by __rmul__)
        result_rev = 5 * obj3D
        assert np.all(result_rev.prob == 5.0)

    def test_addition_success(self, obj3D):
        """Test adding two identical objects"""
        # Create a second object with the same coords but different data
        other = type(obj3D)(obj3D.prob * 2, **obj3D.coords)
        
        result = obj3D + other
        assert np.all(result.prob == 3.0)
        assert isinstance(result, type(obj3D))

    def test_scalar_addition_sucess(self, obj3D):
            """Test obj + 1.5."""
            result = obj3D + 1.5
            assert np.all(result.prob == 2.5)
            # Verify coordinates are still there
            assert len(result.x) == len(obj3D.x)

    def test_subtraction_success(self, obj3D):
        """Test adding two identical objects"""
        # Create a second object with the same coords but different data
        other = type(obj3D)(obj3D.prob * 2, **obj3D.coords)
        
        result = obj3D - other
        assert np.all(result.prob == -1.0)
        assert isinstance(result, type(obj3D))

    def test_scalar_subtraction_sucess(self, obj3D):
        """Test obj - scalar and scalar - obj."""
        # obj - scalar
        res1 = obj3D - 1.0
        assert np.all(res1.prob == 0.0) # 1.0 - 1.0
        
        # scalar - obj
        res2 = 10.0 - obj3D
        assert np.all(res2.prob == 9.0) # 10.0 - 1.0

    def test_division_sucess(self, obj3D):
        """Test multiplying two identical objects (e.g., Likelihood * Prior)."""
        # Create a second object with the same coords but different data
        other = type(obj3D)(obj3D.prob * 2, **obj3D.coords)
        
        result = obj3D / other
        
        assert np.all(result.prob == 0.5)
        assert isinstance(result, type(obj3D))

    def test_scalar_division_sucess(self, obj3D):
        """Test obj / scalar and scalar / obj."""
        # obj / scalar
        res1 = obj3D / 2.0
        assert np.all(res1.prob == 0.5)
        
        # scalar / obj
        res2 = 2.0 / obj3D
        assert np.all(res2.prob == 2.0)

    def test_division_by_zero(self, obj3D):
        """Verify standard NumPy behavior for div by zero (usually a warning/inf)."""
        # NumPy will issue a RuntimeWarning and return inf
        with np.errstate(divide='ignore'):
            res = obj3D / 0.0
            assert np.all(np.isinf(res.prob))

    def test_math_mismatch_coords(self, obj3D, obj3D_not_compatible):
        """Math operations should fail if coordinate values are different."""
        # Same names, but x-axis is shifted
        
        with pytest.raises(ValueError, match="Coordinates must match exactly to add"):
            _ = obj3D + obj3D_not_compatible
        with pytest.raises(ValueError, match="Coordinates must match exactly to multiply"):
            _ = obj3D * obj3D_not_compatible
        with pytest.raises(ValueError, match="Coordinates must match exactly to divide"):
            _ = obj3D / obj3D_not_compatible
        with pytest.raises(ValueError, match="Coordinates must match exactly to subtract"):
            _ = obj3D - obj3D_not_compatible

    def test_get_axis_log_dv_sucess(self, MockSciObject3D):
        '''Test the return value of get_axis_log_dv'''
        prob = np.ones( (3,3,3) )
        x = np.array([1,2,3])
        y = np.array([0.5, 1.0, 1.5])
        z = np.array([0.1, 0.2, 0.3])
        obj = MockSciObject3D(prob, x=x, y=y, z=z)
        dv = obj._get_axis_log_dv([0,1,2])
        np.testing.assert_allclose(dv, np.log(0.05))

    def test_log_sum_exp_global_sum(self, obj3D):
        '''Test the total sum for probability in log'''
        # Overwriting the prob in obj3D just for this test
        # (The coordinate dimensions won't match)        
        # Create a 2x2x2 cube of ln(1) [which is 0.0]
        # Sum of eight 1s should be ln(8)
        obj3D.prob = np.zeros((2, 2, 2)) 
    
        result = obj3D._log_sum_exp([0,1,2])
    
        # Ensure it's a scalar (or a 0D array)
        assert np.isscalar(result) or result.ndim == 0
        # Check the actual value
        assert np.isclose(result, np.log(8.0))

    def test_log_sum_exp_axis_sum(self, obj3D):
        '''Test the sum over a single axis for the probability in log'''
        # Overwriting the prob in obj3D just for this test
        # (The coordinate dimensions won't match)        
        # Create a 2x2x2 cube of ln(1) [which is 0.0]
        # Sum of eight 1s should be ln(8)
        obj3D.prob = np.zeros((2, 2, 2)) 
    
        result = obj3D._log_sum_exp([0])
        v = np.log(2)
        expected = np.array([[v,v],[v,v]])
        # Ensure it's a 2D array with the correct shape
        assert result.ndim == 2
        assert result.shape == (2,2)
        # Check the actual values
        np.testing.assert_allclose(result, expected)

    def test_log_sum_exp_extreme_values(self, obj3D):
        """
        Test that _log_sum_exp handles values that usually cause overflows.
        exp(1000) is larger than a 64-bit float can handle.
        """
        # ln(P) values that are massive
        # ln(e^1000 + e^1000) should be 1000 + ln(2)
        large_data = np.array([1000.0, 1000.0])

        # Overwriting the prob in obj3D just for this test
        # (The coordinate dimensions won't match)        

        obj = obj3D
        obj.prob = large_data #overwriting the "prob" just for this test
        
        result = obj._log_sum_exp([0])
        
        assert np.isclose(result, 1000.0 + np.log(2.0))
        assert np.isfinite(result) # Ensure it didn't become 'inf'

    def test_log_sum_all_minf(self, MockSciObject3D):
        '''Testing when all values are -inf in logP'''

        # Overwriting the prob in obj3D just for this test
        # (The coordinate dimensions won't match)   
        prob = np.full((2, 3, 4), -np.inf)
        obj = MockSciObject3D(prob, x=[1,2], y=[1,2,3], z=[1,2,3,4])
        
        # Case 1: Collapse ALL axes ([0, 1, 2])
        # Expected: A single scalar value (no shape), or a 0D array
        res_all = obj._log_sum_exp([0, 1, 2])
        np.testing.assert_equal(res_all, -np.inf)
        assert np.isscalar(res_all) or res_all.ndim == 0

        # Case 2: Collapse only axis 1 (the 'y' axis, length 3)
        # Expected shape: (2, 4) — since axis 0 (len 2) and axis 2 (len 4) remain
        res_one = obj._log_sum_exp([1])
        np.testing.assert_equal(res_one, np.full((2, 4), -np.inf))
        assert res_one.shape == (2, 4)

        # Case 3: Collapse axes 0 and 2 (the 'x' and 'z' axes)
        # Expected shape: (3,) — only axis 1 (the 'y' axis, length 3) remains
        res_two = obj._log_sum_exp([0, 2])
        np.testing.assert_equal(res_two, np.full((3,), -np.inf))
        assert res_two.shape == (3,)

    def test_left_edge_integration1D(self):
        """Verify that 4 left-edges with dx=1 covers a volume of 4.0."""
        class MockLog(bo.BaseBayesObject):
            REQUIRED_COORDS = ['x'] 
            PROB_IS_LOG = True
        
        # Left edges at 0, 1, 2, 3. Total width should be 4.
        x = np.array([0, 1, 2, 3])
        prob = np.zeros(4) # ln(1)
        obj = MockLog(prob, x=x)
        
        # sum(exp(0)) = 4. 
        # Integral = 4 (sum) * 1 (dx) = 4.0
        # ln(Integral) = ln(4.0)
        result = obj._log_integrate_uniform([0])
        np.testing.assert_allclose(result, np.log(4.0))        

    def test_left_edge_integration_2d_to_scalar(self):
        """
        Integrate 2D -> Scalar.
        Grid: x (width 0.5), y (width 2.0).
        P = 1.0 everywhere.
        Expected Integral: (N_x * dx) * (N_y * dy)
        """
        class MockLog(bo.BaseBayesObject):
            REQUIRED_COORDS = ['x', 'y']; PROB_IS_LOG = True

        # x: 10 bins, dx=0.5 -> Total span = 5.0
        # y: 5 bins,  dy=2.0 -> Total span = 10.0
        x = np.arange(0, 5, 0.5) 
        y = np.arange(0, 10, 2.0)
        prob = np.zeros((10, 5)) # ln(1.0)
        
        obj = MockLog(prob, x=x, y=y)
        
        # Integrate over both axes (None)
        # Sum is ln(50)
        # ln_weight is ln(0.5) + ln(2.0) = ln(1.0) = 0
        # Result should be ln(50 * 1.0) = ln(50)
        result = obj._log_integrate_uniform([0,1])
        
        assert np.isclose(result, np.log(50.0))

    def test_left_edge_integration_2d_to_1D(self):
        """
        Integrate 2D -> 1D (Integrate over y, keep x).
        We expect a 1D array where each element is the integral along the y-column.
        """
        class MockLog(bo.BaseBayesObject):
            REQUIRED_COORDS = ['x', 'y']; IS_LOG = True

        x = np.array([0, 1])     # dx = 1
        y = np.array([0, 2, 4])  # dy = 2
        
        # Create data where x=0 is all ln(1) and x=1 is all ln(2)
        prob = np.array([
            [np.log(1), np.log(1), np.log(1)],
            [np.log(2), np.log(2), np.log(2)]
        ])

        obj = MockLog(prob, x=x, y=y)
        
        # Integrate over y (axis 1)
        # For x=0: sum is 3, weight is dy=2 -> integral is 6
        # For x=1: sum is 6, weight is dy=2 -> integral is 12
        result_prob = obj._log_integrate_uniform([1])
        
        expected = np.log([6.0, 12.0])
        assert np.allclose(result_prob, expected)
        assert result_prob.shape == (2,)

    def test_left_edge_integration_with_negative_inf(self):
        """
        Ensure that zero-probability regions (-inf) are handled correctly 
        and don't result in NaNs when combined with log-weights.
        """
        class MockLog(bo.BaseBayesObject):
            REQUIRED_COORDS = ['x']; IS_LOG = True
            
        x = np.array([0, 1, 2]) # dx = 1
        prob = np.array([-np.inf, 0.0, -np.inf]) # [0, 1, 0] in linear
        
        obj = MockLog(prob, x=x)
        
        # Integral should be 1.0 * dx = 1.0. ln(1.0) = 0.0
        result = obj._log_integrate_uniform([0])
        assert np.isclose(result, 0.0)

class Test_User_Facing_Marginalize:

    def test_marginalize_linear_removes_metadata_and_sums(self):
        """Verify 3D -> 2D reduction in linear space drops the correct axis."""
        class MockLinear3D(bo.BaseBayesObject):
            REQUIRED_COORDS = ['x', 'y', 'z']
            PROB_IS_LOG = False

        # Setup a 3D space: shapes (2, 3, 4)
        x = np.array([1, 2])
        y = np.array([10, 20, 30])
        z = np.array([100, 200, 300, 400])
        
        # Fill with ones: a sum over 'y' (axis length 3) should result in 30.0s
        prob = np.ones((2, 3, 4))
        obj = MockLinear3D(prob, x=x, y=y, z=z)

        # Act: marginalize out 'y'
        new_prob, new_coords = obj.marginalize(axis='y')

        # Assert 1: Data values and shapes are correct
        expected_prob = np.full((2, 4), 30.0)  # np.sum over axis=1
        np.testing.assert_allclose(new_prob, expected_prob)
        
        # Assert 2: Metadata keys are correct
        assert 'y' not in new_coords
        assert list(new_coords.keys()) == ['x', 'z']
        
        # Assert 3: Remaining coordinate arrays are untouched
        np.testing.assert_array_equal(new_coords['x'], x)
        np.testing.assert_array_equal(new_coords['z'], z)    

    def test_marginalize_log_removes_metadata_and_sums(self):
        """Verify 3D -> 2D reduction in log space drops the correct axis."""
        class MockLinear3D(bo.BaseBayesObject):
            REQUIRED_COORDS = ['x', 'y', 'z']
            PROB_IS_LOG = True

        # Setup a 3D space: shapes (2, 3, 4)
        x = np.array([1, 2])
        y = np.array([10, 20, 30])
        z = np.array([100, 200, 300, 400])
        
        # Fill with ones: a sum over 'y' (axis length 3) should result in 30.0s
        prob = np.zeros((2, 3, 4))
        obj = MockLinear3D(prob, x=x, y=y, z=z)

        # Act: marginalize out 'y'
        new_prob, new_coords = obj.marginalize(axis='y')

        # Assert 1: Data values and shapes are correct
        expected_prob = np.full((2, 4), np.log(30.0))  # np.sum over axis=1
        np.testing.assert_allclose(new_prob, expected_prob)
        
        # Assert 2: Metadata keys are correct
        assert 'y' not in new_coords
        assert list(new_coords.keys()) == ['x', 'z']
        
        # Assert 3: Remaining coordinate arrays are untouched
        np.testing.assert_array_equal(new_coords['x'], x)
        np.testing.assert_array_equal(new_coords['z'], z) 
