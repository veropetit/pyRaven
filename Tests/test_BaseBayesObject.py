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
        with pytest.raises(TypeError, match="Can't instantiate abstract class BaseBayesObject"):
            bo.BaseBayesObject()

    def test_no_REQUIRED_COORDS_defined_in_subclass(self):
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
        """Should fail if a required name is missing."""
        prob = np.zeros((5, 5, 5))
        arr = np.zeros((5))
        # Missing 'z'
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

    def test_dimension_mismatch(self, MockSciObject3D):
        """Test that it raises GridDimensionError when dims don't match coord count."""
        prob_2d = np.zeros((5, 5))
        arr = np.zeros(5)
        # Only providing one coord for a 2D array
        with pytest.raises(bo.GridDimensionError, match="Dimension mismatch: Prob is 2D, "
                f"but 3 coordinates were provided."):
            MockSciObject3D(prob_2d, x=arr, y=arr, z=arr)

    def test_length_mismatch(self, MockSciObject3D):
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
        """Returns a valid BaseBayesObject for use in tests."""
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
        
        assert np.array_equal(result.prob, obj3D.prob**2 * 2)
        assert isinstance(result, type(obj3D))

    def test_math_mismatch_coords(self, obj3D, obj3D_not_compatible):
        """Should fail if coordinate values are different."""
        # Same names, but x-axis is shifted
        
        with pytest.raises(ValueError, match="Coordinates must match exactly to add"):
            _ = obj3D + obj3D_not_compatible
        with pytest.raises(ValueError, match="Coordinates must match exactly to multiply"):
            _ = obj3D * obj3D_not_compatible