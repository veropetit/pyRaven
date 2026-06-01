import pytest
import matplotlib.pyplot as plt

import numpy as np

from pyRaven import BayesObjects2 as bo

class Test_Chi_Object:

    def test_chi_metadata_assignment(self):
        """Verify that subclass-specific properties are correctly assigned."""
        data = np.zeros((2, 3, 4, 2))
        beta = np.array([10, 20])
        bpole = np.array([100, 200, 300])
        phi = np.array([0, 90, 180, 270])
        incl = np.array([0,90])
        
        chi_obj = bo.Chi(data, beta, bpole, phi, incl, obsID="OBS_A")
        
        assert chi_obj.obsID == "OBS_A"    

    def test_chi_slicing_returns_chi_type(self):
        """Verify slicing a Chi object retains the exact child class type and fields."""
        data = np.zeros((2, 3, 4, 2))
        chi_obj = bo.Chi(data, 
                         [10, 20], 
                         [100, 200, 300], 
                         [0, 90, 180, 270], 
                         [0, 90], 
                         obsID="B")
        
        # Slice it
        sliced = chi_obj[0:1, :, :, :]
        
        # Assert type polymorphism rules
        assert isinstance(sliced, bo.Chi)
        assert sliced.obsID == "B"

    def test_chi_coordinate_accessors(self):
        """Verify that coordinates are bound to the exact expected REQUIRED_COORDS strings."""
        data = np.zeros((2, 3, 4, 1))
        chi_obj = bo.Chi(data, 
                         [10, 20], 
                         [100, 200, 300], 
                         [0, 90, 180, 270], 
                         [45.0], 
                         obsID="TEST")
        
        # Test __getattr__ routing through base class works with the object's explicit science keys
        np.testing.assert_array_equal(chi_obj.beta_coord, np.array([10, 20]))
        np.testing.assert_array_equal(chi_obj.Bpole_coord, np.array([100, 200, 300]))
        np.testing.assert_array_equal(chi_obj.phi_coord, np.array([0, 90, 180, 270]))
        np.testing.assert_array_equal(chi_obj.incl_coord, np.array([45]))

    def test_chi_dimension_mismatch_raises_error(self):
        """Verify that providing transposed arrays triggers GridDimensionError."""
        # Intentional misalignment: data shape does not align with REQUIRED_COORDS array
        wrong_shaped_data = np.zeros((4, 3, 2, 1)) 
        beta = np.array([1, 2])          # len = 2
        bpole = np.array([10, 20, 30])   # len = 3
        phi = np.array([0, 1, 2, 3])     # len = 4
        incl = np.array([45])     # len = 1
        
        from pyRaven.BaseBayesObject import GridDimensionError
        
        with pytest.raises(GridDimensionError):
            bo.Chi(wrong_shaped_data, beta, bpole, phi, incl, obsID="ERR")

    def test_chi_empty_factory_initialization(self):
        """
        Verify that the Chi.empty alternative constructor correctly accepts 
        explicit scientific parameters, allocates a 3D zeroed array of the 
        proper shape, and preserves subclass attributes.
        """
        # 1. Set up realistic sample grids for a chi2 calculation
        beta_grid = [15.0, 30.0, 45.0]               # len = 3
        bpole_grid = np.array([1000, 2000, 3000, 4000]) # len = 4
        phi_grid = (0, 90, 180, 270, 360)            # len = 5
        incl_grid = 45                              # len = 1
        
        test_obsID = "OBS_12345"

        # 2. Call the explicit alternative constructor
        # This verifies the method signature accepts positional/keyword arguments cleanly
        chi_obj = bo.Chi.empty(
            beta_coord=beta_grid,
            Bpole_coord=bpole_grid,
            phi_coord=phi_grid,
            incl_coord=incl_grid,
            obsID=test_obsID
        )

        # 3. Assertions
        # Verify type integrity
        assert isinstance(chi_obj, bo.Chi)

        # Verify the multi-dimensional shape calculation (3, 4, 5, 1)
        expected_shape = (3, 4, 5, 1)
        assert chi_obj.prob.shape == expected_shape
        
        # Verify the matrix is perfectly pre-allocated with zeros
        np.testing.assert_array_equal(chi_obj.prob, np.zeros(expected_shape))

        # Verify that coordinate array transformations passed safely through the constructor
        np.testing.assert_array_equal(chi_obj.beta_coord, np.array(beta_grid))
        np.testing.assert_array_equal(chi_obj.Bpole_coord, bpole_grid)
        np.testing.assert_array_equal(chi_obj.phi_coord, np.array(phi_grid))
        np.testing.assert_array_equal(chi_obj.incl_coord, np.array(incl_grid))

        # Verify subclass-specific metadata attributes are securely bound
        assert chi_obj.obsID == test_obsID    

    def test_chi_write_and_read_roundtrip(self, tmp_path):
        """
        Integration test ensuring a Chi object can be written to disk 
        and successfully reloaded via Chi.read() with perfect data fidelity.
        """
        file_path = tmp_path / "roundtrip_chi.h5"

        # 1. Setup a fully populated Chi object using your .empty() constructor
        original_chi = bo.Chi.empty(
            beta_coord=[10, 20], 
            Bpole_coord=[1000, 2000], 
            phi_coord=[0, 90, 180], 
            incl_coord=30.0, 
            obsID="ROUNDTRIP_TEST"
        )
        # Give the probability array some distinctive dummy data values
        original_chi.prob[:] = np.random.random(original_chi.prob.shape)

        # 2. Write it to disk using the base class writer method
        original_chi.write(str(file_path))

        # 3. Reload it using your new Chi.read() classmethod
        loaded_chi = bo.Chi.read(str(file_path))

        # 4. Assertions: Verify everything survived the round trip perfectly
        assert isinstance(loaded_chi, bo.Chi)
        assert loaded_chi.obsID == original_chi.obsID
        
        # Check that data arrays match exactly
        np.testing.assert_array_equal(loaded_chi.prob, original_chi.prob)
        np.testing.assert_array_equal(loaded_chi.beta_coord, original_chi.beta_coord)
        np.testing.assert_array_equal(loaded_chi.Bpole_coord, original_chi.Bpole_coord)
        np.testing.assert_array_equal(loaded_chi.phi_coord, original_chi.phi_coord)
        np.testing.assert_array_equal(loaded_chi.incl_coord, original_chi.incl_coord)

class Test_LnLikelihood_Object:

    def test_lnlikelihood_metadata_assignment(self):
        """Verify that subclass-specific properties are correctly assigned."""
        data = np.zeros((2, 3, 4, 2, 1))
        beta = np.array([10, 20])
        bpole = np.array([100, 200, 300])
        phi = np.array([0, 90, 180, 270])
        incl = np.array([0, 90])
        noise = 1.0
        
        lnlh_obj = bo.LnLikelihood(data, beta, bpole, phi, incl, noise, obsID="OBS_A")
        
        assert lnlh_obj.obsID == "OBS_A"      

    def test_lnlikelihood_slicing_returns_LnLikelihood_type(self):
        """Verify slicing a Chi object retains the exact child class type and fields."""
        data = np.zeros((2, 3, 4, 2, 1))
        lnlh_obj = bo.LnLikelihood(data, 
                         [10, 20], 
                         [100, 200, 300], 
                         [0, 90, 180, 270], 
                         [0, 90],
                         1.0,
                         obsID="B")
        
        # Slice it
        sliced = lnlh_obj[0:1, :, :, :, :]
        
        # Assert type polymorphism rules
        assert isinstance(sliced, bo.LnLikelihood)
        assert sliced.obsID == "B"

    def test_lnlikelihood_coordinate_accessors(self):
        """Verify that coordinates are bound to the exact expected REQUIRED_COORDS strings."""
        data = np.zeros((2, 3, 4, 2, 1))
        lnlh_obj = bo.LnLikelihood(data, 
                         [10, 20], 
                         [100, 200, 300], 
                         [0, 90, 180, 270], 
                         [0,90],
                         1.0,
                         obsID="TEST")
        
        # Test __getattr__ routing through base class works with the object's explicit science keys
        np.testing.assert_array_equal(lnlh_obj.beta_coord, np.array([10, 20]))
        np.testing.assert_array_equal(lnlh_obj.Bpole_coord, np.array([100, 200, 300]))
        np.testing.assert_array_equal(lnlh_obj.phi_coord, np.array([0, 90, 180, 270]))
        np.testing.assert_array_equal(lnlh_obj.incl_coord, np.array([0, 90]))
        np.testing.assert_array_equal(lnlh_obj.noise_coord, np.array([1.0]))

    def test_lnlikelihood_dimension_mismatch_raises_error(self):
        """Verify that providing transposed arrays triggers GridDimensionError."""
        # Intentional misalignment: data shape does not align with REQUIRED_COORDS array
        wrong_shaped_data = np.zeros((4, 3, 2, 2)) 
        beta = np.array([1, 2])          # len = 2
        bpole = np.array([10, 20, 30])   # len = 3
        phi = np.array([0, 1, 2, 3])     # len = 4
        incl = np.array([0,90])          # len = 2
        noise = np.array([0.5, 1.0, 2.0]) # len = 3
        
        from pyRaven.BaseBayesObject import GridDimensionError
        
        with pytest.raises(GridDimensionError):
            bo.LnLikelihood(wrong_shaped_data, beta, bpole, phi, incl, noise, obsID="ERR")    

    def test_lnlikelihood_empty_factory_initialization(self):
        """
        Verify that the LnLikelihood.empty alternative constructor correctly accepts 
        explicit scientific parameters, allocates a 4D zeroed array of the 
        proper shape, and preserves subclass attributes.
        """
        # 1. Set up realistic sample grids for a chi2 calculation
        beta_grid = [15.0, 30.0, 45.0]               # len = 3
        bpole_grid = np.array([1000, 2000, 3000, 4000]) # len = 4
        phi_grid = (0, 90, 180, 270, 360)            # len = 5
        incl_grid = [0, 90.0]                         #len = 2
        noise_grid = [1.0]                             # len = 1
        test_obsID = "OBS_12345"

        # 2. Call the explicit alternative constructor
        # This verifies the method signature accepts positional/keyword arguments cleanly
        lnlh_obj = bo.LnLikelihood.empty(
            beta_coord=beta_grid,
            Bpole_coord=bpole_grid,
            phi_coord=phi_grid,
            incl_coord=incl_grid,
            noise_coord = noise_grid,
            obsID=test_obsID
        )

        # 3. Assertions
        # Verify type integrity
        assert isinstance(lnlh_obj, bo.LnLikelihood)

        # Verify the multi-dimensional shape calculation (3, 4, 5, 2, 1)
        expected_shape = (3, 4, 5, 2, 1)
        assert lnlh_obj.prob.shape == expected_shape
        
        # Verify the matrix is perfectly pre-allocated with zeros
        np.testing.assert_array_equal(lnlh_obj.prob, np.zeros(expected_shape))

        # Verify that coordinate array transformations passed safely through the constructor
        np.testing.assert_array_equal(lnlh_obj.beta_coord, np.array(beta_grid))
        np.testing.assert_array_equal(lnlh_obj.Bpole_coord, bpole_grid)
        np.testing.assert_array_equal(lnlh_obj.phi_coord, np.array(phi_grid))
        np.testing.assert_array_equal(lnlh_obj.incl_coord, np.array(incl_grid))
        np.testing.assert_array_equal(lnlh_obj.noise_coord, np.array(noise_grid))

        # Verify subclass-specific metadata attributes are securely bound
        assert lnlh_obj.obsID == test_obsID 

    def test_lnlikelihood_write_and_read_roundtrip(self, tmp_path):
        """
        Integration test ensuring a LnLikelihood object can be written to disk 
        and successfully reloaded via Chi.read() with perfect data fidelity.
        """
        file_path = tmp_path / "roundtrip_chi.h5"

        # 1. Setup a fully populated LnLikelihood object using your .empty() constructor
        original_lnlh = bo.LnLikelihood.empty(
            beta_coord=[10, 20], 
            Bpole_coord=[1000, 2000], 
            phi_coord=[0, 90, 180], 
            incl_coord=[0,90], 
            noise_coord=1.0,
            obsID="ROUNDTRIP_TEST"
        )
        # Give the probability array some distinctive dummy data values
        original_lnlh.prob[:] = np.random.random(original_lnlh.prob.shape)

        # 2. Write it to disk using the base class writer method
        original_lnlh.write(str(file_path))

        # 3. Reload it using your new Chi.read() classmethod
        loaded_lnlh = bo.LnLikelihood.read(str(file_path))

        # 4. Assertions: Verify everything survived the round trip perfectly
        assert isinstance(loaded_lnlh, bo.LnLikelihood)
        assert loaded_lnlh.obsID == original_lnlh.obsID
        
        # Check that data arrays match exactly
        np.testing.assert_array_equal(loaded_lnlh.prob, original_lnlh.prob)
        np.testing.assert_array_equal(loaded_lnlh.beta_coord, original_lnlh.beta_coord)
        np.testing.assert_array_equal(loaded_lnlh.Bpole_coord, original_lnlh.Bpole_coord)
        np.testing.assert_array_equal(loaded_lnlh.phi_coord, original_lnlh.phi_coord)
        np.testing.assert_array_equal(loaded_lnlh.incl_coord, original_lnlh.incl_coord)
        np.testing.assert_array_equal(loaded_lnlh.noise_coord, original_lnlh.noise_coord)

class Test_statmethod_compute_core_matrix_in_LnLikelihood:

    @pytest.mark.parametrize('chi2_red, noise, expected', [
        (1.0, 1.0, -0.5),
        (2.0, 1.0, -1.0),
        (1, 2, np.log(2)/2-1)
    ])
    def test_math_chi_and_noise_success(self, chi2_red, noise, expected):
        """Test that the non-constant portion of the math is correct"""
        dummy_chi = np.array([[[[chi2_red]]]]) # Simple 4D array
        dummy_noise = np.array([noise])     # Simple 1D array
    
        # Test just the variable broadcasting math by leaving metadata as None
        raw_result = bo.LnLikelihood._compute_core_matrix(dummy_chi, dummy_noise)

        assert raw_result.shape == (1,1,1,1,1)

        # Assert exact expected values from your formula
        np.testing.assert_almost_equal(raw_result[0,0,0,0,0], expected)

    def test_broadcast_of_noise(self):
        """Test that the array broadcast correctly"""
        dummy_chi = np.ones((1,1,2,2)) # Simple 4D array filled with ones
        dummy_chi[0,0,:,0] = 10.0 # set the last dimension to be 2.0
        dummy_noise = np.array([1,2,3])     # Simple 1D array
    
        # Test just the variable broadcasting math by leaving metadata as None
        raw_result = bo.LnLikelihood._compute_core_matrix(dummy_chi, dummy_noise)

        assert raw_result.shape == (1,1,2,2,3)

        slice = raw_result[0,0,:,:,0] # this should be a 2x2
        expected = np.full((2,2), -0.5)
        expected[:,0] = -5.0
        np.testing.assert_allclose(slice, expected)

        slice = raw_result[0,0,:,:,1] # this should be a 2x2
        expected = np.full((2,2), np.log(2)/2-1)
        expected[:,0] = np.log(2)/2-10
        np.testing.assert_allclose(slice, expected)

    def test_add_constant(self):
        """Test the calculation with the constant"""
        dummy_chi = np.array([[[[1]]]]) # Simple 4D array
        dummy_noise = np.array([1])     # Simple 1D array
        exp_value_of_variable_part_of_math = -0.5
        dummy_N = 20
        dummy_avg_ln_error = -10.0
        exp_value_of_const_part_of_math = -0.5*np.log(2*np.pi) - dummy_avg_ln_error
    
        # Test just the variable broadcasting math by leaving metadata as None
        raw_result = bo.LnLikelihood._compute_core_matrix(dummy_chi, dummy_noise, 
                                                          num_datapoints=dummy_N, avg_ln_error=dummy_avg_ln_error)

        assert raw_result.shape == (1,1,1,1,1)

        # Assert exact expected values from your formula
        expected = (exp_value_of_const_part_of_math + exp_value_of_variable_part_of_math) * dummy_N
        np.testing.assert_almost_equal(raw_result[0,0,0,0,0], expected)    

class Test_LnLikelihood_from_chi2:

    @pytest.fixture
    def correct_chi_obj(self):
        "Returns a correctly populated chi object"
        chi = np.ones((2, 3, 4, 1))
        beta = np.array([10, 20])
        bpole = np.array([100, 200, 300])
        phi = np.array([0, 90, 180, 270])
        incl = np.array([45])
        return bo.Chi(chi, beta, bpole, phi, incl, obsID="OBS_A")      

    def test_sucess_with_default_noise(self, correct_chi_obj):
        """Test the successful construction of a lnLHobject"""
        dummy_N = 20
        dummy_avg_ln_err = -10
        lnlh = bo.LnLikelihood.from_redchi2(correct_chi_obj, dummy_N, dummy_avg_ln_err)

        assert isinstance(lnlh, bo.LnLikelihood)
        assert lnlh.prob.ndim == 5
        assert lnlh.prob.shape == (2,3,4,1,1)
        np.testing.assert_array_equal(correct_chi_obj.beta_coord, lnlh.beta_coord)
        np.testing.assert_array_equal(correct_chi_obj.Bpole_coord, lnlh.Bpole_coord)
        np.testing.assert_array_equal(correct_chi_obj.phi_coord, lnlh.phi_coord)
        np.testing.assert_array_equal(correct_chi_obj.incl_coord, lnlh.incl_coord)
        np.testing.assert_array_equal(np.array([1]), lnlh.noise_coord)

        assert lnlh.obsID == 'OBS_A'

        np.testing.assert_almost_equal(lnlh.prob[0,0,0,0,0], dummy_N*(-0.5 - dummy_avg_ln_err - 0.5*np.log(2*np.pi)) )

    def test_sucess_with_vector_noise(self, correct_chi_obj):
        """Test the successful construction of a lnLHobject"""
        dummy_N = 20
        dummy_avg_ln_err = -10
        lnlh = bo.LnLikelihood.from_redchi2(correct_chi_obj, dummy_N, dummy_avg_ln_err,  noise_coord=[1,2,3])

        assert isinstance(lnlh, bo.LnLikelihood)
        assert lnlh.prob.ndim == 5
        assert lnlh.prob.shape == (2,3,4,1,3)
        np.testing.assert_array_equal(correct_chi_obj.beta_coord, lnlh.beta_coord)
        np.testing.assert_array_equal(correct_chi_obj.Bpole_coord, lnlh.Bpole_coord)
        np.testing.assert_array_equal(correct_chi_obj.phi_coord, lnlh.phi_coord)
        np.testing.assert_array_equal(correct_chi_obj.incl_coord, lnlh.incl_coord)
        np.testing.assert_array_equal(np.array([1,2,3]), lnlh.noise_coord)

        assert lnlh.obsID == 'OBS_A'            

        np.testing.assert_almost_equal(lnlh.prob[0,0,0,0,0], dummy_N*(-0.5 - dummy_avg_ln_err - 0.5*np.log(2*np.pi)) )
        np.testing.assert_almost_equal(lnlh.prob[0,0,0,0,1], dummy_N*(0.5*np.log(2)-1 - dummy_avg_ln_err - 0.5*np.log(2*np.pi)) )

    def test_input_failure(self, correct_chi_obj):

        with pytest.raises(TypeError, match = "User Error: num_datapoints must be a valid number. Received type: str"):
            bo.LnLikelihood.from_redchi2(correct_chi_obj, 'wrong_Npoints', -10,  noise_coord=[1,2,3])

        with pytest.raises(TypeError, match='User Error: avg_ln_error must be a valid number. Received type: list'):
            bo.LnLikelihood.from_redchi2(correct_chi_obj, 20, [2],  noise_coord=[1,2,3])

        with pytest.raises(TypeError, match='noise_coord must be a list, numpy array, or float. Got str instead.'):
            bo.LnLikelihood.from_redchi2(correct_chi_obj, 20, -10,  noise_coord='wrong_string')

        with pytest.raises(TypeError, match='User Error: chi_obj must be a valid Chi object. Received type: str'):
            bo.LnLikelihood.from_redchi2('wrong_chi', 20, -10,  noise_coord=[1,2,3])

        
           
