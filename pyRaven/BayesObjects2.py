from __future__ import annotations  # This is to allow using the class in the typeing 'hint' in the constructor
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from . import BaseBayesObject as base
from . import validators as valid

class Chi(base.BaseBayesObject):
    """
    Class definition for the objects that will store the Chi2 matrices.
    Inherits core slicing, math overloads, and integration features from BaseBayesObject.
    """    

    # 1. MUST define REQUIRED_COORDS to match the expected dimension ordering of the data array
    REQUIRED_COORDS = ['beta_coord', 'Bpole_coord', 'phi_coord', 'incl_coord']    

    # The chi are not stored in log
    PROB_IS_LOG = False

    def __init__(self, prob, beta_coord, Bpole_coord, phi_coord, incl_coord, obsID='UNKOWN'):
        """
        Initialization of a Chi object.
        """
        # Forward the core array and labeled coordinates to the Base Class constructor.
        super().__init__(
            prob=prob, 
            beta_coord=beta_coord, 
            Bpole_coord=Bpole_coord, 
            phi_coord=phi_coord,
            incl_coord=incl_coord
        )
        
        # Store the extra scientific metadata attributes
        self.obsID = obsID

    @classmethod
    def empty(cls, 
        beta_coord: np.typing.ArrayLike, 
        Bpole_coord: np.typing.ArrayLike, 
        phi_coord: np.typing.ArrayLike, 
        incl_coord: np.typing.ArrayLike, 
        obsID: str | int = 'UNKOWN'
        ) -> Chi:
        """
        Initialize a Chi object pre-allocated with zeros for long loops.
        """
        # Call the base class factory using super()
        # The base class will handle the shape calculation and np.zeros allocation completely!
        return super().empty(
            beta_coord=beta_coord,
            Bpole_coord=Bpole_coord,
            phi_coord=phi_coord,
            incl_coord=incl_coord,
            obsID=obsID
        )

    @classmethod
    def read(cls, filename: str) -> Chi:
        """
        Read an HDF5 file and return a fully populated Chi object.
        """
        # Simply hand the filename up to the base class engine
        return super().read(filename)
 
class LnLikelihood(base.BaseBayesObject):
    '''
    Likelihood class for one observation
    '''

    #-------------------------------------
    # 1. Properties & Initialization
    #-------------------------------------

    REQUIRED_COORDS = ['beta_coord', 'Bpole_coord', 'phi_coord', 'incl_coord', 'noise_coord']
    PROB_IS_LOG = True
    CLASS_ID = 'LnLikelihood'

    def __init__(
        self, 
        prob: np.ndarray, 
        beta_coord: np.typing.ArrayLike, 
        Bpole_coord: np.typing.ArrayLike, 
        phi_coord: np.typing.ArrayLike, 
        incl_coord: np.typing.ArrayLike,
        noise_coord: np.typing.ArrayLike,
        obsID: str | int = 'UNKOWN'
    ):
        # Pass everything safely up to the base class structure
        super().__init__(
            prob=prob, 
            beta_coord=beta_coord, 
            Bpole_coord=Bpole_coord, 
            phi_coord=phi_coord,
            incl_coord=incl_coord,
            noise_coord=noise_coord
        )

        # Store the extra scientific metadata attributes
        self.obsID = obsID

    @classmethod
    def empty(cls, 
        beta_coord: np.typing.ArrayLike, 
        Bpole_coord: np.typing.ArrayLike, 
        phi_coord: np.typing.ArrayLike, 
        incl_coord: np.typing.ArrayLike, 
        noise_coord: np.typing.ArrayLike,
        obsID: str | int = 'UNKOWN'
        ) -> LnLikelihood:
        """
        Initialize a LnLikelihood object pre-allocated with zeros for long loops.
        """
        # Call the base class factory using super()
        # The base class will handle the shape calculation and np.zeros allocation completely!
        return super().empty(
            beta_coord=beta_coord,
            Bpole_coord=Bpole_coord,
            phi_coord=phi_coord,
            incl_coord=incl_coord,
            noise_coord=noise_coord,
            obsID=obsID
        )
    
    #-------------------------------------
    # 2. User-Facing Public API
    #-------------------------------------

    @classmethod
    def read(cls, filename: str) -> LnLikelihood:
        """
        Read an HDF5 file and return a fully populated LnLikelihood object.
        """
        # Simply hand the filename up to the base class engine
        return super().read(filename)
    

    @classmethod
    def from_redchi2(cls, 
        chi_obj: Chi,
        num_datapoints: int, 
        avg_ln_error: float, 
        noise_coord: np.typing.ArrayLike = 1.0,
        ) -> "LnLikelihood":       
    
        """
        Computes the log-likelihood grid across a 1D noise coordinate space.

        This method bridges the 4D Chi-squared space and the 5D Likelihood space 
        by broadcasting across the noise scaling parameter dimension if given 
        (otherwise we assume a single value of the scale-noise parameter of 1.0,
        which is appropriate for odds ratio calculations)

        Parameters
        ----------
        chi_obj : Chi
            An instance of the 4D Chi-squared [reduced!] science class containing the 
            underlying `prob` data matrix and shared 4D coordinates.
        num_datapoints : int
            The total number of valid physical data points in the active observation.
        avg_ln_error : float
            The average of the natural logarithm of the observation's error bars 
            (calculated beforehand in the analysis pipeline).
        noise_coord : float or list or np.ndarray, optional
            The noise coordinate tracking values. Can be a single scalar float, 
            a list, or an existing array; will be automatically validated and 
            expanded to a minimum 1D array.
            Defaults to 1.0

        Returns
        -------
        LnLikelihood
            A fully initialized 5D LnLikelihood object containing the resolved 
            probability matrix and all 5 coordinate axes normalized to 1D arrays.

        Raises
        ------
        TypeError
            If num_datapoints or avg_log_error are not numerical values.
        """

        # 1. Gatekeeper User-Input Type Verification for Scalar Numbers
        if not isinstance(chi_obj, Chi):
            raise TypeError(
                f"User Error: chi_obj must be a valid Chi object. "
                f"Received type: {type(chi_obj).__name__}"
            )

        if not isinstance(num_datapoints, (int, float, np.number)):
            raise TypeError(
                f"User Error: num_datapoints must be a valid number. "
                f"Received type: {type(num_datapoints).__name__}"
            )
        if not isinstance(avg_ln_error, (int, float, np.number)):
            raise TypeError(
                f"User Error: avg_ln_error must be a valid number. "
                f"Received type: {type(avg_ln_error).__name__}"
            )

        # 3. Coordinate Harvesting from parent 4D Chi grid
        noise_arr = valid.convert_to_numpy_and_validate_numerical("noise_coord", noise_coord)
        chi_coords = [getattr(chi_obj, name) for name in chi_obj.REQUIRED_COORDS]

        # 4. Call the math engine with metadata -> allocation happens once
        prob_matrix = cls._compute_core_matrix(
            chi2red_matrix=chi_obj.prob,
            noise_coord=noise_arr,
            num_datapoints=num_datapoints,
            avg_ln_error=avg_ln_error
        )
        # 5.Instantiate the 5D child class object via the standard base constructor
        return cls(prob_matrix, *chi_coords, noise_arr, obsID=chi_obj.obsID)
    
    #-------------------------------------
    # 3. Core Mathematical Backends
    #-------------------------------------
    
    @staticmethod
    def _compute_core_matrix(
        chi2red_matrix: np.ndarray[float], 
        noise_coord: np.ndarray[float], 
        num_datapoints: Optional[int] = None, 
        avg_ln_error: Optional[float] = None
        ) -> np.ndarray:
        """
        Pure mathematical backend. Broadcasts 4D chi2 and 1D noise_coord into a 5D layout.
        If the observation-related constants are provided, it applies the full physical equation in a single pass.
        (Basically, the default None only helps with simplifying the broadcast result in the testing unit)
        (In the user-facing method, they are required).

        """

        # 1. Shape expansion for zero-overhead broadcasting
        chi2_5d = chi2red_matrix[..., np.newaxis]
        n_slices = [np.newaxis] * chi2red_matrix.ndim + [slice(None)]
        noise_5d = noise_coord[tuple(n_slices)]
        # 2. Compute ln(noise)/2 - noise/2 * chi2 in a single chained statement
        # This allows NumPy's internal C-engine to optimize memory allocations
        result = (np.log(noise_5d) / 2.0) - (noise_5d / 2.0 * chi2_5d)

        # 3. Add the constant in-place if the science pipeline provides the metadata
        if num_datapoints is not None and avg_ln_error is not None:

            # Calculate the constant scalar
            # -ln(2 pi)/2 - <ln(sigma_i)>
            constant_add = -0.5 * np.log(2*np.pi) - avg_ln_error
            
            # '+=' modifies the array IN-PLACE. Zero new memory allocated
            result += constant_add 
            result *= num_datapoints

        return result