from __future__ import annotations  # This is to allow using the class in the typeing 'hint' in the constructor

import numpy as np
import matplotlib.pyplot as plt

from . import BaseBayesObject as base

class Chi(base.BaseBayesObject):
    """
    Class definition for the objects that will store the Chi2 matrices.
    Inherits core slicing, math overloads, and integration features from BaseBayesObject.
    """    

    # 1. MUST define REQUIRED_COORDS to match the expected dimension ordering of the data array
    REQUIRED_COORDS = ['beta_coord', 'Bpole_coord', 'phi_coord']    

    # The chi are not stored in log
    PROB_IS_LOG = False

    def __init__(self, prob, beta_coord, Bpole_coord, phi_coord, incl, obsID):
        """
        Initialization of a Chi object.
        """
        # Forward the core array and labeled coordinates to the Base Class constructor.
        super().__init__(
            prob=prob, 
            beta_coord=beta_coord, 
            Bpole_coord=Bpole_coord, 
            phi_coord=phi_coord
        )
        
        # Store the extra scientific metadata attributes
        self.incl = incl
        self.obsID = obsID

    @classmethod
    def empty(cls, 
        beta_coord: np.typing.ArrayLike, 
        Bpole_coord: np.typing.ArrayLike, 
        phi_coord: np.typing.ArrayLike, 
        incl: float, 
        obsID: str | int
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
            incl=incl,
            obsID=obsID
        )

    @classmethod
    def read(cls, filename: str) -> Chi:
        """
        Read an HDF5 file and return a fully populated Chi object.
        """
        # Simply hand the filename up to the base class engine
        return super().read(filename)

    def plot(self, index_phi):
        '''Function to plot a slice of Chi'''
        # Validate the index
        #   1. is an integer
        if not isinstance(index_phi, (int, np.integer)):
            raise TypeError(
                f"Invalid index_phi type: {type(index_phi).__name__}. "
                f"The index must be a single integer."
            )
        #   2. is within range
        n_phi = len(self.phi_coord)
        if index_phi < 0 or index_phi >= n_phi:
            raise IndexError(
                f"Invalid index_phi={index_phi}. Must be between 0 and {n_phi - 1} "
                f"(phi range: {self.phi_coord[0]} to {self.phi_coord[-1]})."
            )

        fig, ax = plt.subplots(1,1)
        im = ax.pcolormesh(self.Bpole_coord, self.beta_coord, self.prob[:,:,index_phi], 
                        cmap='Purples_r', vmin=0, vmax=np.max(self.prob))
        co = plt.colorbar(im)
        co.ax.set_ylabel('Chi2')
        ax.set_xlabel('Bpole')
        ax.set_ylabel('Beta')
        ax.set_title('Obs: {} incl: {:3.1f}, phi: {:3.1f}'.format(self.obsID, self.incl,self.phi_coord[index_phi]))
        return(fig, ax)
    
class LnLikelihood(base.BaseBayesObject):
    '''
    Likelihood class for the odds ratios
    '''

    #-------------------------------------
    # 1. Properties & Initialization
    #-------------------------------------

    REQUIRED_COORDS = ['beta_coord', 'Bpole_coord', 'phi_coord', 'incl_coord']
    PROB_IS_LOG = True
    CLASS_ID = 'LnLikelihood'

    def __init__(
        self, 
        prob: np.ndarray, 
        beta_coord: np.typing.ArrayLike, 
        Bpole_coord: np.typing.ArrayLike, 
        phi_coord: np.typing.ArrayLike, 
        incl_coord: np.typing.ArrayLike,
        obsID: str | int
    ):
        # Pass everything safely up to the base class structure
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
        obsID: str | int
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
    
    #-------------------------------------
    # 3. Core Mathematical Backends
    #-------------------------------------

    @staticmethod
    def _calculate_ln_likelihood_from_chi(chi_array: np.ndarray, N_data: int, sigma:np.ndarray, scale_noise: float = 1.0) -> np.ndarray:
        """
        The single source of truth for the log-likelihood calculation.
        Computes the physics equation mapping Chi^2 to Ln(Likelihood) including scale noise.
        """

        constant_term = -N_data/2*np.log(2*np.pi)+np.sum(np.log(1/sigma))


        
        return 