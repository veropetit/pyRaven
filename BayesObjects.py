import numpy as np
import h5py

## Develop by Robin Moore and Veronique Petit

###########################################
###########################################
###########################################

class chi:
    '''
    Class definition for the objects that will store the chi2s
    
    See __init__ for class content. 
    '''

    def __init__(self, data, beta_arr, Bpole_arr, phi_arr, incl, obsID):
        '''
        Initialization of a chi object

        :param data: a (n_beta, n_Bpole, n_phi) array containing the chi2s
        :param beta_arr: a 1D array with the grid values for the beta axis (in degree)
        :param Bpole_arr: a 1D array with the grid values for the Bpole axis (in Gauss)
        :param phi_arra: a 1D array with the grid values for the phase axis (in degree)
        :param incl: (float) the inclination value for this set of chi2s (in degree)
        :param obsID: (string or float) the observation ID for this set of chi2s
        '''
        self.data = data
        self.beta_arr = beta_arr
        self.Bpole_arr = Bpole_arr
        self.phi_arr = phi_arr
        self.obsID = obsID
        self.incl = incl

    def write(self, fname):
        '''
        Class function to write the chi object to a h5 file
        
        :param fname: (string) the path/name of the h5 file to be created
        '''
        with h5py.File(fname, 'w') as f:
            f.create_dataset('data',data=self.data)
            f.create_dataset('beta_arr',data=self.beta_arr)
            f.create_dataset('Bpole_arr',data=self.Bpole_arr)
            f.create_dataset('phi_arr',data=self.phi_arr)
            f.attrs['incl'] = self.incl
            f.attrs['obsID'] = self.obsID


def create_chi(beta_arr, Bpole_arr, phi_arr, incl, obsID):
    '''
    Function to create an empty chi2 object. This is used in the loop.py
    The chi.data will be a np.zeros array of the approprate size for the beta_arr, Bpole_arr, and phi_arr given
    
    :param beta_arr: numpy 1D array with grid values for the beta axis (in degree)
    :param Bpole_arr: numpy 1D array with grid values for the Bpole axis (in Gauss)
    :param phi_arr: numpy 1D array with grid values for the phase axis (in degree)
    :param incl: (float) the inclination value for this set of chi2s (in degree)
    :param obsID: (string or float) the observation ID for this set of chi2s
    '''
    data = np.zeros((beta_arr.size, Bpole_arr.size, phi_arr.size))
    return(chi(data, beta_arr, Bpole_arr, phi_arr, incl, obsID))

def read_chi(fname):
    '''
    Function to read in a chi object from an h5 file
    
    :param fname: (string) the name of the h5 file
    '''
    with h5py.File(fname, 'r') as f:
        data = np.array(f['data'])
        beta_arr = np.array(f['beta_arr'])
        Bpole_arr = np.array(f['Bpole_arr'])
        phi_arr = np.array(f['phi_arr'])
        incl = f.attrs.get('incl')
        obsID = f.attrs.get('obsID')

    return(chi(data, beta_arr, Bpole_arr, phi_arr, incl, obsID))
