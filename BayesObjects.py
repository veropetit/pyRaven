import numpy as np
import h5py
import matplotlib.pyplot as plt

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
        self.incl = incl
        self.obsID = obsID

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

#########################
#########################

class lnLH_odds:
    '''
    Class definition for objects that will store the ln likelihood for the odds ratios

    See __init__ for class content. 
    '''

    def __init__(self, data, beta_arr, Bpole_arr, phi_arr, incl_arr, obsID):
        '''
        Initialization of a LH_odd object

        :param data: a (n_beta, n_Bpole, n_phi, n_incl) array containing the lnLH values
        :param beta_arr: a 1D array with the grid values for the beta axis (in degree)
        :param Bpole_arr: a 1D array with the grid values for the Bpole axis (in Gauss)
        :param phi_arr: a 1D array with the grid values for the phase axis (in degree)
        :param incl_arr: a 1D array with the grid values for the inclination axis (in degree)
        :param obsID: (string or float) the observation ID for this set of chi2s
        '''
        self.data = data
        self.beta_arr = beta_arr
        self.Bpole_arr = Bpole_arr
        self.phi_arr = phi_arr
        self.incl_arr = incl_arr
        self.obsID = obsID

    def writef(self, f):
        '''
        Helper class function to create datasets in the passed h5 file object

        :param f: a h5 file object
        '''
        f.create_dataset('data',data=self.data)
        f.create_dataset('beta_arr',data=self.beta_arr)
        f.create_dataset('Bpole_arr',data=self.Bpole_arr)
        f.create_dataset('phi_arr',data=self.phi_arr)
        f.create_dataset('incl_arr',data=self.incl_arr)
        f.attrs['obsID'] = self.obsID        

    def write(self, fname):
        '''
        Class function to write the lnLH_odds object to a h5 file
        
        :param fname: (string) the path/name of the h5 file to be created
        '''
        with h5py.File(fname, 'w') as f:
            self.writef(f)

def read_lnLH_odds(fname):
    '''
    Function to read in a lnLH_odds object from an h5 file
    
    :param fname: (string) the name of the h5 file
    '''
    with h5py.File(fname, 'r') as f:
        data = np.array(f['data'])
        beta_arr = np.array(f['beta_arr'])
        Bpole_arr = np.array(f['Bpole_arr'])
        phi_arr = np.array(f['phi_arr'])
        incl_arr = np.array(f['incl_arr'])
        obsID = f.attrs.get('obsID')

    return(lnLH_odds(data, beta_arr, Bpole_arr, phi_arr, incl_arr, obsID))


def create_empty_lnLH_odds(beta_arr, Bpole_arr, phi_arr, incl_arr, obsID):
    '''
    Helper function to create an empty LH_odds object. 
    The lnLH_odds.data will be a np.zeros array of the approprate size for the beta_arr, Bpole_arr, and phi_arr given
    
    :param beta_arr: numpy 1D array with grid values for the beta axis (in degree)
    :param Bpole_arr: numpy 1D array with grid values for the Bpole axis (in Gauss)
    :param phi_arr: numpy 1D array with grid values for the phase axis (in degree)
    :param incl_arr: numpy 1D array with grid values for the inclination axis (in degree)
    :param obsID: (string or float) the observation ID for this set of chi2s
    '''
    data = np.zeros((beta_arr.size, Bpole_arr.size, phi_arr.size, incl_arr.size))
    return(lnLH_odds(data, beta_arr, Bpole_arr, phi_arr, incl_arr, obsID))

def create_lnLH_odds_from_chi(folder_path, param, datapacket):
    '''
    Function to calculate the lnLH_odds from a set of already calcualted chi2. (see REF)

    :param folder_path: the path of the folder that contains the chi2 files
    :param param: the parameter object used to calculate the chi2
    :param datapacket: the datapacket used to calculate the chi2 (used to reconstruct the filenames)
    '''

    # one LH file per observation for the odds ratio. 

    Stokes = ['V', 'N1']

    # loop over the two stokes parameter used
    for S in Stokes:
        # loop over all observations
        for o in range(0,datapacket.nobs):
            lnLH = create_empty_lnLH_odds(
                                    param['grid']['beta_grid'],
                                    param['grid']['Bpole_grid'],
                                    param['grid']['phase_grid'],
                                    param['grid']['incl_grid'],
                                    datapacket.obs_names[o]
            )

            # calculate the constant term for this observation
            N = datapacket.cutfit.lsds[o].vel.size
            if S == 'V':
                sigma = datapacket.cutfit.lsds[o].specSigV
            else:
                sigma = datapacket.cutfit.lsds[o].specSigN1
            constant_term = -N/2*np.log(2*np.pi)+np.sum(np.log(1/sigma))

            # loop over all inclination files
            for i in range(0,param['grid']['incl_grid'].size):

                filename = '{}/chi{}_i{}obs{}.h5'.format(folder_path, S, i, o )
                chi = read_chi(filename)
                
                lnLH.data[:,:,:,i] = -0.5*chi.data

            lnLH.data = lnLH.data+constant_term

            # write the LH object to disk
            lnLH.write('lnLH_ODDS_{}_obs{}.h5'.format(S,o))

    return


#########################
#########################

class lnLH_pars(lnLH_odds):
    '''
    Class definition for the lnLH_pars. This class inherits from lnLH_odds.
    '''

    def __init__(self, data, beta_arr, Bpole_arr, phi_arr, incl_arr, obsID, noise_arr):
        ## calling the constructor of the parent class
        super().__init__(data, beta_arr, Bpole_arr, phi_arr, incl_arr, obsID)
        # adding the noise_arr
        self.noise_arr = noise_arr

    def write(self, fname):
        '''
        Class function to write the lnLH_pars object to a h5 file
        
        :param fname: (string) the path/name of the h5 file to be created
        '''
        with h5py.File(fname, 'w') as f:
            # calling the write helper function from the parent class
            super().writef(f)
            # writting the noise_arr. 
            f.create_dataset('noise_arr',data=self.noise_arr)

    def find_bestLH(self):
        '''
        Function to return dictionary witht the parameters of the max likelihood
        (NEED TO: marginalized for the scale noise parameter)
        '''
        index = np.argmax(self.data, keepdims=True)
        params = {'beta':self.beta_arr[index[0]],
                  'Bpole':self.Bpole_arr[index[1]],
                  'phi':self.phi_arr[index[2]],
                  'incl':self.incl_arr[index[3]]}
        return(params)
    
    def plot_mar(self):
        fig, ax = plt.subplots(3,2)
        # it's easy to overflow the LH when taking the exponential. 
        # Here, we don't care about the absolute value cause the whole thing
        # will be normalized in the end. So I am first normalizing by the 
        # max value (and will renormalized so that the sum of probability is 1 afterwards)
        LH = np.exp(self.data-self.data.max())
        LH = LH/np.sum(LH)

        mar = np.sum(LH,axis=(1,2,3,4))
        ax[0,0].plot(self.beta_arr, mar/np.sum(mar))
        ax[0,0].set_xlabel('beta')

        mar = np.sum(LH,axis=(0,2,3,4))
        ax[0,1].plot(self.Bpole_arr, mar/np.sum(mar))
        ax[0,1].set_xlabel('Bpole')

        mar = np.sum(LH,axis=(0,1,3,4))
        ax[1,0].plot(self.phi_arr, mar/np.sum(mar))
        ax[1,0].set_xlabel('phi')

        mar = np.sum(LH,axis=(0,1,2,4))
        ax[1,1].plot(self.incl_arr, mar/np.sum(mar))
        ax[1,1].set_xlabel('incl')

        mar = np.sum(LH,axis=(0,1,2,3))
        ax[2,0].plot(self.noise_arr, mar/np.sum(mar))
        ax[2,0].set_xlabel('scale noise')

        plt.tight_layout()
        return(fig, ax)

def read_lnLH_pars(fname):
    '''
    Function to read in a lnLH_pars object from an h5 file
    
    :param fname: (string) the name of the h5 file
    '''
    with h5py.File(fname, 'r') as f:
        data = np.array(f['data'])
        beta_arr = np.array(f['beta_arr'])
        Bpole_arr = np.array(f['Bpole_arr'])
        phi_arr = np.array(f['phi_arr'])
        incl_arr = np.array(f['incl_arr'])
        obsID = f.attrs.get('obsID')
        noise_arr = np.array(f['noise_arr'])


    return(lnLH_pars(data, beta_arr, Bpole_arr, phi_arr, incl_arr, obsID, noise_arr))

def create_empty_lnLH_pars(beta_arr, Bpole_arr, phi_arr, incl_arr, obsID, noise_arr):
    '''
    Helper function to create an empty LH_odds object. 
    The lnLH_pars.data will be a np.zeros array of the approprate size for the beta_arr, Bpole_arr, phi_arr, and noise_arr given
    
    :param beta_arr: numpy 1D array with grid values for the beta axis (in degree)
    :param Bpole_arr: numpy 1D array with grid values for the Bpole axis (in Gauss)
    :param phi_arr: numpy 1D array with grid values for the phase axis (in degree)
    :param incl_arr: numpy 1D array with grid values for the inclination axis (in degree)
    :param obsID: (string or float) the observation ID for this set of chi2s
    :param noise_arr: numpy 1D array with grid values for the scale noise parameter (no units)
    '''
    data = np.zeros((beta_arr.size, Bpole_arr.size, phi_arr.size, incl_arr.size, noise_arr.size))
    return(lnLH_pars(data, beta_arr, Bpole_arr, phi_arr, incl_arr, obsID,noise_arr))

def create_lnLH_pars_from_chi(folder_path, param, datapacket):
    '''
    Function to calculate the lnLH_pars from a set of already calcualted chi2. (see REF)

    :param folder_path: the path of the folder that contains the chi2 files
    :param param: the parameter object used to calculate the chi2. Need to include the 'grid' information
    :param datapacket: the datapacket used to calculate the chi2 (used to reconstruct the filenames)
    '''

    # one lnLH file per observation for the parameter estimation. 

    Stokes = ['V', 'N1']

    # loop over the two stokes parameter used
    for S in Stokes:
        # loop over all observations
        for o in range(0,datapacket.nobs):
            lnLH = create_empty_lnLH_pars(
                                    param['grid']['beta_grid'],
                                    param['grid']['Bpole_grid'],
                                    param['grid']['phase_grid'],
                                    param['grid']['incl_grid'],
                                    datapacket.obs_names[o],
                                    param['grid']['noise_grid']
            )

            # calculate the constant term for this observation
            N = datapacket.cutfit.lsds[o].vel.size
            if S == 'V':
                sigma = datapacket.cutfit.lsds[o].specSigV
            else:
                sigma = datapacket.cutfit.lsds[o].specSigN1
            constant_term = -N/2*np.log(2*np.pi)+np.sum(np.log(1/sigma))

            # loop over all inclination files
            for i in range(0,param['grid']['incl_grid'].size):

                filename = '{}/chi{}_i{}obs{}.h5'.format(folder_path, S, i, o )
                chi = read_chi(filename)

                # loop over the scale noise values
                for b, val_b in enumerate(param['grid']['noise_grid']):
                
                    lnLH.data[:,:,:,i,b] = -0.5*val_b*chi.data + 0.5*N*np.log(val_b)

            lnLH.data = lnLH.data+constant_term

            # write the LH object to disk
            lnLH.write('lnLH_PARS_{}_obs{}.h5'.format(S,o))

    return

#####################
#####################

def get_prior_Bpole(Bpole_arr):
    '''
    Get the modified jeffreys prior for the Bpole. 
    '''
    Jeff_B = Bpole_arr[5]
    prior_B = 1.0 / ( (Bpole_arr+ Jeff_B) * np.log(  (Jeff_B+Bpole_arr[-1])/Jeff_B ) )
    return(prior_B)

def get_prior_incl(incl_arr):
    '''
    Get the sin prior for the inclination
    
    :param incl_arr: the inclination grid values (in degree)
    '''
    prior_incl = np.sin(incl_arr*np.pi/180) /2	# p(incl)dincl = sin(incl)dincl (P(i<io) = 1-cos(io))
    return(prior_incl)

def get_prior_beta(beta_arr):
    '''
    Get the flat prior for beta
    '''
    prior_beta = np.array( [ 1.0 / (beta_arr[-1] - beta_arr[0]) ] * beta_arr.size )
    return(prior_beta)

def get_prior_phi(phi_arr):
    '''
    Get the flat prior for phi
    '''
    prior_phi = np.array( [ 1.0 / (phi_arr[-1] - phi_arr[0]) ] * phi_arr.size )
    return(prior_phi)

def get_prior_noise(noise_arr):
    '''
    Get the Jeffreys prior for the scale noise parameter
    '''
    prior_noise = 1.0 / ( noise_arr* np.log(noise_arr[-1] / noise_arr[0]) )
    return(prior_noise)

