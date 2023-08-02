import numpy as np
import h5py
import matplotlib.pyplot as plt
import copy
from matplotlib.backends.backend_pdf import PdfPages


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

    def get_deltas(self):
        return(
            self.beta_arr[1]-self.beta_arr[0],
            self.Bpole_arr[1]-self.Bpole_arr[0],
            self.phi_arr[1]-self.phi_arr[0],
            self.incl_arr[1]-self.incl_arr[0],
            self.noise_arr[1]-self.noise_arr[0],
        )

    def plot_mar(self, fig=None, ax=None, right=False, **kwargs):
        '''
        Function to plot the 1D marginalization for a LH PARS object. 

        :param fig: (None) an already created fig object (for two side-by-side graphs). 
            If None and right=False, it will create a single corner plot. 
            If None and right=True, it will create a double corner plot and plot the current to the right side. 
            To get the left corner plot, call plot_corner again, passing the fig and ax and set right 
        :param ax: (None) an already created ax object (for two side-by-side graphs)
        :param right: (False) is False, only one corner plot is displayed. If true, it will create a ax for a double corner plot.  
        '''
        if fig == None and right == False:
            fig, ax = plt.subplots(3,2, figsize=(6,6))

        if fig == None and right == True:
            fig, ax = plt.subplots(3,4, figsize=(12,6))
        if right == True:
            k = 2
        else:
            k=0
        # it's easy to overflow the LH when taking the exponential. 
        # Here, we don't care about the absolute value cause the whole thing
        # will be normalized in the end. So I am first normalizing by the 
        # max value (and will renormalized so that the sum of probability is 1 afterwards)
        LH = np.exp(self.data-self.data.max())
        LH = LH/np.sum(LH)

        d_beta, d_Bpole, d_phi, d_incl, d_noise = self.get_deltas()

        mar = np.sum(LH,axis=(1,2,3,4)) * d_Bpole * d_phi * d_incl * d_noise
        ax[0,0+k].plot(self.beta_arr, mar/(np.sum(mar)*d_beta), **kwargs)
        ax[0,0+k].set_xlabel('beta')

        mar = np.sum(LH,axis=(0,2,3,4)) * d_beta * d_phi * d_incl * d_noise
        ax[0,1+k].plot(self.Bpole_arr, mar/(np.sum(mar)*d_Bpole), **kwargs)
        ax[0,1+k].set_xlabel('Bpole')

        mar = np.sum(LH,axis=(0,1,3,4)) * d_beta * d_Bpole * d_incl * d_noise
        ax[1,0+k].plot(self.phi_arr, mar/(np.sum(mar)*d_phi), **kwargs)
        ax[1,0+k].set_xlabel('phi')

        mar = np.sum(LH,axis=(0,1,2,4)) * d_beta * d_Bpole * d_phi * d_noise
        ax[1,1+k].plot(self.incl_arr, mar/(np.sum(mar)*d_incl), **kwargs)
        ax[1,1+k].set_xlabel('incl')

        mar = np.sum(LH,axis=(0,1,2,3)) * d_beta * d_Bpole * d_phi * d_incl
        ax[2,0+k].plot(self.noise_arr, mar/(np.sum(mar)*d_noise), **kwargs)
        ax[2,0+k].set_xlabel('scale noise')

        ax[2,1+k].set_axis_off()

        plt.tight_layout()
        return(fig, ax)

    def plot_prior(self, fig=None, ax=None, right=False, **kwargs):
        '''
        Function to plot the 1D prior based on the grid definition of a LH PARS object. 

        :param fig: (None) an already created fig object (for two side-by-side graphs). 
            If None and right=False, it will create a single corner plot. 
            If None and right=True, it will create a double corner plot and plot the current to the right side. 
            To get the left corner plot, call plot_corner again, passing the fig and ax and set right 
        :param ax: (None) an already created ax object (for two side-by-side graphs)
        :param right: (False) is False, only one corner plot is displayed. If true, it will create a ax for a double corner plot.  
        '''
        if fig == None:
            overplot = False
        else:
            overplot = True
        
        if fig == None and right == False:
            fig, ax = plt.subplots(3,2, figsize=(6,6))
        if fig == None and right == True:
            fig, ax = plt.subplots(3,4, figsize=(12,6))
        if right == True:
            k = 2
        else:
            k=0

        d_beta, d_Bpole, d_phi, d_incl, d_noise = self.get_deltas()
  
        prior = get_prior_beta(self.beta_arr)
        ax[0,0+k].plot(self.beta_arr, prior/(np.sum(prior)*d_beta) , **kwargs)
        if overplot == False: ax[0,0+k].set_xlabel('beta')

        prior = get_prior_Bpole(self.Bpole_arr)
        ax[0,1+k].plot(self.Bpole_arr, prior/(np.sum(prior)*d_Bpole), **kwargs)
        if overplot == False: ax[0,1+k].set_xlabel('Bpole')

        prior = get_prior_phi(self.phi_arr)
        ax[1,0+k].plot(self.phi_arr, prior/(np.sum(prior)*d_phi), **kwargs)
        if overplot == False: ax[1,0+k].set_xlabel('phi')

        prior = get_prior_incl(self.incl_arr)
        ax[1,1+k].plot(self.incl_arr, prior/(np.sum(prior)*d_incl), **kwargs)
        if overplot == False: ax[1,1+k].set_xlabel('incl')

        prior = get_prior_noise(self.noise_arr)
        ax[2,0+k].plot(self.noise_arr, prior/(np.sum(prior)*d_noise), **kwargs)
        if overplot == False: ax[2,0+k].set_xlabel('scale noise')

        if overplot == False: ax[2,1+k].set_axis_off()

        plt.tight_layout()
        return(fig, ax) 

    def apply_priors(self):
        '''
        Function to apply the priors to a LH PARS object. Returns a LH PARS object that is NOT normalized. 
        '''
        # I could use numpy broadcasting in numpy for that, 
        # but considering that we are only doing this operation a few times
        # using a loop does not add that much more time
        # and it makes the code more readable. 
        post = copy.deepcopy(self)
        prior_beta= np.log(get_prior_beta(self.beta_arr))
        prior_Bpole = np.log(get_prior_Bpole(self.Bpole_arr))
        prior_phi = np.log(get_prior_phi(self.phi_arr))
        prior_incl = np.log(get_prior_incl(self.incl_arr))
        prior_noise = np.log(get_prior_noise(self.noise_arr))
        for beta, p_beta in enumerate(prior_beta):
            for Bpole, p_Bpole in enumerate(prior_Bpole):
                for phi, p_phi in enumerate(prior_phi):
                    for incl, p_incl in enumerate(prior_incl):
                        for noise, p_noise in enumerate(prior_noise):
                            post.data[beta, Bpole, phi, incl, noise] += (p_beta+p_Bpole+p_phi+p_incl+p_noise)
        return(post)

    def mar_phase_noise(self, normalize=True):
        '''
        Function to return the data array that has been marginalized over phase and noise scale. 
        Returns a lnpost object. 
        '''
        # it's easy to overflow the LH when taking the exponential. 
        # Here, we don't care about the absolute value cause the whole thing
        # will be normalized in the end. So I am first normalizing by the 
        # max value (and will renormalized so that the sum of probability is 1 afterwards)

        # Scaling the LH by a constant factor, to avoid overflows. 
        LH = np.exp(self.data-self.data.max())

        d_beta, d_Bpole, d_phi, d_incl, d_noise = self.get_deltas()

        post_mar = np.sum(LH, axis=(2,4))*d_phi*d_noise
        if normalize == True:
            # If the result is to be normalized, then we don't need to add the constant back
            total_sum = np.sum(LH)*d_beta* d_Bpole* d_phi* d_incl* d_noise
            ln_post_mar = np.log(post_mar)-np.log(total_sum)
        else:
            # If we don't want the resulting probability to be normalized (e.g. for the observations combination)
            # we need to add the constant back in there. 
            ln_post_mar = np.log(post_mar)+self.data.max()
        #create a ln_post object to return
        ln_post = lnpost(ln_post_mar,self.beta_arr,self.Bpole_arr,self.incl_arr, self.obsID)
        return(ln_post)


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
    Jeff_B = 100 #gauss
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

#####################
#####################
class lnpost():
    '''
    Class to store posterior probabilities (marginalized for phase and noise scale)
    '''

    def __init__(self, data, beta_arr, Bpole_arr, incl_arr, obsID):
        self.data = data
        self.beta_arr = beta_arr
        self.Bpole_arr = Bpole_arr
        self.incl_arr = incl_arr
        self.obsID = obsID

    def write(self, fname):
        '''
        Function to write a posterior object to a h5 files
        '''
        with h5py.File(fname, 'w') as f:
            f.create_dataset('data',data=self.data)
            f.create_dataset('beta_arr',data=self.beta_arr)
            f.create_dataset('Bpole_arr',data=self.Bpole_arr)
            f.create_dataset('incl_arr',data=self.incl_arr)
            f.create_dataset('obsID', data=self.obsID ) 
 
    def get_deltas(self):
        return(
            self.beta_arr[1]-self.beta_arr[0],
            self.Bpole_arr[1]-self.Bpole_arr[0],
            self.incl_arr[1]-self.incl_arr[0],
        )

    def normalize(self):
        '''
        Return a normalized version of self
        '''
        d_beta, d_Bpole, d_incl = self.get_deltas()
        P = np.exp(self.data - self.data.max())

        ln_total = np.log( np.sum(P)*d_beta*d_Bpole*d_incl )
        
        return(lnpost(np.log(P)-ln_total, self.beta_arr,self.Bpole_arr,self.incl_arr,self.obsID))

    def plot_corner(self, fig=None, ax=None, right=False):
        '''
        Function to display the corner plot for the posterior PARS. 

        :param fig: (None) an already created fig object (for two side-by-side graphs). 
            If None and right=False, it will create a single corner plot. 
            If None and right=True, it will create a double corner plot and plot the current to the right side. 
            To get the left corner plot, call plot_corner again, passing the fig and ax and set right 
        :param ax: (None) an already created ax object (for two side-by-side graphs)
        :param right: (False) is False, only one corner plot is displayed. If true, it will create a ax for a double corner plot.  
        '''

        if fig == None and right == False:
            fig, ax = plt.subplots(3,3, figsize=(6,6))
            for item in ax.flatten():
                item.set_axis_off()
        if fig == None and right == True:
            fig, ax = plt.subplots(3,6, figsize=(12,6))
            for item in ax.flatten():
                item.set_axis_off()
        if right == False:
            k = 0
        else:
            k = 3 

        cmap = copy.copy(plt.cm.Purples)
        cmap.set_bad('green', 1.0) # the masked values are masked (the 1 is for the alpha)
        #cmap.set_over('yellow', 1.0)
        #cmap.set_under('yellow', 1.0)

        d_beta, d_Bpole, d_incl = self.get_deltas()

        P = np.exp(self.data-self.data.max())

        # Bpole
        mar = np.sum(P, axis=(0,2))*d_beta*d_incl
        mar /= np.sum(mar)*d_Bpole
        ax[0,0+k].set_axis_on()
        ax[0,0+k].plot(self.Bpole_arr, mar)

        #beta
        mar = np.sum(P, axis=(1,2))*d_Bpole*d_incl
        mar /= np.sum(mar)*d_beta
        ax[2,2+k].set_axis_on()
        ax[2,2+k].plot(self.beta_arr, mar)

        #incl
        mar = np.sum(P, axis=(0,1))*d_beta*d_Bpole
        mar /= np.sum(mar)*d_incl
        ax[1,1+k].set_axis_on()
        ax[1,1+k].plot(self.incl_arr, mar)

        # Bpole - beta
        mar = np.sum(P, axis=2)*d_incl
        mar /= np.sum(mar)*d_beta*d_Bpole
        ax[2,0+k].set_axis_on()
        ax[2,0+k].pcolormesh(self.Bpole_arr,self.beta_arr, mar, shading='auto', cmap=cmap, vmin=0, vmax=mar.max())

        # Bpole - incl
        mar = np.sum(P, axis=0)*d_beta
        mar /= np.sum(mar)*d_Bpole*d_incl
        ax[1,0+k].set_axis_on()
        ax[1,0+k].pcolormesh(self.Bpole_arr,self.incl_arr, mar.T, shading='auto', cmap=cmap, vmin=0, vmax=mar.max())

        # beta - incl
        mar = np.sum(P, axis=1)*d_Bpole
        mar /= np.sum(mar)*d_beta*d_incl
        ax[2,1+k].set_axis_on()
        ax[2,1+k].pcolormesh(self.incl_arr,self.beta_arr, mar, shading='auto', cmap=cmap, vmin=0, vmax=mar.max())

        ax[2,2+k].set_xlabel('beta (deg)')
        ax[2,1+k].set_xlabel('incl (deg)')
        ax[2,0+k].set_xlabel('Bpole (G)')
        ax[2,0+k].set_ylabel('beta (deg)')
        ax[1,0+k].set_ylabel('incl (deg)')

        plt.tight_layout()
        return(fig, ax)

def read_lnpost(fname):
    '''
    Function to read in a lnpost object from an h5 file
    
    :param fname: (string) the name of the h5 file
    '''
    with h5py.File(fname, 'r') as f:
        data = np.array(f['data'])
        beta_arr = np.array(f['beta_arr'])
        Bpole_arr = np.array(f['Bpole_arr'])
        incl_arr = np.array(f['incl_arr'])
        obsID = np.array(f['obsID'])

    return(lnpost(data, beta_arr, Bpole_arr, incl_arr, obsID))    

def combine_obs(nobs):
    '''
    Wrapper function to calculate a variety of posterior probabilities and
    combine the probabilities for multiple observtions. 
    '''

    Stokes = ['V', 'N1']

    for S in Stokes: 

        ### Dealing with the parameter estimation first
        # 1. read in the first observation
        ln_LH = read_lnLH_pars('lnLH_PARS_{}_obs0.h5'.format(S))

        # 2. Write the normalized the post_noprior to disk
        ln_LH.mar_phase_noise(normalize=True).write('lnpost_PARS_noprior_{}_obs0.h5'.format(S))
        # now get the unnormalized post_noprior
        ln_post_noprior = ln_LH.mar_phase_noise(normalize=False)

        # 3. multiply by the prior
        ln_post = ln_LH.apply_priors()
            #write to disk
        ln_post.write('lnpost_PARS_fullpar_{}_obs0.h5'.format(S))
        # 4. write the normalized marginalize phi and noise scale
        ln_post.mar_phase_noise(normalize=True).write('lnpost_PARS_{}_obs0.h5'.format(S))
        # now get the unnormalized post_noprior
        ln_post_mar = ln_post.mar_phase_noise(normalize=False)


        # 4. Keeping track of the obsID used
        obsID = [ln_LH.obsID]
        # if there are more than one observation:
        if nobs > 1:
            for i in range(1,nobs):
                # 1. read in the LH for ith observation
                ln_LH = read_lnLH_pars('lnLH_PARS_{}_obs{}.h5'.format(S,i))
                # write the normalized post noprior
                ln_LH.mar_phase_noise(normalize=True).write('lnpost_PARS_noprior_{}_obs{}.h5'.format(S,i))
                # now get the unnormalized version
                mar = ln_LH.mar_phase_noise(normalize=False)
                # combine the probabilities
                ln_post_noprior.data = ln_post_noprior.data + mar.data
                
                # 2. multiply by the prior
                ln_post = ln_LH.apply_priors()
                ln_post.write('lnpost_PARS_fullpar_{}_obs{}.h5'.format(S,i))
                # write the normalized post noprior
                ln_post.mar_phase_noise(normalize=True).write('lnpost_PARS_{}_obs{}.h5'.format(S,i))
                # get the unnormalized version
                mar = ln_post.mar_phase_noise(normalize=False)
                # combine the probabilities
                ln_post_mar.data = ln_post_mar.data + mar.data
                # keep track of the obsID combined
                obsID.append(ln_LH.obsID)
        # replace the obsID in the object in which the combination was done
        ln_post_mar.obsID = obsID
        ln_post_noprior.obsID = obsID
        # normalize the results
        ln_post_mar = ln_post_mar.normalize()
        ln_post_noprior = ln_post_noprior.normalize()
        #write to disk
        ln_post_mar.write('lnpost_PARS_{}.h5'.format(S))  
        ln_post_noprior.write('lnpost_PARS_noprior_{}.h5'.format(S))        
      
    return()

def overview_plots(nobs):
    '''
    Function to create a PDF with overview plots of the probabilities. 
    This function assumed that the files created by the combine_obs function are in the current directory. 
    '''
    #Stokes = ['V', 'N1']
    with PdfPages('post_summary.pdf') as pdf:

        # For each observation, the full parameter 1D marginalization
        for i in range(0,nobs):
            # The likelihood alone (aka no prior, with scale noise)
            P = read_lnLH_pars('lnLH_PARS_N1_obs{}.h5'.format(i))
            fig, ax = P.plot_mar(right=True, c='k',ls='--')
            P = read_lnLH_pars('lnLH_PARS_V_obs{}.h5'.format(i))
            fig, ax = P.plot_mar(fig=fig, ax=ax, right=False, c='k',ls='--')            
            # The posterior (aka with prior and scale noise)
            P = read_lnLH_pars('lnpost_PARS_fullpar_N1_obs{}.h5'.format(i))
            fig, ax = P.plot_mar(fig=fig, ax=ax, right=True, c='k')
            P = read_lnLH_pars('lnpost_PARS_fullpar_V_obs{}.h5'.format(i))
            fig, ax = P.plot_mar(fig=fig, ax=ax, right=False, c='k')
            # over plot the priors
            fig, ax = P.plot_prior(fig=fig, ax=ax, right=True, c='orchid', alpha=0.5, lw=3 )
            fig, ax = P.plot_prior(fig=fig, ax=ax, right=False, c='orchid', alpha=0.5, lw=3 )
            # add labels and titles
            ax[0,0].set_title('Stokes V')
            ax[0,2].set_title('Null')
            for item in ax.flatten():
                item.set_ylim(bottom=0.0)
            fig.suptitle("Observation {}, scale noise, no prior (--) and with prior (solid)".format(i), fontsize=16)
            fig.tight_layout()
            fig.subplots_adjust(top=0.88)
            pdf.savefig()

        # For each observation, the corner plot of beta, Bpole, incl. 
        for i in range(0,nobs):
            # No prior, with scale noise
            P = read_lnpost('lnpost_PARS_noprior_N1_obs{}.h5'.format(i))
            fig, ax = P.plot_corner(right=True)
            P = read_lnpost('lnpost_PARS_noprior_V_obs{}.h5'.format(i))
            fig, ax = P.plot_corner(fig=fig, ax=ax, right=False)
            ax[0,0].set_title('Stokes V')
            ax[0,3].set_title('Null')
            fig.suptitle("Observation {}, scale noise, no prior".format(i), fontsize=16)
            for item in ax.flatten():
                item.set_ylim(bottom=0)
            fig.tight_layout()
            fig.subplots_adjust(top=0.88)
            pdf.savefig()
            # With prior, with scale noise
            P = read_lnpost('lnpost_PARS_N1_obs{}.h5'.format(i))
            fig, ax = P.plot_corner(right=True)
            P = read_lnpost('lnpost_PARS_V_obs{}.h5'.format(i))
            fig, ax = P.plot_corner(fig=fig, ax=ax, right=False)
            ax[0,0].set_title('Stokes V')
            ax[0,3].set_title('Null')
            fig.suptitle("Observation {}, scale noise, with prior".format(i), fontsize=16)
            for item in ax.flatten():
                item.set_ylim(bottom=0)
            fig.tight_layout()
            fig.subplots_adjust(top=0.88)
            pdf.savefig()

        P = read_lnpost('lnpost_PARS_noprior_N1.h5')
        fig, ax = P.plot_corner(right=True)
        P = read_lnpost('lnpost_PARS_noprior_V.h5')
        fig, ax = P.plot_corner(fig=fig, ax=ax, right=False)
        ax[0,0].set_title('Stokes V')
        ax[0,3].set_title('Null')
        for item in ax.flatten():
            item.set_ylim(bottom=0)
        fig.suptitle("Combined observations, scale noise, no prior".format(i), fontsize=16)
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        pdf.savefig()

        P = read_lnpost('lnpost_PARS_N1.h5')
        fig, ax = P.plot_corner(right=True)
        P = read_lnpost('lnpost_PARS_V.h5')
        fig, ax = P.plot_corner(fig=fig, ax=ax, right=False)
        ax[0,0].set_title('Stokes V')
        ax[0,3].set_title('Null')
        for item in ax.flatten():
            item.set_ylim(bottom=0)
        fig.suptitle("Combined observations, scale noise, with prior".format(i), fontsize=16)
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        pdf.savefig()




    return()

