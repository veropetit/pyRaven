import numpy as np
import h5py
import matplotlib.pyplot as plt
import copy
from matplotlib.backends.backend_pdf import PdfPages
from . import data as rav_data

## Develop by Robin Moore and Veronique Petit

###########################################
###########################################
###########################################

def exp_check(lnP):
    '''
    Function with an automated check for nans in the expoential
    '''
    P = np.exp(lnP)
    if type(P) is np.ndarray:
        if False in np.isfinite(P):
            print('WARNING: nan or inf in the exp(P)')    
    else:
        if False in np.array([P]):
            print('WARNING: nan or inf in the exp(P)') 
    return(P)

def ln_mar_check(lnP_array, axis=None, verbose=False):
    '''
    Function to marginalize an array of ln(P) and return a ln(P).
    Note, this function does not know about the grid size. 
    Therefore the multiplication with the binsize should be done by the caller of this function. 

    :param array: the array of logarithmic probabilities to marginalize
    :param axis: (None) the axis information to pass to np.sum
    :param verbose: (True) outputs a warning for overflow/underflow/nans
    '''
    # normalizing the array by its maximum value.
    norm = lnP_array.max()
    if verbose:
        print('Max of lnP is: {}'.format(norm))
    P_array = np.exp(lnP_array - norm)
    ln_mar = np.log( np.sum(P_array, axis=axis) ) + norm

    if type(ln_mar) is np.ndarray:
        if False in np.isfinite(ln_mar):
            print('WARNING: nan or inf in the marginalization')    
    else:
        if False in np.array([ln_mar]):
            print('WARNING: nan or inf in the marginalization') 

    return(ln_mar)


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

    def plot(self, index_phi):
        fig, ax = plt.subplots(1,1)
        im = ax.pcolormesh(self.Bpole_arr, self.beta_arr, self.data[:,:,index_phi], 
                        cmap='Purples_r', vmin=0, vmax=np.max(self.data))
        co = plt.colorbar(im)
        co.ax.set_ylabel('chi2')
        ax.set_xlabel('Bpole (Gauss)')
        ax.set_ylabel('Beta (degree)')
        ax.set_title('Obs: {} incl: {:3.1f}, phi: {:3.1f}'.format(self.obsID, self.incl,self.phi_arr[index_phi]))
        return(fig, ax)

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

class lnP_odds:
    '''
    Class definition for objects that will store the ln likelihood for the odds ratios

    See __init__ for class content. 
    '''

    def __init__(self, data, beta_arr, Bpole_arr, phi_arr, incl_arr, obsID, bestChi):
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
        self.bestChi = bestChi

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
        f.attrs['bestChi'] = self.bestChi        

    def write(self, fname):
        '''
        Class function to write the lnLH_odds object to a h5 file
        
        :param fname: (string) the path/name of the h5 file to be created
        '''
        with h5py.File(fname, 'w') as f:
            self.writef(f)

    def get_deltas(self, ln=False):
        if ln == False:
            return(
                self.beta_arr[1]-self.beta_arr[0],
                self.Bpole_arr[1]-self.Bpole_arr[0],
                self.phi_arr[1]-self.phi_arr[0],
                self.incl_arr[1]-self.incl_arr[0],
            )
        else:
            return(
                np.log(self.beta_arr[1]-self.beta_arr[0]),
                np.log(self.Bpole_arr[1]-self.Bpole_arr[0]),
                np.log(self.phi_arr[1]-self.phi_arr[0]),
                np.log(self.incl_arr[1]-self.incl_arr[0]),
            )

    def normalize(self):
        '''
        Return a normalized version of self
        '''
        lnd_beta, lnd_Bpole, lnd_phi, lnd_incl = self.get_deltas(ln=True)

        ln_norm = ln_mar_check(self.data) + lnd_beta+lnd_Bpole+lnd_incl+lnd_phi 
        
        return(lnP_odds(self.data-ln_norm, self.beta_arr,self.Bpole_arr,self.phi_arr, self.incl_arr,self.obsID,self.bestChi))

    def mar_phase(self):
        '''
        Function to return the a marginalization of a lnLH object over phase, 
        in a new ln_odds object. 
        '''

        lnd_beta, lnd_Bpole, lnd_phi, lnd_incl = self.get_deltas(ln=True)

        ln_post_mar = ln_mar_check(self.data, axis=(2))+ lnd_phi

        #create a ln_odds object to return
        ln_post = lnP_mar(ln_post_mar,self.beta_arr,self.Bpole_arr,self.incl_arr, self.obsID)
        return(ln_post)

    def apply_all_priors(self, flat=False, Jeff_B=None):
        '''
        Function to apply the all priors to a LH ODDS object. Returns a LH ODDS object that is NOT normalized. 

        :param flat: (False) If set to true, a flat prior will be used for Bpole and incl. 
        '''
        # I could use numpy broadcasting in numpy for that, 
        # but considering that we are only doing this operation a few times
        # using a loop does not add that much more time
        # and it makes the code more readable. 
        lnpost = copy.deepcopy(self)
        ln_prior_beta= np.log(get_prior_beta(self.beta_arr))
        ln_prior_Bpole = np.log(get_prior_Bpole(self.Bpole_arr, flat=flat, Jeff_B=Jeff_B))
        ln_prior_phi = np.log(get_prior_phi(self.phi_arr))
        ln_prior_incl = np.log(get_prior_incl(self.incl_arr, flat=flat))
        for beta, lnp_beta in enumerate(ln_prior_beta):
            for Bpole, lnp_Bpole in enumerate(ln_prior_Bpole):
                for phi, lnp_phi in enumerate(ln_prior_phi):
                    for incl, lnp_incl in enumerate(ln_prior_incl):
                        lnpost.data[beta, Bpole, phi, incl] += (lnp_beta+lnp_Bpole+lnp_phi+lnp_incl)
        return(lnpost)
    
    def apply_phase_priors(self):
        '''
        Function to apply the phi prior (only) to a LH ODDS object. Returns a LH ODDS object that is NOT normalized. 
        This is used to apply the phi prior to the LH ODDS prior to marginalizing the LH over the phases. 
        '''
        # I could use numpy broadcasting in numpy for that, 
        # but considering that we are only doing this operation a few times
        # using a loop does not add that much more time
        # and it makes the code more readable. 
        lnpost = copy.deepcopy(self)
        ln_prior_phi = np.log(get_prior_phi(self.phi_arr))
        for phi, lnp_phi in enumerate(ln_prior_phi):
            lnpost.data[:, :, phi, :] += (lnp_phi)
        return(lnpost)

    def plot_mar(self, fig=None, ax=None, right=False, **kwargs):
        '''
        Function to plot the normalized 1D marginalization for a lnP_odds object. 

        :param fig: (None) an already created fig object (for two side-by-side graphs). 
            If None and right=False, it will create a single corner plot. 
            If None and right=True, it will create a double corner plot and plot the current to the right side. 
            To get the left corner plot, call plot_corner again, passing the fig and ax and set right 
        :param ax: (None) an already created ax object (for two side-by-side graphs)
        :param right: (False) is False, only one corner plot is displayed. If true, it will create a ax for a double corner plot.  
        '''

        if right == True:
            k = 2
        else:
            k=0

        if fig == None and right == False:
            fig, ax = plt.subplots(3,2, figsize=(6,6))
            ax[2,1+k].set_axis_off()
            ax[2,0+k].set_axis_off()


        if fig == None and right == True:
            fig, ax = plt.subplots(3,4, figsize=(12,6))
            ax[2,1].set_axis_off()
            ax[2,0].set_axis_off()

        lnd_beta, lnd_Bpole, lnd_phi, lnd_incl = self.get_deltas(ln=True)

        lnmar = ln_mar_check(self.normalize().data, axis=(1,2,3)) + lnd_Bpole + lnd_phi + lnd_incl
        ax[0,0+k].plot(self.beta_arr, exp_check(lnmar), **kwargs)
        ax[0,0+k].set_xlabel('beta')

        lnmar = ln_mar_check(self.normalize().data,axis=(0,2,3)) + lnd_beta + lnd_phi + lnd_incl
        ax[0,1+k].plot(self.Bpole_arr, exp_check(lnmar), **kwargs)
        ax[0,1+k].set_xlabel('Bpole')

        lnmar = ln_mar_check(self.normalize().data,axis=(0,1,3)) + lnd_beta + lnd_Bpole + lnd_incl
        ax[1,0+k].plot(self.phi_arr, exp_check(lnmar), **kwargs)
        ax[1,0+k].set_xlabel('phi')

        lnmar = ln_mar_check(self.normalize().data,axis=(0,1,2)) + lnd_beta + lnd_Bpole + lnd_phi
        ax[1,1+k].plot(self.incl_arr, exp_check(lnmar), **kwargs)
        ax[1,1+k].set_xlabel('incl')


        plt.tight_layout()
        return(fig, ax)

def read_lnP_odds(fname):
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
        bestChi = f.attrs.get('bestChi')

    return(lnP_odds(data, beta_arr, Bpole_arr, phi_arr, incl_arr, obsID, bestChi))

def create_empty_lnP_odds(beta_arr, Bpole_arr, phi_arr, incl_arr, obsID):
    '''
    Helper function to create an empty lnP_odds object. 
    The lnP_odds.data will be a np.zeros array of the approprate size for the beta_arr, Bpole_arr, and phi_arr given
    
    :param beta_arr: numpy 1D array with grid values for the beta axis (in degree)
    :param Bpole_arr: numpy 1D array with grid values for the Bpole axis (in Gauss)
    :param phi_arr: numpy 1D array with grid values for the phase axis (in degree)
    :param incl_arr: numpy 1D array with grid values for the inclination axis (in degree)
    :param obsID: (string or float) the observation ID for this set of chi2s
    '''
    data = np.zeros((beta_arr.size, Bpole_arr.size, phi_arr.size, incl_arr.size))
    return(lnP_odds(data, beta_arr, Bpole_arr, phi_arr, incl_arr, obsID, bestChi=0.0))

def create_lnLH_odds_from_chi(folder_path, param, datapacket, output_path=None):
    '''
    Function to calculate the lnLH_odds from a set of already calcualted chi2. (see REF)

    :param folder_path: the path of the folder that contains the chi2 files. Default is current directory
    :param param: the parameter object used to calculate the chi2
    :param datapacket: the datapacket used to calculate the chi2 (used to reconstruct the filenames)
    :paran output_path: the path fo the folder to output the files to. Default is current directory
    '''
    if folder_path==None:
        folder_path='.'

    if output_path==None:
        output_path='.'

    # one lnLH file per observation for the odds ratio. 

    Stokes = ['V', 'N1']

    # loop over the two stokes parameter used
    for S in Stokes:
        # loop over all observations
        for o in range(0,datapacket.nobs):
            lnLH = create_empty_lnP_odds(
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


            loc=np.where(lnLH.data/-.5==np.min(lnLH.data/-.5)) #finds the minimum chi value
            lnLH.bestChi=lnLH.data[loc[0],loc[1],loc[2],loc[3]][0]/-.5

            lnLH.data = lnLH.data+constant_term

            # write the LH object to disk
            lnLH.write('{}/lnLH_ODDS_{}_obs{}.h5'.format(output_path,S,o))

    return


#########################
#########################

class lnP_pars(lnP_odds):
    '''
    Class definition for the lnP_pars. This class inherits from lnP_odds.
    '''

    def __init__(self, data, beta_arr, Bpole_arr, phi_arr, incl_arr, obsID, noise_arr):
        ## calling the constructor of the parent class
        super().__init__(data, beta_arr, Bpole_arr, phi_arr, incl_arr, obsID,bestChi=0.0)
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

    def find_best(self):
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

    def get_deltas(self, ln=False):
        if ln == False:
            return(
                self.beta_arr[1]-self.beta_arr[0],
                self.Bpole_arr[1]-self.Bpole_arr[0],
                self.phi_arr[1]-self.phi_arr[0],
                self.incl_arr[1]-self.incl_arr[0],
                self.noise_arr[1]-self.noise_arr[0]
            )
        else:
            return(
                np.log(self.beta_arr[1]-self.beta_arr[0]),
                np.log(self.Bpole_arr[1]-self.Bpole_arr[0]),
                np.log(self.phi_arr[1]-self.phi_arr[0]),
                np.log(self.incl_arr[1]-self.incl_arr[0]),
                np.log(self.noise_arr[1]-self.noise_arr[0])
            )

    def plot_mar(self, fig=None, ax=None, right=False, **kwargs):
        '''
        Function to plot the 1D marginalization for a lnP_pars object. 

        :param fig: (None) an already created fig object (for two side-by-side graphs). 
            If None and right=False, it will create a single corner plot. 
            If None and right=True, it will create a double corner plot and plot the current to the right side. 
            To get the left corner plot, call plot_corner again, passing the fig and ax and set right 
        :param ax: (None) an already created ax object (for two side-by-side graphs)
        :param right: (False) is False, only one corner plot is displayed. If true, it will create a ax for a double corner plot.  
        '''

        if right == True:
            k = 2
        else:
            k=0

        if fig == None and right == False:
            fig, ax = plt.subplots(3,2, figsize=(6,6))
            ax[2,1+k].set_axis_off()

        if fig == None and right == True:
            fig, ax = plt.subplots(3,4, figsize=(12,6))
            ax[2,1+k].set_axis_off()

        lnd_beta, lnd_Bpole, lnd_phi, lnd_incl, lnd_noise = self.get_deltas(ln=True)

        lnmar = ln_mar_check(self.data, axis=(1,2,3,4)) + lnd_Bpole + lnd_phi + lnd_incl + lnd_noise
        ax[0,0+k].plot(self.beta_arr, exp_check(lnmar), **kwargs)
        ax[0,0+k].set_xlabel('beta')
        ax[0,0+k].set_xlim(0,180)

        lnmar = ln_mar_check(self.data,axis=(0,2,3,4)) + lnd_beta + lnd_phi + lnd_incl + lnd_noise
        ax[0,1+k].plot(self.Bpole_arr, exp_check(lnmar), **kwargs)
        ax[0,1+k].set_xlabel('Bpole')

        lnmar = ln_mar_check(self.data,axis=(0,1,3,4)) + lnd_beta + lnd_Bpole + lnd_incl + lnd_noise
        ax[1,0+k].plot(self.phi_arr, exp_check(lnmar), **kwargs)
        ax[1,0+k].set_xlabel('phi')
        ax[1,0+k].set_xlim(0,360)


        lnmar = ln_mar_check(self.data,axis=(0,1,2,4)) + lnd_beta + lnd_Bpole + lnd_phi + lnd_noise
        ax[1,1+k].plot(self.incl_arr, exp_check(lnmar), **kwargs)
        ax[1,1+k].set_xlabel('incl')
        ax[1,1+k].set_xlim(0,180)


        lnmar = ln_mar_check(self.data,axis=(0,1,2,3)) + lnd_beta + lnd_Bpole + lnd_phi + lnd_incl
        ax[2,0+k].plot(self.noise_arr, exp_check(lnmar), **kwargs)
        ax[2,0+k].set_xlabel('scale noise')


        plt.tight_layout()
        return(fig, ax)

    def plot_prior(self, fig=None, ax=None, right=False, Jeff_B=None, **kwargs):
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

        d_beta, d_Bpole, d_phi, d_incl, d_noise = self.get_deltas(ln=False)
  
        prior = get_prior_beta(self.beta_arr)
        ax[0,0+k].plot(self.beta_arr, prior/(np.sum(prior)*d_beta) , **kwargs)
        if overplot == False: ax[0,0+k].set_xlabel('beta')

        prior = get_prior_Bpole(self.Bpole_arr, Jeff_B=Jeff_B)
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

    def apply_all_priors(self, Jeff_B=None):
        '''
        Function to apply all the priors to a LH PARS object. Returns a LH PARS object that is NOT normalized. 
        '''
        # I could use numpy broadcasting in numpy for that, 
        # but considering that we are only doing this operation a few times
        # using a loop does not add that much more time
        # and it makes the code more readable. 
        lnpost = copy.deepcopy(self)
        ln_prior_beta= np.log(get_prior_beta(self.beta_arr))
        ln_prior_Bpole = np.log(get_prior_Bpole(self.Bpole_arr, Jeff_B=Jeff_B))
        ln_prior_phi = np.log(get_prior_phi(self.phi_arr))
        ln_prior_incl = np.log(get_prior_incl(self.incl_arr))
        ln_prior_noise = np.log(get_prior_noise(self.noise_arr))
        for beta, lnp_beta in enumerate(ln_prior_beta):
            for Bpole, lnp_Bpole in enumerate(ln_prior_Bpole):
                for phi, lnp_phi in enumerate(ln_prior_phi):
                    for incl, lnp_incl in enumerate(ln_prior_incl):
                        for noise, lnp_noise in enumerate(ln_prior_noise):
                            lnpost.data[beta, Bpole, phi, incl, noise] += (lnp_beta+lnp_Bpole+lnp_phi+lnp_incl+lnp_noise)
        return(lnpost)
    
    def apply_phase_noise_priors(self):
        '''
        Function to apply the phi and noise priors to a LH PARS object. Returns a LH PARS object that is NOT normalized. 

        This is used to apply the phi and noise prior to the likelihood prior to marginalizing the LH. 
        '''
        # I could use numpy broadcasting in numpy for that, 
        # but considering that we are only doing this operation a few times
        # using a loop does not add that much more time
        # and it makes the code more readable. 
        lnpost = copy.deepcopy(self)
        ln_prior_phi = np.log(get_prior_phi(self.phi_arr))
        ln_prior_noise = np.log(get_prior_noise(self.noise_arr))
        for phi, lnp_phi in enumerate(ln_prior_phi):
            for noise, lnp_noise in enumerate(ln_prior_noise):
                lnpost.data[:, :, phi, :, noise] += (lnp_phi+lnp_noise)
        return(lnpost)

    def mar_phase_noise(self):
        '''
        Function to return the a marginalization of a lnLH object over phase and noise scale, 
        in a new ln_post object. 
        '''

        lnd_beta, lnd_Bpole, lnd_phi, lnd_incl, lnd_noise = self.get_deltas(ln=True)

        ln_post_mar = ln_mar_check(self.data, axis=(2,4))+ lnd_phi + lnd_noise

        #create a ln_post object to return
        ln_post = lnP_mar(ln_post_mar,self.beta_arr,self.Bpole_arr,self.incl_arr, self.obsID)
        return(ln_post)

    def normalize(self):
        '''
        Return a normalized version of self
        '''
        lnd_beta, lnd_Bpole, lnd_phi, lnd_incl, lnd_noise = self.get_deltas(ln=True)

        ln_norm = ln_mar_check(self.data) + lnd_beta+lnd_Bpole+lnd_incl+lnd_phi+lnd_noise 
        
        return(lnP_pars(self.data-ln_norm, self.beta_arr,self.Bpole_arr,self.phi_arr, self.incl_arr,self.obsID, self.noise_arr))


def read_lnP_pars(fname):
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
        


    return(lnP_pars(data, beta_arr, Bpole_arr, phi_arr, incl_arr, obsID, noise_arr))

def create_empty_lnP_pars(beta_arr, Bpole_arr, phi_arr, incl_arr, obsID, noise_arr):
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
    return(lnP_pars(data, beta_arr, Bpole_arr, phi_arr, incl_arr, obsID,noise_arr))

def create_lnLH_pars_from_chi(folder_path, param, datapacket, output_path):
    '''
    Function to calculate the lnLH_pars from a set of already calcualted chi2. (see REF)

    :param folder_path: the path of the folder that contains the chi2 files. Default is current directory
    :param param: the parameter object used to calculate the chi2. Need to include the 'grid' information
    :param datapacket: the datapacket used to calculate the chi2 (used to reconstruct the filenames)
    :param output_path: the path of the folder to output the files to. Default is current directory
    '''

    if folder_path==None:
        folder_path='.'

    if output_path==None:
        output_path='.'

    # one lnLH file per observation for the parameter estimation. 

    Stokes = ['V', 'N1']

    # loop over the two stokes parameter used
    for S in Stokes:
        # loop over all observations
        for o in range(0,datapacket.nobs):
            lnLH = create_empty_lnP_pars(
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
            lnLH.write('{}/lnLH_PARS_{}_obs{}.h5'.format(output_path,S,o))

    return

#####################
#####################

def get_prior_Bpole(Bpole_arr, flat=False, Jeff_B=100):
    '''
    Get the modified jeffreys or flat prior for the Bpole. 
    '''
    if flat == False:
        #Jeff_B = 100 #gauss
        prior_B = 1.0 / ( (Bpole_arr+ Jeff_B) * np.log(  (Jeff_B+Bpole_arr[-1])/Jeff_B ) )
        return(prior_B)
    else:
        prior_B = np.array( [ 1.0 / (Bpole_arr[-1] - Bpole_arr[0]) ] * Bpole_arr.size )
        return(prior_B)

def get_prior_incl(incl_arr, flat=False):
    '''
    Get the sin prior or flat prior for the inclination. 
    If incl = 0 or 180, the prior is formally zero. This causes problems in ln-format calculations.
    Normally, if the star has a measured vsini, i cannot be zero, and this issue is avoided 
    by the inclination grid not going to 0 and 180. 
    
    :param incl_arr: the inclination grid values (in degree)
    '''
    if flat == False:
        prior_incl = np.sin(incl_arr*np.pi/180) /2	# p(incl)dincl = sin(incl)dincl (P(i<io) = 1-cos(io))
        prior_incl = prior_incl * np.pi / 180 # This is because our grid is set in degrees (so we need prob per degree)
        return(prior_incl)
    else:
        prior_incl = np.array( [ 1.0 / (incl_arr[-1] - incl_arr[0]) ] * incl_arr.size )
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
class lnP_mar():
    '''
    Class to store (beta, Bpole, incl) probabilities 
    (so usually lnP_odds or lnP_pars marginalized for phase and noise scale parameter)
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
 
    def get_deltas(self, ln=False):
        if ln == False:
            return(
                self.beta_arr[1]-self.beta_arr[0],
                self.Bpole_arr[1]-self.Bpole_arr[0],
                self.incl_arr[1]-self.incl_arr[0],
            )
        else:
            return(
                np.log(self.beta_arr[1]-self.beta_arr[0]),
                np.log(self.Bpole_arr[1]-self.Bpole_arr[0]),
                np.log(self.incl_arr[1]-self.incl_arr[0]),
            )

    def normalize(self):
        '''
        Return a normalized version of self
        '''
        lnd_beta, lnd_Bpole, lnd_incl = self.get_deltas(ln=True)

        ln_norm = ln_mar_check(self.data) + lnd_beta+lnd_Bpole+lnd_incl 
        
        return(lnP_mar(self.data-ln_norm, self.beta_arr,self.Bpole_arr,self.incl_arr,self.obsID))

    def apply_priors(self, flat=False, Jeff_B=None):
        '''
        Function to apply the all priors to a lnP_mar object. Returns a lnP_mar object that is NOT normalized. 

        :param flat: (False) If set to true, a flat prior will be used for Bpole and incl. 
        '''
        # I could use numpy broadcasting in numpy for that, 
        # but considering that we are only doing this operation a few times
        # using a loop does not add that much more time
        # and it makes the code more readable. 
        lnpost = copy.deepcopy(self)
        ln_prior_beta= np.log(get_prior_beta(self.beta_arr))
        ln_prior_Bpole = np.log(get_prior_Bpole(self.Bpole_arr, flat=flat, Jeff_B=Jeff_B))
        ln_prior_incl = np.log(get_prior_incl(self.incl_arr, flat=flat))
        for beta, lnp_beta in enumerate(ln_prior_beta):
            for Bpole, lnp_Bpole in enumerate(ln_prior_Bpole):
                for incl, lnp_incl in enumerate(ln_prior_incl):
                    lnpost.data[beta, Bpole, incl] += (lnp_beta+lnp_Bpole+lnp_incl)
        return(lnpost)

    def get_mars1D(self):
        '''
        Function that returns a mars1D object with the marginalizations for Bpole, beta, and incl. 
        '''
        # Get the grid deltas:
        lnd_beta, lnd_Bpole, lnd_incl = self.get_deltas(ln=True)
        
        # Bpole
        lnmar = ln_mar_check(self.data, axis=(0,2)) + lnd_beta+lnd_incl
        Bpole = mar1D(self.Bpole_arr, exp_check(lnmar))

        #beta
        lnmar = ln_mar_check(self.data, axis=(1,2)) + lnd_Bpole+lnd_incl
        beta = mar1D(self.beta_arr, exp_check(lnmar))

        #incl
        lnmar = ln_mar_check(self.data, axis=(0,1)) + lnd_beta+lnd_Bpole
        incl = mar1D(self.incl_arr, exp_check(lnmar))

        return mars1D(Bpole, beta, incl)

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

        lnd_beta, lnd_Bpole, lnd_incl = self.get_deltas(ln=True)

        # Bpole
        lnmar = ln_mar_check(self.data, axis=(0,2)) + lnd_beta+lnd_incl
        ax[0,0+k].set_axis_on()
        ax[0,0+k].plot(self.Bpole_arr, exp_check(lnmar))

        #beta
        lnmar = ln_mar_check(self.data, axis=(1,2)) + lnd_Bpole+lnd_incl
        ax[2,2+k].set_axis_on()
        ax[2,2+k].set_xlim(0,180)
        ax[2,2+k].plot(self.beta_arr, exp_check(lnmar))

        #incl
        lnmar = ln_mar_check(self.data, axis=(0,1)) + lnd_beta+lnd_Bpole
        ax[1,1+k].set_axis_on()
        ax[1,1+k].set_xlim(0,180)
        ax[1,1+k].plot(self.incl_arr, exp_check(lnmar))

        # Bpole - beta
        lnmar = ln_mar_check(self.data, axis=2) + lnd_incl
        ax[2,0+k].set_axis_on()
        ax[2,0+k].set_ylim(0,180)
        ax[2,0+k].pcolormesh(self.Bpole_arr,self.beta_arr, exp_check(lnmar), shading='auto', cmap=cmap, vmin=0, vmax=exp_check(lnmar).max())

        # Bpole - incl
        lnmar = ln_mar_check(self.data, axis=0) + lnd_beta
        ax[1,0+k].set_axis_on()
        ax[1,0+k].set_ylim(0,180)
        ax[1,0+k].pcolormesh(self.Bpole_arr,self.incl_arr, exp_check(lnmar).T, shading='auto', cmap=cmap, vmin=0, vmax=exp_check(lnmar).max())

        # beta - incl
        lnmar = ln_mar_check(self.data, axis=1) + lnd_Bpole
        ax[2,1+k].set_axis_on()
        ax[2,1+k].set_ylim(0,180)
        ax[2,1+k].set_xlim(0,180)
        ax[2,1+k].pcolormesh(self.incl_arr,self.beta_arr, exp_check(lnmar), shading='auto', cmap=cmap, vmin=0, vmax=exp_check(lnmar).max())

        ax[2,2+k].set_xlabel('beta (deg)')
        ax[2,1+k].set_xlabel('incl (deg)')
        ax[2,0+k].set_xlabel('Bpole (G)')
        ax[2,0+k].set_ylabel('beta (deg)')
        ax[1,0+k].set_ylabel('incl (deg)')

        plt.tight_layout()
        return(fig, ax)

    def get_globalLH(self):
        '''
        Function to calculate the global LH associated with this marginalized 
        probability density. Important: the probability needs to be un-normalized
        (of the 'ODDS' type).

        :param self: a lnP_mar object with unormalized probability density
        :returns: 
        '''
        lnd_beta, lnd_Bpole, lnd_incl = self.get_deltas(ln=True)

        return ln_mar_check(self.data, axis=None) + lnd_beta + lnd_Bpole + lnd_incl

def read_lnP_mar(fname):
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

    return(lnP_mar(data, beta_arr, Bpole_arr, incl_arr, obsID))    


def calc_all_LHs(param, datapacket, chi_folder, output_path='.'):
    '''
    Wrapper function to calculate the most used probability products
    '''

    # 1. use the function to create the LH objects and files for each observations. 
    create_lnLH_odds_from_chi(chi_folder, param, datapacket, output_path)
    create_lnLH_pars_from_chi(chi_folder, param, datapacket, output_path)

    Stokes = ['V', 'N1']
    for S in Stokes:

        ###############################################
        ### Dealing with the odds first
        ###############################################            

        # 1. read in the first observation
        ln_P0 = read_lnP_odds('{}/lnLH_ODDS_{}_obs0.h5'.format(output_path,S))
        # 2. apply the phase prior (it's already flat)
        ln_P0 = ln_P0.apply_phase_priors()
        # 3. marginalize the LH over phase
        ln_P0 = ln_P0.mar_phase()

        # Keeping track of the obsID used
        obsID = [ln_P0.obsID]
        # if there are more than one observation:
        if datapacket.nobs > 1:
            for i in range(1,datapacket.nobs):  
                ln_P = read_lnP_odds('{}/lnLH_ODDS_{}_obs{}.h5'.format(output_path,S,i))
                obsID.append(ln_P.obsID)
                ln_P = ln_P.apply_phase_priors()
                ln_P = ln_P.mar_phase()
                # Multiply the marginalized LH
                ln_P0.data = ln_P0.data + ln_P.data

        # replace the obsID in the object in which the combination was done
        ln_P0.obsID = obsID
        # Write to disk
        ln_P0.write('{}/lnLH_ODDS_{}_mar_combined.h5'.format(output_path,S))

        ###############################################
        ### Dealing with the pars next
        ###############################################            

        # 1. read in the first observation
        ln_P0_flat = read_lnP_pars('{}/lnLH_PARS_{}_obs0.h5'.format(output_path,S))
        # 2. apply the phase and noise priors ()
        ln_P0_withprior = ln_P0_flat.apply_phase_noise_priors()
        # 3. marginalize the LH over phase
        ln_P0_flat = ln_P0_flat.mar_phase_noise()
        ln_P0_withprior = ln_P0_withprior.mar_phase_noise()
        # Keeping track of the obsID used
        obsID = [ln_P0_flat.obsID]
        # if there are more than one observation:
        if datapacket.nobs > 1:
            for i in range(1,datapacket.nobs):  
                # Read in next observation
                ln_P_flat = read_lnP_pars('{}/lnLH_PARS_{}_obs{}.h5'.format(output_path,S,i))
                obsID.append(ln_P_flat.obsID)
                # Apply the prior to phase and noise
                ln_P_withprior = ln_P_flat.apply_phase_noise_priors()
                # mar over phase and noise
                ln_P_flat = ln_P_flat.mar_phase_noise()
                ln_P_withprior = ln_P_withprior.mar_phase_noise()
                # Multiply the marginalized LH
                ln_P0_flat.data = ln_P0_flat.data + ln_P_flat.data
                ln_P0_withprior.data = ln_P0_withprior.data + ln_P_withprior.data

        print()

        # replace the obsID in the object in which the combination was done
        ln_P0_flat.obsID = obsID
        ln_P0_withprior.obsID = obsID
        # Write to disk
        ln_P0_flat.normalize().write('{}/lnLH_PARS_{}_mar_combined_flatprior.h5'.format(output_path,S))
        ln_P0_withprior.normalize().write('{}/lnLH_PARS_{}_mar_combined_withprior.h5'.format(output_path,S))

def combine_obs(nobs,folder_path, Jeff_B=100):
    '''
    DEPRECATED  Wrapper function to calculate a variety of posterior probabilities and
    combine the probabilities for multiple observtions. 

    :param nobs: Number of observations
    :param folder_path: Path of the lnLH and lnpost files. Default is current directory
    '''
    print('DEPRECATED')

    if folder_path==None:
        folder_path='.'

    ## Eventually, pass the param structure, and stick Jeff_B in it, instead of passing it. 

    Stokes = ['V', 'N1']

    for S in Stokes: 

        ###############################################
        ### Dealing with the parameter estimation first
        ###############################################
        # Note: for parameter estimation, we normalize the posterior probability densities
        # Therefore applying a flat prior is the same as simply not applying the priors
        # as all, because the normalization process will just get rid of the constant flat priors
        # 1. read in the first observation
        ln_LH = read_lnP_pars('{}/lnLH_PARS_{}_obs0.h5'.format(folder_path,S))

        # 2. write the normalized LH to disk
        ln_LH.normalize().write('{}/lnpost_PARS_noprior_{}_obs0.h5'.format(folder_path,S))

        # 3. Marginalize for the phase and noise scale (without priors), 
        ln_post_mar_noprior0 = ln_LH.mar_phase_noise()
        
        # 4. Write to disk the normalized verison
        ln_post_mar_noprior0.normalize().write('{}/lnpost_PARS_mar_noprior_{}_obs0.h5'.format(folder_path,S))

        # 5. multiply LH by the prior
        ln_post = ln_LH.apply_priors(Jeff_B=Jeff_B)
        
        # 6. write to disk the normalize version of the posterior for the observation
        ln_post.normalize().write('{}/lnpost_PARS_wprior_{}_obs0.h5'.format(folder_path,S))

        # 7. marginalize the observation for phi and noise scale
        ln_post_mar0 = ln_post.mar_phase_noise()
        
        # 8. Write to disk the normalize marginalized posterior for this observation. 
        ln_post_mar0.normalize().write('{}/lnpost_PARS_mar_wprior_{}_obs0.h5'.format(folder_path,S))
        
        # 9. Keeping track of the obsID used
        obsID = [ln_LH.obsID]
        # if there are more than one observation:
        if nobs > 1:
            for i in range(1,nobs):
                # steps 1-3 from above
                ln_LH = read_lnP_pars('{}/lnLH_PARS_{}_obs{}.h5'.format(folder_path,S,i))
                ln_LH.normalize().write('{}/lnpost_PARS_noprior_{}_obs{}.h5'.format(folder_path,S,i))
                ln_post_mar_noprior = ln_LH.mar_phase_noise()
                ln_post_mar_noprior.normalize().write('{}/lnpost_PARS_mar_noprior_{}_obs{}.h5'.format(folder_path,S,i))
                
                # combine the probabilities into the first observation data structure
                ln_post_mar_noprior0.data = ln_post_mar_noprior0.data + ln_post_mar_noprior.data
                
                # steps 5-8 from above
                ln_post = ln_LH.apply_priors(Jeff_B=Jeff_B)
                ln_post.normalize().write('{}/lnpost_PARS_wprior_{}_obs{}.h5'.format(folder_path,S,i))
                ln_post_mar = ln_post.mar_phase_noise()
                ln_post_mar.normalize().write('{}/lnpost_PARS_mar_wprior_{}_obs{}.h5'.format(folder_path,S,i))

                # combine the probabilities into the first observation data structure
                ln_post_mar0.data = ln_post_mar0.data + ln_post_mar.data
                
                # keep track of the obsID combined
                obsID.append(ln_LH.obsID)

        # replace the obsID in the object in which the combination was done
        ln_post_mar0.obsID = obsID
        ln_post_mar_noprior0.obsID = obsID

        #write to combined normalized posterior to disk
        ln_post_mar0.normalize().write('{}/lnpost_PARS_mar_wprior_{}.h5'.format(folder_path,S))  
        ln_post_mar_noprior0.normalize().write('{}/lnpost_PARS_mar_noprior_{}.h5'.format(folder_path,S))        

        ###############################################
        ### Now dealing with the odds ratio calculation
        ###############################################
        # Note: for odds ratio calculations, we need to need the exact value of the 
        # Global likelihood (so the un-normalized marginalzation of prior*likelihood)
        # Therefore we need to directly apply the flat priors 
        # unlike for the parameters estimations where we simply saved the normalized likelihood
        # to get the posterior with flat priors. 
        # 1. read in the first observation
        ln_LH = read_lnP_odds('{}/lnLH_ODDS_{}_obs0.h5'.format(folder_path,S))

        # 2. Apply flat prior and write the un-normalized LH to disk
        ln_post_noprior0 = ln_LH.apply_priors(flat=True)
        ln_post_noprior0.write('{}/lnpost_ODDS_flatprior_{}_obs0.h5'.format(folder_path,S))

        # 3. Marginalize for the phase and noise scale (with flat priors), 
        ln_post_mar_noprior0 = ln_post_noprior0.mar_phase()
        
        # 4. Write to disk the un-normalized verison
        ln_post_mar_noprior0.write('{}/lnpost_ODDS_mar_flatprior_{}_obs0.h5'.format(folder_path,S))

        # 5. multiply LH by the prior (non-flat version)
        ln_post = ln_LH.apply_priors(Jeff_B=Jeff_B)
        
        # 6. write to disk the un-normalized version of the posterior for the observation
        ln_post.write('{}/lnpost_ODDS_wprior_{}_obs0.h5'.format(folder_path,S))

        # 7. marginalize the observation for phi
        ln_post_mar0 = ln_post.mar_phase()
        
        # 8. Write to disk the un-normalize marginalized posterior for this observation. 
        ln_post_mar0.write('{}/lnpost_ODDS_mar_wprior_{}_obs0.h5'.format(folder_path,S))
        
        # 9. Keeping track of the obsID used
        obsID = [ln_LH.obsID]
        # if there are more than one observation:
        if nobs > 1:
            for i in range(1,nobs):
                # steps 1-3 from above
                ln_LH = read_lnP_odds('{}/lnLH_ODDS_{}_obs{}.h5'.format(folder_path,S,i))
                ln_post_noprior = ln_LH.apply_priors(flat=True)
                ln_post_noprior.write('{}/lnpost_ODDS_flatprior_{}_obs{}.h5'.format(folder_path,S,i))
                ln_post_mar_noprior = ln_post_noprior.mar_phase()
                ln_post_mar_noprior.write('{}/lnpost_ODDS_mar_flatprior_{}_obs{}.h5'.format(folder_path,S,i))
                
                # combine the probabilities into the first observation data structure
                ln_post_mar_noprior0.data = ln_post_mar_noprior0.data + ln_post_mar_noprior.data
                
                # steps 5-8 from above
                ln_post = ln_LH.apply_priors(Jeff_B=Jeff_B)
                ln_post.write('{}/lnpost_ODDS_wprior_{}_obs{}.h5'.format(folder_path,S,i))
                ln_post_mar = ln_post.mar_phase()
                ln_post_mar.write('{}/lnpost_ODDS_mar_wprior_{}_obs{}.h5'.format(folder_path,S,i))

                # combine the probabilities into the first observation data structure
                ln_post_mar0.data = ln_post_mar0.data + ln_post_mar.data
                
                # keep track of the obsID combined
                obsID.append(ln_LH.obsID)

        # replace the obsID in the object in which the combination was done
        ln_post_mar0.obsID = obsID
        ln_post_mar_noprior0.obsID = obsID

        #write to combined normalized posterior to disk
        ln_post_mar0.write('{}/lnpost_ODDS_mar_wprior_{}.h5'.format(folder_path,S))  
        ln_post_mar_noprior0.write('{}/lnpost_ODDS_mar_flatprior_{}.h5'.format(folder_path,S))        


    return()

def create_odds_dict(nobs):
    '''
    Helper function that creates a list of dictionaries to store the odds ratio results. 
    '''
    d = {'Obs':'',
         'V ln(GLH_M0)':'',
         'N1 ln(GLH_M0)':'',
         'V ln(GLH_M1)':'',
         'N1 ln(GLH_M1)':'',
         'log10 Odds V':'',
         'log10 Odds N1':''}
    d_list = []
    # A deep copy was needed overwise doing
    # return [d]*(nobs+1) was linking all the values together. 
    for i in range(0,nobs+1):
        d_list.append(copy.deepcopy(d))
    return d_list

def get_all_odds(datapacket, param, folder_path=None):
    '''
    Function to use the standard data products to get the odds ratios
    '''
    if folder_path==None:
        folder_path='.'

    nobs = datapacket.nobs
    odds_dict = create_odds_dict(nobs)
    odds_dict_flat = create_odds_dict(nobs)
    global_LH = datapacket.cutfit.get_globalLH_M0()

    Stokes = ['V', 'N1']

    for i in range(0,nobs):

        odds_dict[i]['Obs']='Observation {}'.format(i)
        odds_dict_flat[i]['Obs']='Observation {}'.format(i)

        for s,S in enumerate(Stokes):

            odds_dict[i]['{} ln(GLH_M0)'.format(S)] = global_LH[s][i]
            odds_dict_flat[i]['{} ln(GLH_M0)'.format(S)] = global_LH[s][i]

            # Open the LH odds for the observation:
            lnP = read_lnP_odds('{}/lnLH_ODDS_{}_obs{}.h5'.format(folder_path,S,i))
            
            # Apply the flat prior, mar over phase to create a ln_mar object, get the global LH
            GLH = lnP.apply_all_priors(flat=False, Jeff_B=param['grid']['Jeff_B']).mar_phase().get_globalLH()
            odds_dict[i]['{} ln(GLH_M1)'.format(S)] = GLH
            odds_dict[i]['log10 Odds {}'.format(S)] = (global_LH[s][i] - GLH)*np.log10(np.exp(1.0))

            # Now apply the flat priors instead
            GLH = lnP.apply_all_priors(flat=True).mar_phase().get_globalLH()
            odds_dict_flat[i]['{} ln(GLH_M1)'.format(S)] = GLH
            odds_dict_flat[i]['log10 Odds {}'.format(S)] = (global_LH[s][i] - GLH)*np.log10(np.exp(1.0))

    # The last row in the list is for the combined obs info
    odds_dict[nobs]['Obs']='Combined'
    odds_dict_flat[nobs]['Obs']='Combined'

    for s,S in enumerate(Stokes):

        GLH_M0 = np.sum(global_LH[s]) # getting the joint GLH for M0
        odds_dict[nobs]['{} ln(GLH_M0)'.format(S)] = GLH_M0
        odds_dict_flat[nobs]['{} ln(GLH_M0)'.format(S)] = GLH_M0

        # Read in the combined likelihood 
        lnP = read_lnP_mar('{}/lnLH_ODDS_{}_mar_combined.h5'.format(folder_path,S))

        # compute the global likelihood with prior
        GLH = lnP.apply_priors(flat=False, Jeff_B=param['grid']['Jeff_B']).get_globalLH()
        odds_dict[nobs]['{} ln(GLH_M1)'.format(S)] = GLH
        odds_dict[nobs]['log10 Odds {}'.format(S)] = (GLH_M0 - GLH)*np.log10(np.exp(1.0))

        # compute the global likelohood with flat prior
        GLH = lnP.apply_priors(flat=True).get_globalLH()
        odds_dict_flat[nobs]['{} ln(GLH_M1)'.format(S)] = GLH
        odds_dict_flat[nobs]['log10 Odds {}'.format(S)] = (GLH_M0 - GLH)*np.log10(np.exp(1.0))

    return(odds_dict, odds_dict_flat)

def overview_plots(datapacket, param, folder_path=None):
    '''
    Function to create a PDF with overview plots of the probabilities. 
    This function assumed that the files created by the combine_obs function are in the current directory. 

    :param datapacket: the datapacket used
    :param param: the param object used
    :param folder_path: Path of the lnLH files. Default is current directory
    '''
    
    if folder_path==None:
        folder_path='.'

    nobs = datapacket.nobs
    Jeff_B = param['grid']['Jeff_B']

    with PdfPages('{}/post_summary.pdf'.format(folder_path)) as pdf:

        # For each observation, the full parameter 1D marginalization
        for i in range(0,nobs):

            #1. Get the lnLH_pars 
            lnP = read_lnP_pars('{}/lnLH_PARS_V_obs{}.h5'.format(folder_path,i))

            #1. Plot the priors 
            fig, ax = lnP.plot_prior(right=True, Jeff_B=Jeff_B, 
                                     c='orchid', alpha=0.5, lw=5 )
            fig, ax = lnP.plot_prior(fig=fig, ax=ax, right=False, Jeff_B=Jeff_B,
                                     c='orchid', alpha=0.5, lw=5, label='Normalized Prior' )

            ##### a. Stokes V

            # 2. The posterior with flat prior (aka no prior, with scale noise)
            fig, ax = lnP.normalize().plot_mar(fig=fig, ax=ax, right=False, 
                                c='k',ls='dotted', lw=4, label='Post, Flat prior, \nwith scale noise')
            
            # 3. The posterior (aka with prior and scale noise)
            fig, ax = lnP.apply_all_priors(Jeff_B=Jeff_B).normalize() \
                    .plot_mar(fig=fig, ax=ax, right=False, 
                                c='k', lw=4, label='Post, With prior, \nwith scale noise')

            #=> reading in the odds LH            
            lnP = read_lnP_odds('{}/lnLH_ODDS_V_obs{}.h5'.format(folder_path,i))

            # 4. The posterior with flat prior without scale noise
            fig, ax = lnP.normalize().plot_mar(fig=fig, ax=ax, right=False, 
                                c='0.75',ls='dotted', label='Post, Flat prior,\n no scale noise')                    
            
            # 5. The posterior with prior without scale noise
            fig, ax = lnP.apply_all_priors(Jeff_B=Jeff_B).normalize() \
                .plot_mar(fig=fig, ax=ax, right=False, 
                                c='0.75', label='Post, With prior,\n no scale noise')                    


            ##### b. Null
            lnP = read_lnP_pars('{}/lnLH_PARS_N1_obs{}.h5'.format(folder_path,i))

            # 2. The posterior with flat prior (aka no prior, with scale noise)
            fig, ax = lnP.normalize().plot_mar(fig=fig, ax=ax, right=True, 
                                c='k',ls='dotted', lw=4)

            # 3. The posterior (aka with prior and scale noise)
            fig, ax = lnP.apply_all_priors(Jeff_B=Jeff_B).normalize() \
                    .plot_mar(fig=fig, ax=ax, right=True, 
                                c='k',lw=4)

            #=> reading in the odds LH            
            lnP = read_lnP_odds('{}/lnLH_ODDS_N1_obs{}.h5'.format(folder_path,i))

            # 4. The posterior with flat prior without scale noise
            fig, ax = lnP.normalize().plot_mar(fig=fig, ax=ax, right=True, 
                                c='0.75',ls='dotted')                    
            
            # 5. The posterior with prior without scale noise
            fig, ax = lnP.apply_all_priors(Jeff_B=Jeff_B).normalize() \
                .plot_mar(fig=fig, ax=ax, right=True, 
                                c='0.75')                    
            
            h, l = ax[0,0].get_legend_handles_labels()
            ax[2,1].legend(h, l, borderaxespad=0)
            ax[2,1].axis("off")

            # add labels and titles
            ax[0,0].set_title('Stokes V')
            ax[0,2].set_title('Null')
            for item in ax.flatten():
                item.set_ylim(bottom=0.0)
            fig.suptitle("Observation {}".format(i), fontsize=16)
            fig.tight_layout()
            fig.subplots_adjust(top=0.88)
            pdf.savefig()

        ### No scale noise, flat prior    
        lnP = read_lnP_mar('{}/lnLH_ODDS_N1_mar_combined.h5'.format(folder_path)).normalize()
        fig, ax = lnP.plot_corner(right=True)
        lnP = read_lnP_mar('{}/lnLH_ODDS_V_mar_combined.h5'.format(folder_path)).normalize()
        fig, ax = lnP.plot_corner(fig=fig, ax=ax, right=False)
        ax[0,0].set_title('Stokes V')
        ax[0,3].set_title('Null')
        for item in ax.flatten():
            item.set_ylim(bottom=0)
        fig.suptitle("Combined observations, without scale noise, flat prior", fontsize=16)
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        pdf.savefig()

        ### With scale noise, flat prior    
        lnP = read_lnP_mar('{}/lnLH_PARS_N1_mar_combined_flatprior.h5'.format(folder_path)).normalize()
        fig, ax = lnP.plot_corner(right=True)
        lnP = read_lnP_mar('{}/lnLH_PARS_V_mar_combined_flatprior.h5'.format(folder_path)).normalize()
        fig, ax = lnP.plot_corner(fig=fig, ax=ax, right=False)
        ax[0,0].set_title('Stokes V')
        ax[0,3].set_title('Null')
        for item in ax.flatten():
            item.set_ylim(bottom=0)
        fig.suptitle("Combined observations, with scale noise, flat prior", fontsize=16)
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        pdf.savefig()

        ### Without scale noise, with prior    
        lnP = read_lnP_mar('{}/lnLH_ODDS_N1_mar_combined.h5'.format(folder_path)).apply_priors(Jeff_B=Jeff_B).normalize()
        fig, ax = lnP.plot_corner(right=True)
        lnP = read_lnP_mar('{}/lnLH_ODDS_V_mar_combined.h5'.format(folder_path)).apply_priors(Jeff_B=Jeff_B).normalize()
        fig, ax = lnP.plot_corner(fig=fig, ax=ax, right=False)
        ax[0,0].set_title('Stokes V')
        ax[0,3].set_title('Null')
        for item in ax.flatten():
            item.set_ylim(bottom=0)
        fig.suptitle("Combined observations, without scale noise, with prior", fontsize=16)
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        pdf.savefig()

        ### With scale noise, with prior    
        lnP = read_lnP_mar('{}/lnLH_PARS_N1_mar_combined_withprior.h5'.format(folder_path)).apply_priors(Jeff_B=Jeff_B).normalize()
        fig, ax = lnP.plot_corner(right=True)
        lnP = read_lnP_mar('{}/lnLH_PARS_V_mar_combined_withprior.h5'.format(folder_path)).apply_priors(Jeff_B=Jeff_B).normalize()
        fig, ax = lnP.plot_corner(fig=fig, ax=ax, right=False)
        ax[0,0].set_title('Stokes V')
        ax[0,3].set_title('Null')
        for item in ax.flatten():
            item.set_ylim(bottom=0)
        fig.suptitle("Combined observations, with scale noise, with prior", fontsize=16)
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        pdf.savefig()

    return()

class mar1D():
    '''
    Class to store a 1D marginalization over a parameter

    Provides functions to compute credible regions and plot them. 
    '''

    def __init__(self, x, mar):
        '''
        Initialization of a mar1D object

        :param x: the array with the parameter values
        :param mar: the array with the marginalized PDF
        '''
        self.x = x
        self.mar = mar

    def __getitem__(self, key):
        """
        Returns a mar1D object with only the values at the specified index(s)

        :param key: the index or slice being checked
        :rtype: mar1D
        """
        x = self.x[key]
        mar = self.mar[key]
        return mar1D(x, mar)
    
    def __setitem__(self, key, newval):
        """
        Sets all values of the mar1D at the specified location equal
        to the input values.

        :param key: the index or slice being overwritten
        :param newval: Spectrum whose values are to replace the overwritten ones
        """
        if not(isinstance(newval, mar1D)):
            raise TypeError()
        else:
            self.x[key] = newval.x
            self.mar[key] = newval.mar

    def __len__(self):
        return len(self.x)
    
    def fwrite(self, f):
        '''
        Helper function to add a mar1D object to a h5 file

        :param f: the h5 object being written
        '''
        f.create_dataset('x', data = self.x)
        f.create_dataset('mar', data = self.mar)

        return

    def _interpolate_normalize(self, refinement):
        '''
        This function normalizes the distribution given some refinement factor.

        :param refinement: int, factor by which data is interpolated over
    
        :return interp_x_range: array, new x range
        :return norm_interp_y_range: array, normalized y range
        :rtype mar1D:
        '''
        ## INTERPOLATE VALUES
        n = len(self.x) * refinement # number of bins
        dn = (self.x[-1] - self.x[0]) / n # width of bin
        interp_x_range = np.linspace(self.x[0], self.x[-1], num = n ) # range x data will be interpolated
        interp_y_range = np.interp(interp_x_range, self.x, self.mar) # interpolation of y data

        ## NORNMALIZE DISTRIBUTION
        integral = np.sum(interp_y_range) * dn # integral of distribution
        norm_interp_y_range = interp_y_range / integral # to normalize integral, divide y values by integral
        #norm_integral = np.sum(norm_interp_y_range) * dn # just a check this equals 1, not returned anywhere

        return mar1D(interp_x_range, norm_interp_y_range)
    
def fread_mar(f):
    '''
    Helper function to read in a mar1D object from an open h5 file
    '''

    x = np.array(f['x'])
    mar = np.array(f['mar'])

    return mar1D(x, mar)

class mars1D():
    '''
    Sets of 1D marginalizations for Bpole, beta, and incl
    '''

    def __init__(self, Bpole, beta, incl):
        '''
        Initialize a mars1D object

        :param Bpole: a mar1D object for the Bpole marginalization
        :param beta: a mar1D object for the beta marginalization
        :param incl: a mar1D object for the incl marginalization
        '''

        self.Bpole = Bpole
        self.beta = beta
        self.incl = incl

    def write(self, fname):
        '''
        Function that writes the contents of a mars1D object to an h5 file

        :param f: h5 file being written
        :param lsds: LSDprofs object being stored in the file
        :param obs_names: (List of string) the IDs to be used for each observations
        '''

        with h5py.File(fname,'w') as f:
            #f.create_dataset('star_name', data = self.star_name)

            f_Bpole = f.create_group('Bpole')
            self.Bpole.fwrite(f_Bpole)

            f_beta = f.create_group('beta')
            self.beta.fwrite(f_beta)

            f_incl = f.create_group('incl')
            self.incl.fwrite(f_incl)

def read_mars1D(fname):
    '''
    Function that reads a h5 file that contains a mars1D object. 

    :param fname: name of the h5 file to read
    '''

    with h5py.File(fname) as f:
        #read the mar1D into objects
        Bpole = fread_mar(f['Bpole'])
        beta = fread_mar(f['beta'])
        incl = fread_mar(f['incl'])

    return mars1D(Bpole, beta, incl)


