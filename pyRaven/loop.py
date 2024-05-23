import numpy as np
from scipy.special import erf
from astropy.convolution import Gaussian1DKernel, convolve
import astropy.convolution as conv

import time
from numba import njit
from numba.typed import List

from . import diskint as disk
from . import misc as rav_misc
from . import localV as rav_localV
from . import profileI as rav_profileI
from . import BayesObjects as rav_BayesObjects
from . import params as rav_params
from . import data as rav_data


'''
The input for both diskint.strong and diskint.weak is of the form:
genparam = {
    'lambda0':5000,    # the central wavelength of the transition
    'vsini':50.0,         # the projected rotational velocity
    'vdop':10.0,          # the thermal broadening
    'av':0.05,             # the damping coefficient of the Voigt profile
    'bnu':1.5,             # the slope of the source function with respect to vertical optical depth
    'logkappa':10**0.98,          # the line strength parameter
    'ndop':int(100),       # the number of sample point per doppler width for the wavelength array
  }

weakparam = {
        'geff':1.0
    }
    
gridparam = {
        'Bpole_grid': np.arange(0,5000, 500),
        'incl_grid': np.arange(0,180,20),
        'beta_grid':np.arange(0,180,20),
        'phase_grid':np.arange(0,360,20)
        }
    
param={'general' : genparam,
       'weak' : weakparam,
       'grid' : gridparam,
       }
'''


@njit
def B_loop_numba(chi2V_data, chi2N1_data, Bpole_grid, 
                    specV, specSigV, specN1, specSigN1, nobs,
                    interpol_modelV, ind_beta, ind_p):
    '''
    Version 1 of the helper function for the Bpole loop
    '''
                
    for ind_bp, bpole in enumerate(Bpole_grid):
    
        # multiply by the Bpole value that I left out from the disk integration

        for o in range(0,nobs):
            chiV  = np.sum( ( (specV[o]  - interpol_modelV[o]*bpole) / specSigV[o]  )**2 )

            chiN1 = np.sum( ( (specN1[o] - interpol_modelV[o]*bpole) / specSigN1[o] )**2 )

            chi2V_data[ind_beta,ind_bp,ind_p,o] = chiV
            chi2N1_data[ind_beta,ind_bp,ind_p,o] = chiN1

    return(chi2V_data, chi2N1_data)

@njit
def B_loop_numba2(chi2V_data, chi2N1_data, Bpole_grid, 
                    specV, specSigV, specN1, specSigN1,
                    interpol_modelV, ind_beta, ind_p):
    '''Version 2 of the helper function for the Bpole loop'''

    for ind_bp, bpole in enumerate(Bpole_grid):
                
        # multiply by the Bpole value that I left out from the disk integration
        # We can sum the whole thing, cause the missing data for shorter lsd profiles are set to zero. 
        chi2V_data[ind_beta,ind_bp,ind_p,:] = np.sum( ( (specV  - interpol_modelV*bpole) / specSigV  )**2, axis=1 )
        chi2N1_data[ind_beta,ind_bp,ind_p,:] = np.sum( ( (specN1  - interpol_modelV*bpole) / specSigN1  )**2, axis=1 )

    return(chi2V_data, chi2N1_data)



def loop_speed2(param, datapacket, path='', ax=None):
    '''
    Function to calculate the chi square between 
    all of the LSD profiles in a pyRaven data packet and 
    the Stokes V profiles for a grid of dipole paramters 
    (inclination, beta, phase, and Bpole)

    :param param: For now, see the documentation notebook for the content of the param dictionary.
    :param datapacket: a pyRaven datapacket for a given set of observations
    '''

    ########
    # Version 1 of the Bpole loop. 
    # Using Version 2 below at the moment
    #
    ## For speed, extracting the data used in the chi2 calculation into lists of arrays, 
    # cause numba cannot interpret the DataPacket class
    # 
    #specV = [datapacket.cutfit.lsds[x].specV for x in range(0,datapacket.nobs)]
    #typed_specV = List()
    #for x in specV: typed_specV.append(x)

    #specSigV = [datapacket.cutfit.lsds[x].specSigV for x in range(0,datapacket.nobs)]
    #typed_specSigV = List()
    #for x in specSigV: typed_specSigV.append(x)

    #specN1 = [datapacket.cutfit.lsds[x].specN1 for x in range(0,datapacket.nobs)]
    #typed_specN1 = List()
    #for x in specN1: typed_specN1.append(x) 
   
    #specSigN1 = [datapacket.cutfit.lsds[x].specSigN1 for x in range(0,datapacket.nobs)]
    #typed_specSigN1 = List()
    #for x in specSigN1: typed_specSigN1.append(x)
   
    ## For speed, extracting the data used in the chi2 calculation into a numpy array
    # padded with zeros (or ones in the case of the sigma), so that I can use
    # np.sum to get the chi2 later on (and save some looping)
    ########

    ########
    # Version 2 of the Bpole loop.
    # keep the list of lsd sizes and get the max length of the lsd profiles
    list_npix = [x.npix for x in datapacket.cutfit.lsds]
    max_npix = np.array(list_npix).max()

    spec_vel = np.zeros((datapacket.nobs, max_npix))
    specV = np.zeros((datapacket.nobs, max_npix))
    specN1 = np.zeros((datapacket.nobs, max_npix))
    specSigV = np.ones((datapacket.nobs, max_npix))
    specSigN1 = np.ones((datapacket.nobs, max_npix))

    for o in range(0, datapacket.nobs):
        spec_vel[o,0:list_npix[o]] = datapacket.cutfit.lsds[o].vel
        specV[o,0:list_npix[o]] = datapacket.cutfit.lsds[o].specV
        specN1[o,0:list_npix[o]] = datapacket.cutfit.lsds[o].specN1
        specSigV[o,0:list_npix[o]] = datapacket.cutfit.lsds[o].specSigV
        specSigN1[o,0:list_npix[o]] = datapacket.cutfit.lsds[o].specSigN1
    ########       
    


    ###############################
    # Intro material
    ###############################
    
    # Checking the parameter dictionary for the necessary keywords
    rav_misc.check_req(param,'loop')
    
    # get the sigma of the macroturbulence+spectral resolution gaussian kernel
    uconv = disk.get_uconv(param)

    ngrid = disk.get_ngrid(param['general']['vsini'], verbose=True)
    #ngrid = 50

    kappa = 10**param['general']['logkappa']    

    perGaussLorentz = disk.get_perGaussLorentz(param['general']['lambda0'], param['general']['vdop'])
    # unno: This gets multiplied by B in the disk integration to get uB,i.
    # weak: This gets multiplied by geff*Bz and the Stokes V shape in the disk integration to get V(uo)
    
    sig = 10 # the width of the Voigt profile. 10 sigma is usually enough
    #small_u = disk.get_small_u(sig, param['general']['ndop'])

    # calculation the Voigt and Voigt-Faraday profiles
    #w_weak, dw_weak = disk.get_w_weak(small_u, param['general']['av'], param['general']['ndop'])
    # Figure out the length of vector that we need for the velocity grid.
    vel_range = param['general']['vsini']/param['general']['vdop']+sig
      
    # Get the total dispersion array
    all_u = disk.get_all_u(vel_range, param['general']['ndop'], verbose=True)

    # create the macroturbulence convolution kernel ahead of time. 
    # if the width of the total kernel less than half of the thermal width, do nothing. 
    if uconv > 0.5:
        ext_all_u = disk.get_all_u(vel_range+10*uconv, param['general']['ndop'], verbose=False)
        kernel = disk.get_resmac_kernel(ext_all_u, uconv)        
        flag_conv = True
    else:
        # This is for using a single variable name in the interpolation 
        # when calculating the chi squares
        ext_all_u = all_u
        flag_conv = False
        
    # create an array in kms unit, for the chi2 calculations
    model_vel = ext_all_u*param['general']['ndop']

    # Surface Grid Setup
    # cartesian coordinates of each grid points in ROT
    # and area of the grid element
    ROT, A = disk.grid_ROT(ngrid) 

    
    #########################################
    #########################################
    ###  Loop over inclination and phase  ###
    #########################################
    #########################################

    # Helping with readability
    nbeta = param['grid']['beta_grid'].size
    nBpole = param['grid']['Bpole_grid'].size
    nphase = param['grid']['phase_grid'].size
    nincl = param['grid']['incl_grid'].size
    

    for ind_i, incl in enumerate(param['grid']['incl_grid']):
        print('Starting inclination loop {}/{}'.format(ind_i+1,nincl))
        tic = time.time()
        # create a matrix for each observation, to store the chi2s (one for each obs)
        # Note that when callong create_chi_data, I then use .data to only keep the numpy data matrix
        # this is to be able to run with numba
        # I will rearrange into a chi object later on. 
        chi2V_data = np.empty((nbeta, nBpole, nphase, datapacket.nobs))
        chi2N1_data = np.empty((nbeta, nBpole, nphase, datapacket.nobs))

        for ind_p, phase in enumerate(param['grid']['phase_grid']):

            #rotation matrix that converts from LOS frame to ROT frame
            los2rot = disk.get_los2rot(incl, phase)
            # the invert of a rotation matrix is the transpose
            rot2los = np.transpose(los2rot)
            # Get the grid coordinates in LOS and MAG systems. 
            LOS = disk.get_LOS(ROT, rot2los)
            # get the necessary values for mu, the projected area, etc. 
            mu_LOS, vis, A_LOS, uLOS, conti_flux = disk.get_LOS_values(LOS, A, param['general']['bnu'], param['general']['vsini'], param['general']['vdop'])

            # Calculating the Voigt grid here
            nvis = vis[vis].size # note: I used this below in the beta loop, although I could change it for np.newaxis method
            u_matrix = np.broadcast_to(all_u, (nvis, all_u.size)) - np.broadcast_to(uLOS[vis], (all_u.size,nvis)).T
            voigt_mat, dvoigt_mat = rav_profileI.get_voigt_weak(u_matrix, param['general']['av'])
            mu_LOS_mat = np.broadcast_to(mu_LOS[vis], (all_u.size,nvis)).T
            A_LOS_mat = np.broadcast_to(A_LOS[vis], (all_u.size,nvis)).T

            ## This is for debugging, in case I need to check the intensity profile:
            #local_I_mat = 1 + mu_LOS_mat*param['general']['bnu'] / (1+kappa*voigt_mat)
            #modelI = np.sum(A_LOS_mat*local_I_mat, axis=0) / conti_flux
            #if flag_conv is True:
            #    modelI = disk.pad_I(ext_all_u, all_u, modelI)
            #    modelI = conv.convolve(modelI,kernel,boundary='fill',fill_value=1.0)
            #ax[0].plot(model_vel, modelI, c='k')
            ##############

            # Now the arrays of local Stokes V profiles
            local_V_mat = ( mu_LOS_mat * dvoigt_mat/(1+kappa*voigt_mat)**2 )
            # These are the constants that were multiplied outside of the old disk integration loop
            # They could be with the last line, it's just to make the code more readable
            local_V_mat = local_V_mat * param['weak']['geff'] /2.0 * perGaussLorentz * param['general']['bnu']*kappa
            # This is the multiplication with the effective area, and normalization to the continuum 
            # which was done during the disk integration in the old loop version. 
            # Could be with the line above, just separated to make things more clear in code
            local_V_mat = local_V_mat * A_LOS_mat / conti_flux
            ## local_V_mat has yet to be multiplied by the local (Bz/Bpole) in the beta loop.
            ## and by Bpole in the Bpole loop (after the disk integration)


            #########################################
            #########################################
            ### Loop over beta values             ###
            #########################################
            #########################################

            for ind_beta, beta in enumerate(param['grid']['beta_grid']):

                # Rotation matrix betweeen the rot and the MAG frames
                #ROT frame to MAG frame
                rot2mag = disk.get_rot2mag(beta) 
                #MAG to ROT frame
                mag2rot = np.transpose(rot2mag)
                # Getting the grid coordinates in the mag frame. 
                MAG = disk.get_MAG(ROT, rot2mag)

                # Get the field values in Bpole/2 units
                B_MAG = disk.get_B_MAG(MAG)
                # Transformation to the LOS frame
                B_LOS = disk.get_B_LOS(B_MAG, mag2rot, rot2los)
                ## The the weak field, all we need is Bz, so B_LOS[z]*Bpole/2

                # Now multiplying the local V profiles by the Bz/Bpole values. 
                local_V_mat_beta = local_V_mat * np.broadcast_to(B_LOS[2,vis], (all_u.size,nvis)).T

                ### Disk integration and convolution with macroturb
                ####################################################

                modelV = np.sum(local_V_mat_beta, axis=0)
                # has yet to be multiplied by Bpole in Bpole loop

                # convolve the model over the instrumental resolution
                ## Stokes V still has to be multiplied by a set of constants and Bpole. 
                #  But (af)*g = a(f*g), so I can make the convolution outside of the 
                # Bpole loop. 

                # if the width of the total kernel less than half of the thermal width, do nothing. 
                if flag_conv is True:
                    modelV = disk.pad_V(ext_all_u, all_u, modelV)
                    modelV = conv.convolve(modelV,kernel,boundary='fill',fill_value=0.0)


                # Interpolate the model to the dispersion of the data
                #interpol_modelV = List() # empty typed list (form numba)
                # loop over the LSD profiles
                #for o in range(0,datapacket.nobs):
                #    interpol_modelV.append(np.interp(datapacket.cutfit.lsds[o].vel, model_vel,modelV))

                interpol_modelV = np.zeros((datapacket.nobs, max_npix))
                for o in range(0, datapacket.nobs):
                    interpol_modelV[o,0:list_npix[o]]= rav_localV.interpol_lin_numba(modelV,  model_vel, spec_vel[o,0:list_npix[o]])
                

                #########################################
                #########################################
                # Loop on the Bpole values
                #########################################
                #########################################
                
                ### this block is just for debbuging!!
                #for ind_bp, bpole in enumerate(param['grid']['Bpole_grid']):
                #    import matplotlib.pyplot as plt
                #    plt.plot(model_vel, modelV*bpole, c='k')
                #plt.show()
                ########################################


                ####
                # Version 1 of the Bpole loop
                # Verion 2 in use below
                #chi2V_data, chi2N1_data = B_loop_numba(
                #                            chi2V_data, chi2N1_data, 
                #                            param['grid']['Bpole_grid'], 
                #                            specV, specSigV, specN1, specSigN1, datapacket.nobs,
                #                            interpol_modelV, 
                #                            ind_beta, ind_p)
                ####
                
                ####
                # Version 2 of the Bpole loop
                # without the numba pre-compilation
                #for ind_bp, bpole in enumerate(param['grid']['Bpole_grid']):
                
                #    # multiply by the Bpole value that I left out from the disk integration
                #    # We can sum the whole thing, cause the missing data for shorter lsd profiles are set to zero. 
                #    chi2V_data[ind_beta,ind_bp,ind_p,:] = np.sum( ( (specV  - interpol_modelV*bpole) / specSigV  )**2, axis=1 )
                #    chi2N1_data[ind_beta,ind_bp,ind_p,:] = np.sum( ( (specN1  - interpol_modelV*bpole) / specSigN1  )**2, axis=1 )                
                ####

                ####
                # Version 2 of the Bpole loop
                # with the numba pre-compilation
                chi2V_data, chi2N1_data = B_loop_numba2(
                                            chi2V_data, chi2N1_data, 
                                            param['grid']['Bpole_grid'], 
                                            specV, specSigV, specN1, specSigN1,
                                            interpol_modelV, 
                                            ind_beta, ind_p)
                ####

                ##
                # I don't see major difference in timing between these
                # versions of the Bpole loop. 
                ##

        # Saving the files for every incl loop
        for o in range(0,datapacket.nobs):
            #transforming into a chi object 
            chi2obj = rav_BayesObjects.chi(chi2V_data[:,:,:,o], param['grid']['beta_grid'], param['grid']['Bpole_grid'], param['grid']['phase_grid'],
                                            incl, datapacket.obs_names[o] )
            chi2obj.write( '{}chiV_i{}obs{}.h5'.format(path,ind_i, o) )
            chi2obj = rav_BayesObjects.chi(chi2N1_data[:,:,:,o], param['grid']['beta_grid'], param['grid']['Bpole_grid'], param['grid']['phase_grid'],
                                            incl, datapacket.obs_names[o] )
            chi2obj.write( '{}chiN1_i{}obs{}.h5'.format(path,ind_i, o) )
        
        toc = time.time()
        print(toc - tic)

    return

def loop_speed2_wrapper(file_param, file_datapacket, path=''):
    '''
    This function is a wrapper to the loop function. 
    It it designed to be run in a directory that contains a parameters json file, and a DataPacket h5 file. 
    The output are written in the same directory by default. 

    This wrapper is useful to generate the chi2 data on a remote server. 

    :param file_param: A json file containing the parameter object (see INSERT LINK)
    :param file_datapackt: A h5 file containing the DataPacket structure
    '''

    param = rav_params.read_parameters(file_param)
    packet = rav_data.read_packet(file_datapacket)

    loop_speed2(param, packet)

    return
