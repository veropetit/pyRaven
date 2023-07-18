import numpy as np
from scipy.special import erf
from astropy.convolution import Gaussian1DKernel, convolve
import astropy.convolution as conv

import pyRaven as rav
import pyRaven.diskint2 as disk
import time

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

def loop(param, datapacket, path=''):
    '''
    Function to calculate the chi square between 
    all of the LSD profiles in a pyRaven data packet and 
    the Stokes V profiles for a grid of dipole paramters 
    (inclination, beta, phase, and Bpole)

    :param param: For now, see the documentation notebook for the content of the param dictionary.
    :param datapacket: a pyRaven datapacket for a given set of observations
    '''


    ###############################
    # Intro material
    ###############################
    
    # Checking the parameter dictionary for the necessary keywords
    rav.misc.check_req(param,'loop')
    
    # get the sigma of the macroturbulence+spectral resolution gaussian kernel
    uconv = disk.get_uconv(param)

    ngrid = disk.get_ngrid(param['general']['vsini'], verbose=True)
    #ngrid = 50

    kappa = 10**param['general']['logkappa']    

    perGaussLorentz = disk.get_perGaussLorentz(param['general']['lambda0'], param['general']['vdop'])
    # unno: This gets multiplied by B in the disk integration to get uB,i.
    # weak: This gets multiplied by geff*Bz and the Stokes V shape in the disk integration to get V(uo)
    
    sig = 10 # the width of the Voigt profile. 10 sigma is usually enough
    small_u = disk.get_small_u(sig, param['general']['ndop'])

    # calculation the Voigt and Voigt-Faraday profiles
    w_weak, dw_weak = disk.get_w_weak(small_u, param['general']['av'], param['general']['ndop'])
    # Figure out the length of vector that we need for the velocity grid.
    vel_range = param['general']['vsini']/param['general']['vdop']+sig
      
    # Get the total dispersion array
    all_u = disk.get_all_u(vel_range, param['general']['ndop'], verbose=True)
    
    # Surface Grid Setup
    # cartesian coordinates of each grid points in ROT
    # and area of the grid element
    ROT, A = disk.grid_ROT(ngrid) 

    
    #########################################
    #########################################
    ###  Loop over inclination and phase  ###
    #########################################
    #########################################

    for ind_i, incl in enumerate(param['grid']['incl_grid']):
        print('Starting inclination loop {}/{}'.format(ind_i,param['grid']['incl_grid'].size))
        tic = time.time()
        # create a structure for each observation, to store the chi2s (one for each obs)
        chi2V_data = []
        for o in range(0, datapacket.nobs):
            chi2V_data.append(rav.BayesObjects.create_chi(param['grid']['beta_grid'], param['grid']['Bpole_grid'], param['grid']['phase_grid'],
                                            incl, datapacket.obs_names[o] ))
        chi2N1_data = []
        for o in range(0, datapacket.nobs):
            chi2N1_data.append(rav.BayesObjects.create_chi(param['grid']['beta_grid'], param['grid']['Bpole_grid'], param['grid']['phase_grid'],
                                            incl, datapacket.obs_names[o] ))


        for ind_p, phase in enumerate(param['grid']['phase_grid']):

            #rotation matrix that converts from LOS frame to ROT frame
            los2rot = disk.get_los2rot(incl, phase)
            # the invert of a rotation matrix is the transpose
            rot2los = np.transpose(los2rot)
            # Get the grid coordinates in LOS and MAG systems. 
            LOS = disk.get_LOS(ROT, rot2los)
            # get the necessary values for mu, the projected area, etc. 
            mu_LOS, vis, A_LOS, uLOS, conti_flux = disk.get_LOS_values(LOS, A, param['general']['bnu'], param['general']['vsini'], param['general']['vdop'])


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

                #########################################
                #########################################
                ### Disk integration loop             ###
                #########################################
                #########################################
 
                modelV = np.zeros(all_u.size)
                for i in range(0,vis[vis].size):
                    local_I, local_V = disk.get_local_weak_interp(small_u, w_weak, dw_weak, all_u,
                                                uLOS[vis][i], mu_LOS[vis][i], 
                                                param['general']['bnu'], kappa, 
                                                B_LOS[2,vis][i]  )

                    # numerical integration (hence the projected area multiplication)
                    modelV += A_LOS[vis][i]*local_V
                # constant stuff that I pulled out of the loop in get_local_weak_interp (except for Bpole)
                modelV = modelV * param['weak']['geff'] /2.0 * perGaussLorentz * param['general']['bnu']*kappa
                 # Normalize the spectra to the continuum.
                modelV /= conti_flux

                # convolve the model over the instrumental resolution
                ## Stokes V still has to be multiplied by a set of constants and Bpole. 
                #  But (af)*g = a(f*g), so I can make the convolution outside of the 
                # Bpole loop. 

                # if the width of the total kernel less than half of the thermal width, do nothing. 
                if uconv > 0.5:
                    ext_all_u = disk.get_all_u(vel_range+10*uconv, param['general']['ndop'], verbose=False)
                    modelV = disk.pad_V(ext_all_u, all_u, modelV)
                    kernel = disk.get_resmac_kernel(ext_all_u, uconv)
                    modelV = conv.convolve(modelV,kernel,boundary='fill',fill_value=0.0)
                else:
                    # This is for using a single variable name in the interpolation below
                    ext_all_u = all_u
                    
                # create an array in kms unit, for the chi2 calculations
                model_vel = ext_all_u*param['general']['ndop']
                
                # Interpolate the model to the dispersion of the data
                interpol_modelV = [] # empty list
                # loop over the LSD profiles
                for o in range(0,datapacket.nobs):
                    interpol_modelV.append(np.interp(datapacket.cutfit.lsds[o].vel, model_vel,modelV))
                
                #########################################
                #########################################
                # Loop on the Bpole values
                #########################################
                #########################################
                
                for ind_bp, bpole in enumerate(param['grid']['Bpole_grid']):
                
                    # multiply by the Bpole value that I left out from the disk integration loop above
                    #model_V_scaled = modelV * bpole

                    for o in range(0,datapacket.nobs):
                        chiV = np.sum(   ( 
                                            (datapacket.cutfit.lsds[o].specV - interpol_modelV[o]*bpole)
                                            / datapacket.cutfit.lsds[o].specSigV
                                        )**2
                                    )
                        chiN1 = np.sum(   ( 
                                            (datapacket.cutfit.lsds[o].specN1 - interpol_modelV[o]*bpole)
                                            / datapacket.cutfit.lsds[o].specSigN1
                                        )**2
                                    )

                        chi2V_data[o].data[ind_beta,ind_bp,ind_p] = chiV
                        chi2N1_data[o].data[ind_beta,ind_bp,ind_p] = chiN1

        for o in range(0,datapacket.nobs):
            chi2V_data[0].write( '{}chiV_i{}obs{}.h5'.format(path,ind_i, o) )
            chi2N1_data[0].write( '{}chiN1_i{}obs{}.h5'.format(path,ind_i, o) )
        
        toc = time.time()
        print(toc - tic)

    return

def loop_wrapper(file_param, file_datapacket, path=''):
    '''
    This function is a wrapper to the loop function. 
    It it designed to be run in a directory that contains a parameters json file, and a DataPacket h5 file. 
    The output are written in the same directory by default. 

    This wrapper is useful to generate the chi2 data on a remote server. 

    :param file_param: A json file containing the parameter object (see INSERT LINK)
    :param file_datapackt: A h5 file containing the DataPacket structure
    '''

    param = rav.params.read_parameters(file_param)
    packet = rav.data.read_packet(file_datapacket)

    loop(param, packet)

    return
