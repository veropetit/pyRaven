import numpy as np
from scipy.special import erf
from astropy.convolution import Gaussian1DKernel, convolve
import astropy.convolution as con
import copy

import pyRaven as rav

'''
The input for both diskint.strong and diskint.weak is of the form:
genparam = {
    'lambda0':5811.969,    # the central wavelength of the transition in AA
    'vsini':50.0,         # the projected rotational velocity in km/s
    'vdop':10.0,          # the thermal broadening in km/s
    'av':0.05,             # the damping coefficient of the Voigt profile
    'bnu':1.5,             # the slope of the source function with respect to vertical optical depth
    'logkappa':10**0.98,   # the line strength parameter
    'Bpole':1.0e5,         # the dipolar field strength in G
    'incl':np.pi/4,      # the inclination of the rotational axis to the line of sight in rad
    'beta':np.pi/4,      # the obliquity of the magnetic axis to the rotational axis in rad
    'phase':0.0,     # the rotational phase in rad
    'ndop':int(100),       # the number of sample point per doppler width for the wavelength array
  }
unnoparam = {
    'down':[0.5,0, 0.5],   # the s, j, l of the lower level
    'up':[0.5, 1, 0.5],    # the s, j, l of the upper level
    }
weakparam = {
        'geff':1.0
    }
param={'general' : genparam,
       'unno' : unnoparam,
       'weak' : weakparam}
'''

def numerical(param,unno):
    '''
    '''
    
    ###############################
    # Constants and zeeman pattern
    ###############################

    ngrid = 1000 + 20*param['general']['vsini']
    kappa = 10**param['general']['logkappa']    

    ## This is e / (4 pi m_e c) into units of (km/s)/(G AA).
    ## See the disk integration demo notebook. 
    constLorentz = 1.3996244936166518e-07
    c_kms = 2.99792458e-5 # used to convert dispersion into wavelength

    if unno==True:
        print('Evaluating with unno method...')
        # get the zeeman pattern
        pattern = rav.pattern.zeeman_pattern(param['unno']['down'],param['unno']['up'])
        
    # unno: This gets multiplied by B in the disk integration to get uB,i.
    # weak: This gets multiplied by geff*Bz and the Stokes V shape in the disk integration to get V(uo)

    perGaussLorentz = constLorentz * param['general']['lambda0'] / param['general']['vdop']

    ###############################
    # Doppler velocity grid setup
    ###############################

    # Compute the Voigt functions only once.
    # Up to 15 vdop (keeping 1 vdop on each side to padd with zeros.

    small_u = np.linspace(-15,15,15*param['general']['ndop']*2)

    # calculation the Voigt and Voigt-Faraday profiles
    w = rav.profileI.voigt_fara(small_u,param['general']['av'])

    # padding with zero on each ends to help with the interpolation later. 
    w[0:param['general']['ndop']]=0.0
    w[w.size-param['general']['ndop']:w.size]=0.0

    # for the weak field, need to multiply by 1/sqrt(pi) now
    # in the unno this is done in the local profile function. 
    shape_V = np.gradient(w.real, small_u)

    
    # Figure out the length of vector that we need for the velocity grid.
    # Take the max of the Zeeman shift or the vsini, then add the 15vdop width.
    # to accomodate the whole w(small_u) array shifted to the maximum value possible.

    if unno == True:
        max1 = np.max(np.abs(pattern['sigma_r']['split']))
        max2 = np.max(np.abs(pattern['pi']['split']))
        max_b = np.max([max1,max2])*param['general']['Bpole']*perGaussLorentz
        max_vsini = param['general']['vsini']/param['general']['vdop']

        print('Max shift due to field: {} vdop'.format(max_b))
        print('Max shift due to vsini: {} vdop'.format(max_vsini))
    
        vel_range = np.max( [max_b+15, max_vsini+15] )

    else:
        vel_range = param['general']['vsini']/param['general']['vdop']

    # rounding up to a integer
    vrange = np.round(vel_range+1)
    print('Max velocity needed: {} vdop'.format(vel_range))

    all_u = np.linspace( -1*vrange,vrange,int(param['general']['ndop']*vrange*2+1))
    print('Number of wavelength/velocity grid points: {}'.format(all_u.size))

    #################################
    # Set up the model structure that will save the resulting spectrum
    #################################    

    if unno == True:
        # If we are doing a unno calculation, save QUV
        model = np.zeros(all_u.size,
                        dtype=[('wave',float),('vel',float),('uo',float),
                            ('flux',float),('fluxnorm', float),
                            ('Q',float),('U',float),('V',float)])
    else:
        # If we are doing a weak field calculation, just save V
        model = np.zeros(all_u.size,
                        dtype=[('wave',float),('vel',float),('uo',float),
                            ('flux',float),('fluxnorm', float),
                            ('V',float)])

    # Saving the u, velocity, and wavelength arrays
    # Note: this uses the astropy constant for c. 
    model['uo'] = all_u
    model['vel'] = all_u * param['general']['vdop'] # result in km/s
    model['wave'] = model['vel']/c_kms*param['general']['lambda0']+param['general']['lambda0']
    

    ##############################
    # Surface Grid Setup
    ##############################

    # We want elements with roughtly the same area
    dA = (4*np.pi)/ngrid
    
    dtheta = np.sqrt(4.0*np.pi/ngrid)
    ntheta = int(np.round(np.pi/dtheta)) #the number of anulus is an integer
    real_dtheta = np.pi/ntheta #the real dtheta
    theta = np.arange(real_dtheta/2.0,np.pi,real_dtheta)
    
    #array of desired dphi for each annulus
    dphi = 4.0*np.pi/(ngrid*np.sin(theta)*real_dtheta)
    nphi = np.round(2*np.pi/dphi)#number of phases per annulus (integer)
    real_dphi = 2.0*np.pi/nphi#real dphi per annulus
    real_n=np.sum(nphi)
    
    # create arrays with informations on each grid point
    A=np.array([])
    grid_theta = np.array([])
    grid_phi = np.array([])
    
    for i in range(ntheta):
        grid_theta = np.append(grid_theta,[theta[i]]*int(nphi[i]))
        arr_phi = np.arange(real_dphi[i]/2.0, 2*np.pi,real_dphi[i])
        grid_phi = np.append(grid_phi,arr_phi)
        arr_A = [np.sin(theta[i])*real_dtheta*real_dphi[i]]*int(nphi[i])
        A = np.append(A,arr_A)
    
    # cartesian coordinates of each grid points in LOS, ROT and MAG
    ROT = np.zeros( (3,int(real_n)))
    ROT[0,:] = np.sin(grid_theta)*np.cos(grid_phi) #X ROT
    ROT[1,:] = np.sin(grid_theta)*np.sin(grid_phi) #Y ROT
    ROT[2,:] = np.cos(grid_theta) #Z ROT

    ####################################
    # LOS Rotation Matrix Setup for a given i phase
    ####################################

    #rotation matrix that converts from LOS frame to ROT frame
    los2rot = np.array( [[np.cos(param['general']['phase']),
                            np.cos(param['general']['incl'])*np.sin(param['general']['phase']),
                            -1*np.sin(param['general']['incl'])*np.sin(param['general']['phase'])],
                            [-1*np.sin(param['general']['phase']),
                            np.cos(param['general']['incl'])*np.cos(param['general']['phase']),
                            -1*np.sin(param['general']['incl'])*np.cos(param['general']['phase'])],
                            [0.0,
                            np.sin(param['general']['incl']),
                            np.cos(param['general']['incl'])] ] )

    # the invert of a rotation matrix is the transpose
    rot2los = np.transpose(los2rot)

    # Get the grid coordinates in LOS and MAG systems. 
    LOS = np.matmul(rot2los,ROT)
    
    ####################################
    # Visible grid element for this i and beta
    ####################################
   
    # Calulating values on the grid
    # visible elements where z component ge 0
    # mu = cos angle between normale to surface and LOS
    # normal = [x,y,z]*[0,0,1] = 1*1*costheta
    # therefore mu = z_LOS / 1
    # the 1.0* is to avoid a symbolic association
    mu_LOS=1.0*LOS[2,:]
    vis = (mu_LOS >= 0.0) #TRUE is the element is visible

    # just put all the unvisible values to zero, to avoid calculation issues
    # later on -- this way all of my arrays have the same size, and when I 
    # make a sum or later on loop over the visible elements, I will slice with vis.
    mu_LOS[vis==False] = 0.0 
    
    # projected surface area in the LOS
    # thus not visible surface points also have zero A_LOS
    A_LOS = A * mu_LOS
    A_LOS /= np.sum(A_LOS[vis]) # Normalized to unity so that the integral over projected area=1.
    
    # value of the continuum intensity on each grid point
    # conti_int = ( 1.+ mu_LOS*bnu ), which includes the continuum limb-darkening
    # as well as the line limb-darkening (the reason why there is an extra
    # mu_LOS in front of the slope of the source function)
    # scaled by the projected area
    conti_int_area = A_LOS*(1.0+mu_LOS*param['general']['bnu'])
    conti_flux = np.sum(conti_int_area[vis])# continuum flux (for profile normalisation)
    
    # radial velocity in doppler units.
    # The rotational axis is tilted towards the LOS y-axis, so the rotational
    # velocity will be in the LOS x direction. 
    uLOS = param['general']['vsini']* LOS[0,:] / param['general']['vdop']
    
    ###########################
    # MAG Rotation matrix setup for a given beta
    ###########################

    # Rotation matrix betweeen the rot and the MAG frames
    #ROT frame to MAG frame
    rot2mag = np.array( [[ 1.0, 0.0, 0.0],
                    [ 0.0, np.cos(param['general']['beta']),np.sin(param['general']['beta'])],
                    [ 0.0, -1*np.sin(param['general']['beta']),np.cos(param['general']['beta'])] ] )
    
    #MAG to ROT frame
    mag2rot = np.transpose(rot2mag)

    # Getting the grid coordinates in the mag frame. 
    MAG = np.matmul(rot2mag,ROT)

    ##########################
    # Calculating the field values on the grid
    ##########################

    # check for sinthetab =0
    # to avoid dividing by zero in the B_MAG calculations below.
    sinthetab = np.sqrt(MAG[0,:]**2 + MAG[1,:]**2)
    no_null = np.where(sinthetab>0)
    B_MAG = np.zeros((3,int(real_n)))
    
    # B in (Bpole/2) units in MAG coordinate
    B_MAG[0,no_null] = 3*MAG[2,no_null]*sinthetab[no_null]*MAG[0,no_null]/sinthetab[no_null]
    B_MAG[1,no_null] = 3*MAG[2,no_null]*sinthetab[no_null]*MAG[1,no_null]/sinthetab[no_null]
    B_MAG[2,:] = 3*MAG[2,:]**2-1.0
    
    # Get the field components in LOS (in Bp/2 units)
    B_LOS = np.matmul(rot2los,np.matmul(mag2rot,B_MAG))
    ## The th weak field, all we need is Bz, so B_LOS[z]*Bpole/2

    if unno == True:
        # but for unno, we need some of the angles
    
        # calcuate the magnitde of the field vectors (in Bp/2 units)
        # (doesn't matter if we do this in MAG or LOS)
        B_magnitude_Bp2 = np.sqrt(B_MAG[0,:]**2+B_MAG[1,:]**2+B_MAG[2,:]**2)
        # Take a copy of B_LOS 
        B_hat = np.copy(B_LOS)
        # divide by the magnitude to get the unit vectors in the direction of the field in LOS
        for i in range(3):
            B_hat[i,:] /= B_magnitude_Bp2
        # calcuate the angle necessary for the RT, mainly the angles between the field
        # direction and the LOS direction (theta) and the reference direction in the
        # plane of the sky (chi). (for the latter, the RT equations need cos(2phi))
        # See the explanatory notebook in the repository. 
        costheta_temp = 1.0*B_hat[2,:]
        sintheta_temp = (1.0 - costheta_temp**2)**0.5 
        # cos(2phi) and sin(2phi)
        sin2chi_temp = 2.0*B_hat[0,:]*B_hat[1,:]/sintheta_temp**2
        cos2chi_temp = 2.0*B_hat[0,:]**2/sintheta_temp**2-1.0
    
        # Scale the unit vector by the field strength
        # to get the magnitude of the field
        B = B_magnitude_Bp2 * param['general']['Bpole']/2.0
    
    ### Vero here

    #############################
    # Disk Int time
    #############################

    # Making the two loops here. It repeat some of the code, 
    # but we don't want to have a check for unno==True for each grid point.

    if unno==True:

        # only iterating on the visible elements, to save calculations
        # checking how many True elements are in vis by slicing. 
        for i in range(0,vis[vis].size):
            out = rav.localV.unno_local_out_IV(
                        B[vis][i]*perGaussLorentz,# uB
                        uLOS[vis][i],              # uLOS
                        pattern,              # the zeeman pattern
                        small_u,              # the uo array for the non-magnetic profile
                        w,                    # th Voigt and Voigt-Fara profiles
                        all_u,                # the complete uo array
                        sintheta_temp[vis][i], costheta_temp[vis][i], # the angles of the field to the LOS
                        sin2chi_temp[vis][i], cos2chi_temp[vis][i], # the angles of the field to the LOS
                        kappa, param['general']['bnu'], # the line strenght parameter and the slope of the source function
                        mu_LOS[vis][i]             # the angle between the surface and the LOS
                        )

            if i == 5:
                #testI = A_LOS[vis][i]*out['I']
                #testV = A_LOS[vis][i]*out['V']
                testI = out['I']
                testV = out['V']
                    
            model['flux'] += A_LOS[vis][i]*out['I']
            model['Q']    += A_LOS[vis][i]*out['Q']
            model['U']    += A_LOS[vis][i]*out['U']
            model['V']    += A_LOS[vis][i]*out['V']
    
    else:

        for i in range(0,vis[vis].size):

            # Shift the profiles to the right uLOS
            prof_V_shift = rav.localV.interpol_lin(shape_V, small_u+uLOS[vis][i], all_u)/np.pi**0.5
            prof_I_shift = rav.localV.interpol_lin(w.real,  small_u+uLOS[vis][i], all_u)/np.pi**0.5

            model['flux'] += A_LOS[vis][i]*(1+(1+kappa*prof_I_shift)**(-1)*mu_LOS[vis][i]*param['general']['bnu'])

            # The value of geff*Bz in gauss
            geffBz = param['weak']['geff'] * B_LOS[2,vis][i] * param['general']['Bpole']/2.0
            # getting the profile scaled by the field. 
            prof_V_shift = perGaussLorentz * geffBz * prof_V_shift
            # scale by the source function slope (mu*bnu), kappa, and the projected area
            model['V'] += A_LOS[vis][i]*(kappa*mu_LOS[vis][i]*param['general']['bnu'])*prof_V_shift

            if i == 5:
                #testI = A_LOS[vis][i]*(1+(1+kappa*prof_I_shift)*mu_LOS[vis][i]*param['general']['bnu'])
                #testV = A_LOS[vis][i]*(kappa*mu_LOS[vis][i]*param['general']['bnu'])
                testI = (1+(1+kappa*prof_I_shift)**(-1)*mu_LOS[vis][i]*param['general']['bnu'])
                testV = (kappa*mu_LOS[vis][i]*param['general']['bnu'])*prof_V_shift

            
         
    

 
 
 
    # Normalize the spectra to the continuum.
    model['flux'] /= conti_flux
    model['V'] /= conti_flux

    if unno == True:
        model['Q'] /= -1*conti_flux# sign flip to be consistent with Zeeman2
        model['U'] /= -1*conti_flux# sign flip to be consistent with Zeeman2

    
    return(model, testI, testV)

def analytical(param):
    
    #models the line profile by convolving the voigt fara function with the rotation profile

    ngrid = 20*param['general']['vsini']
    kappa = 10**param['general']['logkappa'] 

    const = { 'larmor' : 1.3996e6,\
        'c' : 2.99792458e5 } #cgs units
    
    varepsilon=1-(1/(1+param['general']['bnu'])) #defines the limb darkening coefficient
    urot = param['general']['vsini'] / param['general']['vdop'] #defines the doppler width of vsini
    
    #Defines the rotation profile
    def rotconv(u0, varepsilon, urot):
        model=np.zeros(len(u0))
        
        for i in range(len(u0)):
            if u0[i]>=-urot and u0[i]<=urot:
                model[i]=(2*(1-varepsilon)*np.sqrt(1-(u0[i]/urot)**2)+0.5*np.pi*varepsilon*(1-(u0[i]/urot)**2))/(np.pi*(1-varepsilon/3)*urot)
        return(model)
    
    print('Max shift due to vsini: {} vdop'.format(urot))
    
    #sets up the u axis
    vel_range = np.max( [15, urot+15] )
    vrange = np.round(vel_range+1+urot)
    print('Max velocity needed: {} vdop'.format(vel_range))
    
    all_u = np.linspace( -1*vrange,vrange,int(param['general']['ndop']*vrange*2+1))
    
    #finds the rotation profile
    rotation=rotconv(all_u,varepsilon,urot)
    if rotation.size%2 == 0:
        rotation=np.append(rotation,rotation[-1])
    
    #models the voigt fara function 
    voigt = rav.profileI.voigt_fara(all_u,param['general']['av']).real
    flux=(1.0+2.0*param['general']['bnu']/(3.0*(1.0+kappa*voigt/np.sqrt(np.pi))))/(1.0+2.0*param['general']['bnu']/3.0)

    #Convolves the rotation profle and voigt fara
    rotk=con.CustomKernel(rotation)
    rot=convolve(flux,rotk,boundary='fill',fill_value=1.0)
    
    #sets up the model
    model = np.zeros(all_u.size,
                    dtype=[('wave',float),('vel',float),('vdop',float),
                            ('flux',float),('fluxnorm',float)])
                
    model['vdop'] = all_u
    model['vel'] = all_u * param['general']['vdop']
    model['wave'] = (model['vel']/const['c'])*param['general']['lambda0']+param['general']['lambda0']
    
    model['flux']=rot #flux
    
    return(model,rotation,flux)
