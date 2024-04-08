import numpy as np
from scipy.special import erf
import astropy.convolution as conv
import copy

#import time

#import pyRaven as rav

import matplotlib.pyplot as plt
from numba import njit

from . import profileI as rav_profileI
from . import localV as rav_localV
from . import misc as rav_misc
from . import pattern as rav_pattern 

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
    'incl':90.0,      # the inclination of the rotational axis to the line of sight in degree
    'beta':90.0,      # the obliquity of the magnetic axis to the rotational axis in degree
    'phase':90.0,     # the rotational phase in rad
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

## This is e / (4 pi m_e c) into units of (km/s)/(G AA).
## See the disk integration demo notebook. 
constLorentz = 1.3996244936166518e-07

# Used for the conversion of velocity to wavelength in the output model array
# and for the calculation of the spectral PSF width in velocity from R
c_kms = 2.99792458e5


def get_ngrid(vsini, verbose=False):
    '''Function to calculate the number of spatial grid point necessary for a given vsini. '''
    ngrid = 1000 + 40*vsini
    if verbose: print('Using {} grid point on the surface'.format(ngrid))
    return(ngrid)

def get_uconv(param, verbose=False):
    '''
    Function to calculate and return the combined sigma of the gaussian kernel that will 
    perform the convolution for the macroturbulence and the spectral instrument resolution
    :param param: a parameter dictionary
    '''

    if 'res' not in list(param['general'].keys()):
        if verbose: print('No spectral resolution given, no PDF convolution performed')
        sig_res = 0.0
    else:
        sig_res = c_kms/param['general']['res']/(8*np.log(2))**0.5
    # check for the macroturbulence keyword in general
    if 'vmac' not in list(param['general'].keys()):
        if verbose: print('No macroturbulence velocity given, no convolution performed')
        sig_mac = 0.0
    else:
        sig_mac = param['general']['vmac']
    uconv = (sig_res**2+sig_mac**2)**0.5/param['general']['vdop']
    return(uconv)

def get_perGaussLorentz(lambda0, vdop):
    '''
    Function to calculate the perGaussLorentz (in 1/Gauss) constant for a given central wavelength and thermal broadening.

    * For the unno calculation: This gets multiplied by B in the disk integration to get uB,i.
    * For the weak-field approximation calculation: This gets multiplied by geff*Bz and the Stokes V shape in the disk integration to get V(uo)

    :param lambda0: central wavelength in AA. 
    :param vdop: thermal broadening in km/s
    '''
    perGaussLorentz = constLorentz * lambda0 / vdop
    # unno: This gets multiplied by B in the disk integration to get uB,i.
    # weak: This gets multiplied by geff*Bz and the Stokes V shape in the disk integration to get V(uo)
    return(perGaussLorentz)    

def get_small_u(sig, ndop):
    '''
    Function to return the small dispersion grid (in uo units) for the Voigt-Fara profile calculations

    :param sig: number of sigmas from line center for the range (usually 8-10)
    :param ndop: number of datapoints per sigma (usually 5-10)
    '''
    small_u = np.linspace(-sig,sig,sig*ndop*2)
    return(small_u)

def get_w_unno(small_u, av, ndop):
    '''
    Function to calculate the Voigt and Voigt-Faraday profiles over the given dispersion grid (in uo units). 
    It will also pad the array at both ends with zeros for one sigma. 
    Note: the result is an array of complex numbers. The real part is the Voigt profile and the imaginary part is the Voigt-Faraday profile.
    Note: Used by the unno solution. The profile is NOT normalized (so not mulitplied by 1/sqrt(pi), because this is done in the local profile function calculation)

    :param small_u: the dispersion grid in uo units
    :param av: the damping factor
    :param ndop: the number of datapoint per sigma used in the creation of the dispersion array. 
    '''
    w = rav_profileI.voigt_fara(small_u,av)
    # padding with zero on each ends to help with the interpolation later. 
    w[0:ndop]=0.0
    w[w.size-ndop:w.size]=0.0
    return(w)

def get_w_weak(small_u, av, ndop):
    '''
    Function to calculate the normalized (so mulitplied by 1/sqrt(pi)) local profile, and the derivative of the local profile, 
    to be used in the the weak-field approximation. 
    It will also pad the array at both ends with zeros for one sigma. 

    :param small_u: the dispersion grid in uo units
    :param av: the damping factor
    :param ndop: the number of datapoint per sigma used in the creation of the dispersion array. 
    '''

    # calculation the Voigt and Voigt-Faraday profiles
    w = rav_profileI.voigt_fara(small_u,av)
    # for the weak field, need to multiply by 1/sqrt(pi) now
    # in the unno this is done in the local profile function. 
    w_weak = w.real/np.sqrt(np.pi)
    dw_weak = np.gradient(w_weak, small_u)
    # padding with zero on each ends to help with the interpolation later. 
    w_weak[0:ndop]=0.0
    w_weak[w.size-ndop:w_weak.size]=0.0
    dw_weak[0:ndop]=0.0
    dw_weak[w.size-ndop:dw_weak.size]=0.0
    return(w_weak, dw_weak)

def get_vel_range_unno(pattern, Bpole, perGaussLorentz, vsini_over_vdop, sig, verbose=False):
    '''
    Function to figure out the length of vector that we need for the velocity grid, for the Unno solution.
    Takes the max of the Zeeman shift or the vsini, then add the sig*vdop widths.
    to accomodate the whole w(small_u) array shifted to the maximum value possible.

    :param pattern: the zeeman pattern object
    :param Bpole: the dipolar (or max) field strength
    :perGaussLorentz: the perGaussLorentz constant for that specific central wavelenght and thermal broadening. 
    :vsini_over_vdop: the rotational broadening in uo units. 
    :sig: the width in sigma that was used to calculate the small dispersion array (the one used to calculated the Voigt profile)
    '''
    max1 = np.max(np.abs(pattern['sigma_r']['split']))
    max2 = np.max(np.abs(pattern['pi']['split']))
    max_b = np.max([max1,max2])*Bpole*perGaussLorentz
    #max_vsini = param['general']['vsini']/param['general']['vdop']

    if verbose:
        print('Max shift due to field: {} vdop'.format(max_b))
        print('Max shift due to vsini: {} vdop'.format(vsini_over_vdop))
    
    vel_range = np.max( [max_b+sig, vsini_over_vdop+sig] )    
    
    return(vel_range)

def get_all_u(vel_range, ndop, verbose=False):
    '''
    Function to create the larger dispersion array (in uo units) that will accomodate the whole disk-integrated profile. 
    
    :param vel_range: the range needed to accomodate the profile in uo units. NOTE: the code with round up to a integer bound
    :param ndop: the number of datapoints per sigma (use the same as for the small dispersion array, please)
    '''
    # rounding up to a integer
    vrange = np.round(vel_range+1)
    all_u = np.linspace( -1*vrange,vrange,int(ndop*vrange*2+1))
    if verbose:
        print('Max velocity needed: {} vdop'.format(vel_range))
        print('Number of wavelength/velocity grid points: {}'.format(all_u.size))
    return(all_u)

def get_empty_model(all_u, vdop, lambda0, unno=True):
    '''
    Function to create a structure to save the final model. If unno=True, IQUV, else, just IV. 

    :param all_u: the total dispersion array in uo units
    :param vdop: the thermal broadening in km/s
    :param lambda0: rest wavelength of transition in AA. 
    '''

    if unno == True:
        # If we are doing a unno calculation, save QUV
        model = np.zeros(all_u.size,
                        dtype=[('wave',float),('vel',float),('uo',float),
                            ('flux',float),
                            ('Q',float),('U',float),('V',float)])
    else:
        # If we are doing a weak field calculation, just save V
        model = np.zeros(all_u.size,
                        dtype=[('wave',float),('vel',float),('uo',float),
                            ('flux',float),
                            ('V',float)])

    # Saving the u, velocity, and wavelength arrays
    # Note: this uses the astropy constant for c. 
    model['uo'] = all_u
    model['vel'] = all_u * vdop # result in km/s
    model['wave'] = model['vel']/c_kms*lambda0+lambda0

    return(model)
     
def grid_ROT(ngrid):
    '''
    Function to set up the grid at the surface of the star,
    in the rotation frame of reference. 
    NOTE: the final number of grid point might be slightly different than the input,
    as the code attempts to have grid points with roughtly the same surface area.
    Returns:
    * a (3, n_real) matrix for the x,y,z coordinates of each grid points
    * a (n_real) array with the surface area of each grid point. 

    :param ngrid: the number of spatial gridpoints on the sphere. 
    '''

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

    return(ROT, A)

def get_los2rot(incl, phase):
    '''
    Function to get the rotation matrix that converts from LOS coordinates to ROT coordinates. 
    NOTE: The angles are passed in degrees. 

    :param incl: the inclination of the rotational axis to the line of sight
    :param phase: the rotational phase (see documentation notebook for the rotation coordinate system conventions)
    '''
    i = incl * np.pi/180.0
    phi = phase * np.pi/180.0

    #rotation matrix that converts from LOS frame to ROT frame
    los2rot = np.array( [
        [np.cos(phi),    np.cos(i)*np.sin(phi), -1*np.sin(i)*np.sin(phi)],
        [-1*np.sin(phi), np.cos(i)*np.cos(phi), -1*np.sin(i)*np.cos(phi)],
        [0.0,            np.sin(i),              np.cos(i)] 
        ] )

    return(los2rot)

def get_rot2mag(beta): 
    '''
    Function to get the rotation matrix that converts from ROT coordinates to MAG coordinates. 
    NOTE: The angles are passed in degrees. 

    :param beta: the obliquity of the magnetic axis to the rotational axis
    '''    
    b = beta * np.pi/180.0

    rot2mag = np.array( [
        [ 1.0,    0.0,          0.0],
        [ 0.0,    np.cos(b),    np.sin(b)],
        [ 0.0, -1*np.sin(b),    np.cos(b)] 
        ] )
    
    return(rot2mag)

def get_LOS(ROT, rot2los):
    '''
    Function to convert a ROT grid coordinates into LOS coordinates

    :param ROT: the (3, n_real) matrix for the x,y,z coordinates of each grid points
    :param rot2los: the rotation matrix
    '''
    return(np.matmul(rot2los,ROT))

def get_MAG(ROT, rot2mag):
    '''
    Function to convert a ROT grid coordinates into MAG coordinates

    :param ROT: the (3, n_real) matrix for the x,y,z coordinates of each grid points
    :param rot2mag: the rotation matrix
    '''
    return(np.matmul(rot2mag,ROT))

def get_LOS_values(LOS, A, S1, vsini, vdop):
    '''
    Function to calculate the mu angle, a bolean value for visibility, the projected area and velocity of each grid point, 
    along with the integrated continuum flux for profile normalization
    
    returns:

    * mu_LOS: angle between the surface normal and the line-of-sight
    * vis: boolean array, TRUE is the grid point is visible
    * A_LOS: the projected area of the visible grid points normalized to unity
    * uLOS: the projected rotational velocity of the grid points in uo units. 
    * conti_flux: the integrated continuum flux for profile normalization. 

    :param LOS: The (3, n_real) grid coordinates in LOS coordinates
    :param A: the area of each grid point
    :param S1: the slope of the source function
    :param vsini: projected rotational equatorial velocity (in km/s)
    :param vdop: the thermal broadening velocity (in km/s)
    '''

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
    conti_int_area = A_LOS*(1.0+mu_LOS*S1)
    conti_flux = np.sum(conti_int_area[vis])# continuum flux (for profile normalisation)
    
    # radial velocity in doppler units.
    # The rotational axis is tilted towards the LOS y-axis, so the rotational
    # velocity will be in the LOS x direction. 
    uLOS = vsini * LOS[0,:] / vdop

    return(mu_LOS, vis, A_LOS, uLOS, conti_flux)

def get_B_MAG(MAG):
    '''
    Function to calculate the field values on the MAG grid in (Bpole/2) units

    :param MAG: the MAG (3, n_real) coordinate grid
    :return B_MAG: a (3, n_real) matric with Bx, By, Bz of the grid points in the MAG coordinate. 
    '''
    # check for sinthetab =0
    # to avoid dividing by zero in the B_MAG calculations below.
    sinthetab = np.sqrt(MAG[0,:]**2 + MAG[1,:]**2)
    no_null = np.where(sinthetab>0)
    B_MAG = np.zeros((3,MAG.shape[1]))
    
    # B in (Bpole/2) units in MAG coordinate
    B_MAG[0,no_null] = 3*MAG[2,no_null]*sinthetab[no_null]*MAG[0,no_null]/sinthetab[no_null]
    B_MAG[1,no_null] = 3*MAG[2,no_null]*sinthetab[no_null]*MAG[1,no_null]/sinthetab[no_null]
    B_MAG[2,:] = 3*MAG[2,:]**2-1.0

    return(B_MAG)

def get_B_LOS(B_MAG, mag2rot, rot2los):
    '''
    Function to convert the magnetic field vectors in the MAG coordinate system to the LOS coordinate system

    :param B_MAG: the (3, n_real) matric with Bx, By, Bz of the grid points in the MAG coordinate
    :param mag2rot: the mag to rot rotation matrix
    :param rot2los: the rot to los rotation matrix
    '''
 
    # Get the field components in LOS (in Bp/2 units)
    B_LOS = np.matmul(rot2los,np.matmul(mag2rot,B_MAG))
    ## The th weak field, all we need is Bz, so B_LOS[z]*Bpole/2
    return(B_LOS)

def get_unno_angles(B_LOS, Bpole):
    '''
    Function to calculate for each grid point:
    1. The angles between the magnetic field vectors and the line of sight (cos(theta), sin(theta), cos(2*chi), sin(2*chi))
    2. The magnitude of the magnetic field 

    :param B_LOS: the (3, n_real) matric with Bx, By, Bz of the grid points in the LOS coordinate system
    '''

    # For unno, we need some of the angles

    # calcuate the magnitde of the field vectors (in Bp/2 units)
    # (doesn't matter if we do this in MAG or LOS)
    B_magnitude_Bp2 = np.sqrt(B_LOS[0,:]**2+B_LOS[1,:]**2+B_LOS[2,:]**2)
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
    # Note, always multiplied by sintheta in the unno calculation for the eta and rho
    # So I can set these two to zero then sintheta_temp = 0
    sin2chi_temp = np.zeros(B_hat[0,:].size)
    cos2chi_temp = np.zeros(B_hat[0,:].size)
    n = sintheta_temp != 0
    sin2chi_temp[n] = 2.0*B_hat[0,n]*B_hat[1,n]/sintheta_temp[n]**2
    cos2chi_temp[n] = 2.0*B_hat[0,n]**2/sintheta_temp[n]**2-1.0

    # Scale the unit vector by the field strength/2
    # to get the magnitude of the field
    B = B_magnitude_Bp2 * Bpole/2

    return(B, costheta_temp, sintheta_temp, cos2chi_temp, sin2chi_temp)

def get_local_weak_interp(small_u, w_weak, dw_weak, all_u, uLOS, mu_LOS, bnu, kappa, Bz):
    '''
    Function to get the local I and V from the weak field approximation, by using an interpolation of a 
    fixed profile function in the rest frame. 

    :param small_u: the uo grid on which the profile is defined
    :param w_weak: the voigt profile (already divided by sqrt(2pi))
    :param dw_weak: the derivative of the voigt profile with respect to uo
    :param all_u: the (larger) dispersion grid on which to interpolate (enough to encompase the vsini) in uo units
    :param uLOS: the (rotational) radial velocity of the grid point in uo units
    :param mu_LOS: the mu angle of the grid point (angle between LOS and normal to surface)
    :param bnu: the slope of the source function with respect to tau_z (the vertical optical depth, the change in slope because of the mu angle is coded in)
    :param kappa: the line strenght parameter
    :param Bz: the Bz of that grid point (in Bpole/2 units)

    returns: the local_I and local_V profiles, used for disk integration. 
    The local_I and local_V need to be multiplied by the projected area (A_LOS) of the grid point during the integration.
    Furthermore, the local_V needs to be multiplied (can be done after the integration) by geff * Bpole/2 * kappa * perGaussLorentz,
    where perGaussLorentz(lambda0, vdop) is defined in the explanatory notebook. 
    '''

    # Shift the profiles to the right uLOS, and interpolate over the all_u velocity grid.         
    w_weak_shift  = rav_localV.interpol_lin(w_weak,  small_u+uLOS, all_u)
    dw_weak_shift = rav_localV.interpol_lin(dw_weak, small_u+uLOS, all_u)

    # See diskint2_doc.ipynb for details. 
    # calculate the local intensity profile
    local_I = 1 + mu_LOS*bnu / (1+kappa*w_weak_shift)
    # These are the full expressions. 
    # leaving them here to make the code more easy to understand
    #geffBz = geff * B_LOS * Bpole/2.0 # The value of geff*Bz in gauss
    #local_V = perGaussLorentz * geffBz * ( mu_LOS*bnu*kappa*dw_weak_shift/(1+kappa*w_weak_shift)**2 )
            
    # Now, I moved some of the multiplications out of the disk integrationloop to shave down some time
    local_V = Bz * ( mu_LOS*dw_weak_shift/(1+kappa*w_weak_shift)**2 )

    return(local_I, local_V)

@njit
def get_local_weak_interp_Vonly_numba(small_u, w_weak, dw_weak, all_u, uLOS, mu_LOS, kappa, Bz):
    '''
    Function to get the local V only (for the loop) from the weak field approximation, by using an interpolation of a 
    fixed profile function in the rest frame. 

    :param small_u: the uo grid on which the profile is defined
    :param w_weak: the voigt profile (already divided by sqrt(2pi))
    :param dw_weak: the derivative of the voigt profile with respect to uo
    :param all_u: the (larger) dispersion grid on which to interpolate (enough to encompase the vsini) in uo units
    :param uLOS: the (rotational) radial velocity of the grid point in uo units
    :param mu_LOS: the mu angle of the grid point (angle between LOS and normal to surface)
    :param bnu: the slope of the source function with respect to tau_z (the vertical optical depth, the change in slope because of the mu angle is coded in)
    :param kappa: the line strenght parameter
    :param Bz: the Bz of that grid point (in Bpole/2 units)

    returns: local_V profile, used for disk integration. 
    The local_V need to be multiplied by the projected area (A_LOS) of the grid point during the integration.
    Furthermore, the local_V needs to be multiplied (can be done after the integration) by geff * Bpole/2 * kappa * perGaussLorentz,
    where perGaussLorentz(lambda0, vdop) is defined in the explanatory notebook. 
    '''

    # Shift the profiles to the right uLOS, and interpolate over the all_u velocity grid.         
    w_weak_shift  = rav_localV.interpol_lin_numba(w_weak,  small_u+uLOS, all_u)
    dw_weak_shift = rav_localV.interpol_lin_numba(dw_weak, small_u+uLOS, all_u)

    # See diskint2_doc.ipynb for details. 
    # calculate the local intensity profile
    # These are the full expressions. 
    # leaving them here to make the code more easy to understand
    #geffBz = geff * B_LOS * Bpole/2.0 # The value of geff*Bz in gauss
    #local_V = perGaussLorentz * geffBz * ( mu_LOS*bnu*kappa*dw_weak_shift/(1+kappa*w_weak_shift)**2 )
            
    # Now, I moved some of the multiplications out of the disk integrationloop to shave down some time
    local_V = Bz * ( mu_LOS*dw_weak_shift/(1+kappa*w_weak_shift)**2 )

    return(local_V)

def pad_model(model, input_ext_model, unno=True):
    '''
    Function to pad a model object with 1.0 (intensity) or 0.0 (Stokes) in order to accomodate a convolution. 

    :param model: the model to pad
    :param input_ext_model: the extended empty model object
    '''
    # Make a copy of the ext_model. It's a numpy recarray, so using numpy.copy
    ext_model = np.copy(input_ext_model)
    # interpolate the params to the new array
    ext_model['flux'] = np.interp(ext_model['uo'], model['uo'], model['flux'], left=1.0, right=1.0)
    ext_model['V'] = np.interp(ext_model['uo'], model['uo'], model['V'], left=0., right=0.)
    if unno==True:
        ext_model['Q'] = np.interp(ext_model['uo'], model['uo'], model['Q'], left=0., right=0.)
        ext_model['U'] = np.interp(ext_model['uo'], model['uo'], model['U'], left=0., right=0.)
    return(ext_model)

def pad_V(ext_all_u, all_u, V):
    '''
    Helper function for the loop that pads Stokes V to accomodate a convolution
    
    :param ext_all_u: the new uo array
    :param all_u: the existing uo array
    :param V: the Stokes V model defined on all_u
    '''
    ext_V = np.interp(ext_all_u, all_u, V, left=0., right=0.)
    return(ext_V)

def pad_I(ext_all_u, all_u, I):
    '''
    Helper function for the loop that pads Stokes I to accomodate a convolution
    
    :param ext_all_u: the new uo array
    :param all_u: the existing uo array
    :param I: the Stokes I model defined on all_u
    '''
    ext_I = np.interp(ext_all_u, all_u, I, left=1., right=1.)
    return(ext_I)


def get_resmac_kernel(all_u, uconv):
    '''
    Function to calculate a combined gaussian kernel for macroturbulence and spectral resolution
    Returns an astropy kernel object. 

    :param all_u: dispersion array in uo units on which the kernel will be calculated
    :param uconv: the sigma of the gaussian kernel in uo units. Should be set to ( u_rot**2 + u_res**2 )**0.5
    '''
    kernel = np.exp(-0.5*all_u**2/uconv**2)/(np.pi**0.5*uconv)
    kernel_object = conv.CustomKernel(kernel)
    return(kernel_object)

def model_convolve_resvmac(model, kernel, unno=True):
    '''Helper function to convolve a model with a kernel. 
    The flux convolution has a fill_value=1.0, and the Stokes convolution have fill_value=0.0. 
    Note -- this function does not extend the arrays. The checks on the kernel is performed in the calling function.
    
    :param model: a model object
    :param kernel: an astropy kernel object
    :param unno: (True) flag for a unno calculation. If True, will convolve Q and U. 
    '''
    model['flux'] = conv.convolve(model['flux'],kernel,boundary='fill',fill_value=1.0)
    model['V'] = conv.convolve(model['V'],kernel,boundary='fill',fill_value=0.0)
    if unno==True:
        model['Q'] = conv.convolve(model['Q'],kernel,boundary='fill',fill_value=0.0)
        model['U'] = conv.convolve(model['U'],kernel,boundary='fill',fill_value=0.0)
    return(model)


def numerical(param,unno=False, verbose=False):
    '''
    Function to calculated a line profile using a numerical disk integration, 
    either in the weak-field approximation (unno=False, default)
    or with the Unno solution (unno=True). 

    The necessary keys in the parameters dictionary object are:
    - general/vsini
    - general/logkappa
    - general/bnu
    - general/vdop
    - general/ndop
    - general/av
    - general/lambda0
    - general/Bpole
    - general/incl
    - general/beta
    - general/phase

    If general/res and general/vmac are not present, the former will be set to infinity and the latter to zero

    For the weak field approximation
    - weak/geff
    For the Unno calculation
    - unno/up
    - unno/down
    '''
    
   

    ###############################
    # Intro material
    ###############################

    # Checking the parameter dictionary for the necessary keywords
    rav_misc.check_req(param,'general')
    if unno==True:
        rav_misc.check_req(param,'unno')
    else:
        rav_misc.check_req(param,'weak')
    
    # get the sigma of the macroturbulence+spectral resolution gaussian kernel
    uconv = get_uconv(param)

    ngrid = get_ngrid(param['general']['vsini'], verbose=verbose)
    #ngrid = 50

    kappa = 10**param['general']['logkappa']    

    perGaussLorentz = get_perGaussLorentz(param['general']['lambda0'], param['general']['vdop'])
    # unno: This gets multiplied by B in the disk integration to get uB,i.
    # weak: This gets multiplied by geff*Bz and the Stokes V shape in the disk integration to get V(uo)
    
    sig = 10 # the width of the Voigt profile. 10 sigma is usually enough
    small_u = get_small_u(sig, param['general']['ndop'])

    if unno==True:
        if verbose: print('Evaluating with unno method...')
        # get the zeeman pattern
        pattern = rav_pattern.zeeman_pattern(param['unno']['down'],param['unno']['up'])
        # calculation the Voigt and Voigt-Faraday profiles
        w = get_w_unno(small_u, param['general']['av'], param['general']['ndop'])
        # Figure out the length of vector that we need for the velocity grid.
        vel_range = get_vel_range_unno(pattern, param['general']['Bpole'], perGaussLorentz,
                                       param['general']['vsini']/param['general']['vdop'],
                                       sig, verbose=True)
    else:
        if verbose: print('Evaluating with weak approximation...')
        # calculation the Voigt and Voigt-Faraday profiles
        w_weak, dw_weak = get_w_weak(small_u, param['general']['av'], param['general']['ndop'])
        # Figure out the length of vector that we need for the velocity grid.
        vel_range = param['general']['vsini']/param['general']['vdop']+sig
      
    # Get the total dispersion array
    all_u = get_all_u(vel_range, param['general']['ndop'], verbose=verbose)

    # Set up the model structure that will save the resulting spectrum
    model = get_empty_model(all_u, 
                            param['general']['vdop'], param['general']['lambda0'],
                            unno=unno)
    

    ##############################
    # Surface Grid Setup
    ##############################
    
    # cartesian coordinates of each grid points in ROT
    # and area of the grid element
    ROT, A = grid_ROT(ngrid) 

    ####################################
    # LOS Rotation Matrix Setup for a given incl and phase
    ####################################

    #rotation matrix that converts from LOS frame to ROT frame
    los2rot = get_los2rot(param['general']['incl'], param['general']['phase'])
    # the invert of a rotation matrix is the transpose
    rot2los = np.transpose(los2rot)
    # Get the grid coordinates in LOS and MAG systems. 
    LOS = get_LOS(ROT, rot2los)
    # get the necessary values for mu, the projected area, etc. 
    mu_LOS, vis, A_LOS, uLOS, conti_flux = get_LOS_values(LOS, A, param['general']['bnu'], param['general']['vsini'], param['general']['vdop'])

    ###########################
    # MAG Rotation matrix setup for a given beta
    ###########################

    # Rotation matrix betweeen the rot and the MAG frames
    #ROT frame to MAG frame
    rot2mag = get_rot2mag(param['general']['beta']) 
    #MAG to ROT frame
    mag2rot = np.transpose(rot2mag)
    # Getting the grid coordinates in the mag frame. 
    MAG = get_MAG(ROT, rot2mag)

    # Get the field values in Bpole/2 units
    B_MAG = get_B_MAG(MAG)
    # Transformation to the LOS frame
    B_LOS = get_B_LOS(B_MAG, mag2rot, rot2los)
    ## The the weak field, all we need is Bz, so B_LOS[z]*Bpole/2

    if unno == True:
        # but for unno, we need some of the angles
        # calcuate the angle necessary for the RT, mainly the angles between the field
        # direction and the LOS direction (theta) and the reference direction in the
        # plane of the sky (chi). (for the latter, the RT equations need cos(2phi))
        # See the explanatory notebook in the repository. 
        # Also returns the magnitude of the field in real units. 
        B, costheta_temp, sintheta_temp, cos2chi_temp, sin2chi_temp = get_unno_angles(B_LOS, param['general']['Bpole'])

    #############################
    # Disk Int time
    #############################

    # Making the two loops here. It repeat some of the code, 
    # but we don't want to have a check for unno==True for each grid point.
    # only iterating on the visible elements, to save calculations
    # checking how many True elements are in vis by slicing. 

    if unno==True:

        for i in range(0,vis[vis].size):
            out = rav_localV.unno_local_out_IV(
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
            # numerical integration (hence the projected area multiplication)
            model['flux'] = model['flux'] + A_LOS[vis][i]*out['I']
            model['Q']    = model['Q']    + A_LOS[vis][i]*out['Q']
            model['U']    = model['U']    + A_LOS[vis][i]*out['U']
            model['V']    = model['V']    + A_LOS[vis][i]*out['V']
    
    else:

        for i in range(0,vis[vis].size):
            local_I, local_V = get_local_weak_interp(small_u, w_weak, dw_weak, all_u,
                                          uLOS[vis][i], mu_LOS[vis][i], 
                                          param['general']['bnu'], kappa, 
                                          B_LOS[2,vis][i]  )

            # numerical integration (hence the projected area multiplication)
            model['flux'] += A_LOS[vis][i]*local_I
            model['V'] += A_LOS[vis][i]*local_V
        # constant stuff that I pulled out of the loop in get_local_weak_interp 
        model['V'] = model['V'] * param['weak']['geff'] * param['general']['Bpole']/2.0 * perGaussLorentz * param['general']['bnu']*kappa

    # Normalize the spectra to the continuum.
    model['flux'] /= conti_flux
    model['V'] /= conti_flux

    if unno == True:
        model['Q'] /= -1*conti_flux# sign flip to be consistent with Zeeman2
        model['U'] /= -1*conti_flux# sign flip to be consistent with Zeeman2

    # Make a convolution with spectral resolution and macroturbulence, if needed. 

    if uconv < 0.5:
        # if the width of the total kernel less than half of the thermal width, do nothing. 
        return(model)

    else:
        # We need to extend the model by 10*uconv to accomodate the convolution.
        ext_all_u = get_all_u(vel_range+10*uconv, param['general']['ndop'], verbose=verbose)
        # get an extended empty all_u array
        ext_model = get_empty_model(ext_all_u, 
                            param['general']['vdop'], param['general']['lambda0'],
                            unno=unno)
        # interpolate the params to the new array
        ext_model = pad_model(model, ext_model, unno=unno)
        
        # Get the gaussian kernel
        # Using the full extended array to calculate the kernel.
        kernel = get_resmac_kernel(ext_all_u, uconv)

        # make the convolutions
        ext_model = model_convolve_resvmac(ext_model, kernel, unno=unno)

        return(ext_model)

        













##############################
##############################
##############################
##############################

def get_rotKernel(uo, S1, urot):
    '''
    Helper function to get the rotational covolution profile

    :param uo: (numpy array) the velocity array in u0 units (v/vthermal)
    :param S1: the slope of the source function (see ADD LINK for conversion between the slope of the source function and epsilon)
    :param urot: the vsini in u0 units (vsini / vthermal)
    '''

    #VERO: shouldn't u0 be a numpy array?
    # Create an empty array with the same length as the passed non-broadened line profile
    # Note that the analytical function has already created an array that is long enough
    # to accomodate the rotatinal broadening. 
    kernel=np.zeros(uo.size)
    
    # The Kernel is only defined inside of vsini
    on = np.logical_and( uo>=-1*urot, uo<=urot )

    # Just an intermediate calculation here 
    # to make the next expression shorter
    yo = 1-(uo/urot)**2
    # This is the kernel function expressed in terms of the slope
    # of the source function (instead of the epsilon parameter of the grey textbook)
    # It's equivalent, see the details in the documentation notebook. 
    kernel[on] = (2*yo[on]**0.5 + 0.5*S1*np.pi*yo[on]) / (np.pi*urot*(1+2/3*S1))

    return(kernel)

def analytical(param, verbose=False):
    '''
    Function to calculate an intensity line profile using an analytical convolution

    The necessary keys in the parameters dictionary object are:
    - general/vsini
    - general/logkappa
    - general/bnu
    - general/vdop
    - general/ndop
    - general/av
    - general/lambda0

    Note: if the vsini is less then vdop/2, then the function directly returns the non-rotating line profile. 

    :param param: a parameters dictionary
    '''
    # Original version by Patrick Stanley
    # Modified slightly by Vero, to use the diskint functions
    # to match the other calculations. 
    # Added a case for vsini=0

    rav_misc.check_req(param,'analytic')

    uconv = get_uconv(param)

    # shortcut for parameters
    kappa = 10**param['general']['logkappa'] 
    S1 = param['general']['bnu']



    #calculate the vsini in u0 units
    urot = param['general']['vsini'] / param['general']['vdop'] 
            
    #sets up the u0 axis to accomodate the rotational broadening 
    # and the thermal width of the spectral line
    sig = 10 # the width of the Voigt profile. 10 sigma is usually enough
    vel_range = urot+sig    
    # generate the uo array to have ndop points per thermal width. 
    all_u = get_all_u(vel_range+10*uconv, param['general']['ndop'], verbose=verbose)
    
    # Set up the model structure that will save the resulting spectrum
    model = get_empty_model(all_u, 
                            param['general']['vdop'], param['general']['lambda0'],
                            unno=False)

    # Calculate the non-broadened continnum-normalized line flux using the analytical expression
    # that takes into account the variation of the line intensity across the stellar disk. 
    # see the documentation notebook for more details. 
    phi = rav_profileI.voigt_fara(all_u, param['general']['av']).real/np.pi**0.5
    flux=(1.0+2.0/3.0*S1/(1.0+kappa*phi)) / (1+2.0/3.0*S1)

    # Check if the kernel is smaller than half of the thermal width
    if urot < 0.5:
        # if yes, just return the unbroadened profile. 
        model['flux']=flux #flux
        return(model)

    else:

        #get the rotation kernel
        #-------
        #VERO: the kernel does not have to be the
        # same size as the all_u, as it is bounded by zeros. 
        # it only need to be vsini/vdop wide.
        # Could speed things up if needed. 
        #-------
        # The kernel is calculated on the same uo grid as the profile. 
        # As we have set the range of the uo array to accomodate all of the broadning
        # the kernel will fit on that array. 
        # Furthermore, we made sure than the kernel will have at least one 
        # some data point (assuming that ndop is > 1.0, which should be the case)
        kernel_array = get_rotKernel(all_u,S1,urot)
        # if the size of the kernel is even
        # add one element. 
        if kernel_array.size%2 == 0:
            kernel_array=np.append(kernel_array,kernel_array[-1])

        #Create a kernel object from the kernel array
        kernal_object= conv.CustomKernel(kernel_array)

        # Perform the convolution using astropy convolution tools. 
        # The continnum is expected to be at 1.0 outside of the line profile
        # so we use the fill_value parameter. This prevents the wings of the Voigt profile
        # to be interpolated in a weird way. 
        rot=conv.convolve(flux,kernal_object,boundary='fill',fill_value=1.0)
                    
        model['flux']=rot #flux


    if uconv < 0.5:
        # if the width of the total kernel less than half of the thermal width, do nothing. 
        return(model)

    else:

        # Get the gaussian kernel
        # Using the full extended array to calculate the kernel.
        kernel = get_resmac_kernel(all_u, uconv)

        # make the convolutions
        model = model_convolve_resvmac(model, kernel, unno=False)
    
        return(model)
