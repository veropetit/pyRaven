import numpy as np
from scipy.special import erf
from astropy.convolution import Gaussian1DKernel, convolve
import astropy.convolution as con

import pyRaven as rav

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
        'Bgrid': np.arange(0,5000, 500),
        'igrid': np.arange(0,180,20),
        'betagrid':np.arange(0,180,20),
        'phasegrid':np.arange(0,360,20)
        }
    
param={'general' : genparam,
       'weak' : weakparam,
       'grid' : gridparam,
       }
'''

def loop(param,datapacket):

    #############################################
    ## Things that only need to be computed once
    #############################################
    
    ngrid = 1000 # set this based on the vsini later.
    
    const = { 'larmor' : 1.3996e6,\
            'c' : 2.99792458e5,\
            'a2cm' : 1.0e-8,\
            'km2cm' : 1.0e5\
    }

    geff=param['weak']['geff']

    # multiply by B(gauss) to get the Lorentz unit in units of vdop
    value = { 'lorentz':param['general']['bnu']*geff/np.sqrt(np.pi)\
        *(const['a2cm']*param['general']['lambda0'])/(param['general']['vdop']*const['km2cm'])*const['larmor']}

    urot=param['general']['vsini']/param['general']['vdop']

    ## Doppler velocity grid setup
    # Compute the Voigt functions only once.
    # Up to 11 vdop (keeping 1 vdop on each side to padd with zeros.

    vel_range = np.max( [15, urot+15] )
    vrange = np.round(vel_range+1)
    print('Max velocity needed: {} vdop'.format(vel_range))
    
    all_u = np.linspace( -1*vrange,vrange,int(param['general']['ndop']*vrange*2+1))

    w = rav.profileI.voigt_fara(all_u,param['general']['av'])

    shape_V = np.gradient(w.real, all_u)
    profile_v_large = param['general']['bnu'] *kappa*geff*shape_V/np.pi**0.5 / (1+kappa/np.pi**0.5*w.real)**2
    profile_i_large = param['general']['bnu']/(1.0+kappa*voigt/np.sqrt(np.pi))


    # Set up the model structure that will save the resulting spectrum
    model = np.zeros(all_u.size,
                    dtype=[('vel',float),('vdop',float),
                        ('flux',float),('fluxnorm', float),('V',float)])
                        
    model['vdop'] = all_u
    model['vel'] = all_u * param['general']['vdop']

    ##Surface Grid Setup

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

    ###############################
    # Loop on inclination and phase
    ###############################

    d2r = np.pi/180.0

    for i, incl in param['grid']['igrid']:
        for p, phase in param['grid']['phasegrid']:

            ##Rotation Matrix Setup

            #rotation matrix that converts from LOS frame to ROT frame
            los2rot = np.array( [[np.cos(phase*d2r),
                                np.cos(incl*d2r)*np.sin(phase*d2r),
                                -1*np.sin(param['general']['incl'])*np.sin(phase*d2r)],
                                [-1*np.sin(phase*d2r),
                                np.cos(incl*d2r)*np.cos(phase*d2r),
                                -1*np.sin(incl*d2r)*np.cos(phase*d2r)],
                                [0.0,
                                np.sin(incl*d2r),
                                np.cos(incl*d2r)] ] )

            # the invert of a rotation matrix is the transpose
            #Converts from ROT to LOS frame
            rot2los = np.transpose(los2rot)

            # Calculate the LOS grid coordinate.
            LOS = np.matmul(rot2los,ROT)

            # Calulating values on the grid
            # visible elements where z component ge 0
            # mu = cos angle between normale to surface and LOS
            # mu = z_LOS / 1
            # the 1.0 is to avoid a symbolic association
            mu_LOS=1.0*LOS[2,:]
            vis = np.where(mu_LOS >= 0.0)[0]#index where grid points are visible
            Aneg = mu_LOS < 0.0#True if invisible
            mu_LOS[Aneg==True] = 0.0#just put all the unvisible values to zero

            # projected surface area in the LOS
            # thus not visible surface points also have zero A_LOS
            A_LOS = A * mu_LOS
            A_LOS /= np.sum(A_LOS)# Normalized to unity so that the integral over projected area=1.

            A_LOS_V = A_LOS * mu_LOS
            A_LOS_I = 1.0 * A_LOS # *1.0 is to force a copy

            # value of the continuum intensity on each grid point
            # conti_int = ( 1.+ mu_LOS*bnu )
            # scaled by the area
            conti_int_area = A_LOS*(1.0+mu_LOS*param['general']['bnu'])
            conti_flux = np.sum(conti_int_area)# continuum flux (for profile normalisation)

            # radial velocity in doppler units
            uLOS = param['general']['vsini']*LOS[0,:] / param['general']['vdop']

            ############################
            # Loop over beta values
            ############################

            for b, beta in param['grid']['betagrid']:


                ##Surface B-Field
                #ROT frame to MAG frame
                rot2mag = np.array( [[ 1.0, 0.0, 0.0],
                            [ 0.0, np.cos(beta*d2r),np.sin(beta*d2r)],
                            [ 0.0, -1*np.sin(beta*d2r),np.cos(beta*d2r)] ] )

                #MAG to ROT frame
                mag2rot = np.transpose(rot2mag)

                MAG = np.matmul(rot2mag,ROT)

                # check for sinthetab =0
                # to avoid dividing by zero in the B_MAG calculations below.
                sinthetab = np.sqrt(MAG[0,:]**2 + MAG[1,:]**2)
                no_null = np.where(sinthetab>0)
                B_MAG = np.zeros((3,int(real_n)))

                # B in (Bpole/2) units in MAG coordinate
                B_MAG[0,no_null] = 3*MAG[2,no_null]*sinthetab[no_null]*MAG[0,no_null]/sinthetab[no_null]
                B_MAG[1,no_null] = 3*MAG[2,no_null]*sinthetab[no_null]*MAG[1,no_null]/sinthetab[no_null]
                B_MAG[2,:] = 3*MAG[2,:]**2-1.0

                # B in LOS (in Bp/2 units)
                # to get the angles necessary for the transfert equations
                # calcuate the magnitde of the field vectors (in Bp/2 u
                B_unit = np.sqrt(B_MAG[0,:]**2+B_MAG[1,:]**2+B_MAG[2,:]**2)
                B_LOS = np.matmul(rot2los,np.matmul(mag2rot,B_MAG))
                # divide by the magnitude to get the unit vectors in the direction of the field in LOS
                for i in range(3):
                    B_LOS[i,:] /= B_unit

                # Scale the unit vector by the field strength
                # to get the magnitude of the field
                B = B_unit * param['general']['Bpole']/2.0

                ##Disk Integration

                # only iterating on the visible elements, to save calculations
                for k in range(0,vis.size):
                    i = vis[k] # index in the grid (to save some typing)

                    prof_V = rav.localV.interpol_lin(profile_v_large,all_u+uLOS[i],model['vdop'])
                    model['V'] += A_LOS_V[i]*B_LOS[2,i]*prof_V*B[i]

                model['V'] = model['V']*value['lorentz']/conti_flux
                model['V'] = model['V']*np.pi/2  #  WHAT IS THE PI/2 HERE?
                
                # Loop over the observations in the Data Packet
                
                # convolve the model over the instrumental resolution
                
                # Interpolate the model to the dispersion of the data
                
                # calculate the chi square between data and model
                
                # Store the chi2 in the data cube.


    return
