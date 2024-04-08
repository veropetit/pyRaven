import numpy as np
from scipy.special import erf
from astropy.convolution import Gaussian1DKernel, convolve
import astropy.convolution as con

import pyRaven as rav

'''
The input for both diskint.strong and diskint.weak is of the form:
genparam = {
    'lambda0':5811.969,    # the central wavelength of the transition
    'vsini':50.0,         # the projected rotational velocity
    'vdop':10.0,          # the thermal broadening
    'av':0.05,             # the damping coefficient of the Voigt profile
    'bnu':1.5,             # the slope of the source function with respect to vertical optical depth
    'logkappa':10**0.98,          # the line strength parameter
    'Bpole':1.0e5,         # the dipolar field strength
    'incl':np.pi/4,      # the inclination of the rotational axis to the line of sight
    'beta':np.pi/4,      # the obliquity of the magnetic axis to the rotational axis
    'phase':0.0,     # the rotational phase
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

def numerical(param,unno, verbose=True):
    ngrid = 1000 + 20*param['general']['vsini']
    kappa = 10**param['general']['logkappa']    

    if unno==True:
        rav.misc.check_req(param,'unno')
        if verbose: print('Evaluating with unno method...')
    
        const = { 'larmor' : 1.3996e6,\
                'c' : 2.99792458e5 } #cgs units
    
        # multiply by B(gauss) to get the Lorentz unit in units of vdop
        value = { 'lorentz':param['general']['lambda0']*1e-13*const['larmor']/param['general']['vdop'] }
    
        # get the zeeman pattern
        pattern = rav.pattern.zeeman_pattern(param['unno']['down'],param['unno']['up'])
    
        ## Doppler velocity grid setup
        # Compute the Voigt functions only once.
        # Up to 11 vdop (keeping 1 vdop on each side to padd with zeros.
    
    
        small_u = np.linspace(-15,15,15*param['general']['ndop']*2)
    
        # To calculate
        w = rav.profileI.voigt_fara(small_u,param['general']['av'])
    
        # padding with zero on each ends
        w[0:param['general']['ndop']]=0.0
        w[w.size-param['general']['ndop']:w.size]=0.0
    
    
        # Figure out the length of vector that we need for the velocity grid.
        # Take the max of the Zeeman shift or the vsini, then add the 15vdop width.
        # to accomodate the whole w(small_u) array shifted to the maximum value possible
        max1 = np.max(np.abs(pattern['sigma_r']['split']))
        max2 = np.max(np.abs(pattern['pi']['split']))
        max_b = np.max([max1,max2])*param['general']['Bpole']*value['lorentz']
        max_vsini = param['general']['vsini']/param['general']['vdop']
    
        if verbose:
            print('Max shift due to field: {} vdop'.format(max_b))
            print('Max shift due to vsini: {} vdop'.format(max_vsini))
    
        vel_range = np.max( [max_b+15, max_vsini+15] )
        vrange = np.round(vel_range+1)
        if verbose: print('Max velocity needed: {} vdop'.format(vel_range))
    
        all_u = np.linspace( -1*vrange,vrange,int(param['general']['ndop']*vrange*2+1))
        if verbose: print('Number of grid points: {}'.format(all_u.size))
    
        # Set up the model structure that will save the resulting spectrum
        model = np.zeros(all_u.size,
                        dtype=[('wave',float),('vel',float),('vdop',float),
                            ('flux',float),('fluxnorm', float),('Q',float),('U',float),('V',float)])
                            
        model['vdop'] = all_u
        model['vel'] = all_u * param['general']['vdop']
        model['wave'] = (model['vel']/const['c'])*param['general']['lambda0']+param['general']['lambda0']
    
        ##Rotation Matrix Setup
    
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
        #Converts from ROT to LOS frame
        rot2los = np.transpose(los2rot)
    
        #ROT frame to MAG frame
        rot2mag = np.array( [[ 1.0, 0.0, 0.0],
                    [ 0.0, np.cos(param['general']['beta']),np.sin(param['general']['beta'])],
                    [ 0.0, -1*np.sin(param['general']['beta']),np.cos(param['general']['beta'])] ] )
    
        #MAG to ROT frame
        mag2rot = np.transpose(rot2mag)
    
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
    
        LOS = np.matmul(rot2los,ROT)
        MAG = np.matmul(rot2mag,ROT)
    
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
    
        # value of the continuum intensity on each grid point
        # conti_int = ( 1.+ mu_LOS*bnu )
        # scaled by the area
        conti_int_area = A_LOS*(1.0+mu_LOS*param['general']['bnu'])
        conti_flux = np.sum(conti_int_area)# continuum flux (for profile normalisation)
    
        # radial velocity in doppler units
        uLOS = param['general']['vsini']*LOS[0,:] / param['general']['vdop']
    
        ##Surface B-Field
    
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
        # calcuate the angle necessary for the RT, mainly the angles between the field
        # direction and the LOS direction (theta) and the reference direction in the
        # plane of the sky (chi)
        sintheta_temp = np.sqrt( B_LOS[0,:]**2+B_LOS[1,:]**2)
        costheta_temp = 1.0*B_LOS[2,:]
        sin2chi_temp = 2.0*B_LOS[2,:]*B_LOS[1,:]/sintheta_temp**2
        cos2chi_temp = 2.0*B_LOS[0,:]**2/sintheta_temp**2-1.0
    
        # Scale the unit vector by the field strength
        # to get the magnitude of the field
        B = B_unit * param['general']['Bpole']/2.0
    
    
        ##Disk Int time
    
        # only iterating on the visible elements, to save calculations
        for k in range(0,vis.size):
            i = vis[k] # index in the grid (to save some typing)
            out = rav.localV.unno_local_out_IV(
                        B[i]*value['lorentz'],# uB
                        uLOS[i],              # uLOS
                        pattern,              # the zeeman pattern
                        small_u,              # the uo array for the non-magnetic profile
                        w,                    # th Voigt and Voigt-Fara profiles
                        all_u,                # the complete uo array
                        sintheta_temp[i], costheta_temp[i], # the angles of the field to the LOS
                        sin2chi_temp[i], cos2chi_temp[i], # the angles of the field to the LOS
                        kappa, param['general']['bnu'], # the line strenght parameter and the slope of the source function
                        mu_LOS[i]             # the angle between the surface and the LOS
                        )
                    
            model['flux'] += A_LOS[i]*out['I']
            model['Q'] += A_LOS[i]*out['Q']
            model['U'] += A_LOS[i]*out['U']
            model['V'] += A_LOS[i]*out['V']
    
        # Normalize the spectra to the continuum.
        model['flux'] /= conti_flux
        model['Q'] /= -1*conti_flux# sign flip to be consistent with Zeeman2
        model['U'] /= -1*conti_flux# sign flip to be consistent with Zeeman2
        model['V'] /= conti_flux
    
    if unno==False:
        rav.misc.check_req(param,'weak')
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
        if verbose: print('Max velocity needed: {} vdop'.format(vel_range))
        
        all_u = np.linspace( -1*vrange,vrange,int(param['general']['ndop']*vrange*2+1))

        # To calculate
        w = rav.profileI.voigt_fara(all_u,param['general']['av'])
        voigt = w.real
        fara=w.imag

        shape_V = np.gradient(w.real, all_u)
        profile_v_large = param['general']['bnu'] *kappa*geff*shape_V/np.pi**0.5 / (1+kappa/np.pi**0.5*w.real)**2

        profile_i_large = param['general']['bnu']/(1.0+kappa*voigt/np.sqrt(np.pi))


        # Set up the model structure that will save the resulting spectrum
        model = np.zeros(all_u.size,
                        dtype=[('wave',float),('vel',float),('vdop',float),
                            ('flux',float),('fluxnorm', float),('Q',float),('U',float),('V',float)])
                            
        model['vdop'] = all_u
        model['vel'] = all_u * param['general']['vdop']
        model['wave'] = (model['vel']/const['c'])*param['general']['lambda0']+param['general']['lambda0']
    
        ##Rotation Matrix Setup
    
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
        #Converts from ROT to LOS frame
        rot2los = np.transpose(los2rot)
    
        #ROT frame to MAG frame
        rot2mag = np.array( [[ 1.0, 0.0, 0.0],
                    [ 0.0, np.cos(param['general']['beta']),np.sin(param['general']['beta'])],
                    [ 0.0, -1*np.sin(param['general']['beta']),np.cos(param['general']['beta'])] ] )
    
        #MAG to ROT frame
        mag2rot = np.transpose(rot2mag)
    
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
    
        LOS = np.matmul(rot2los,ROT)
        MAG = np.matmul(rot2mag,ROT)
    
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

        ##Surface B-Field
    
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

        ##Disk Int
    
        # only iterating on the visible elements, to save calculations
        for k in range(0,vis.size):
            i = vis[k] # index in the grid (to save some typing)

            prof_V = rav.localV.interpol_lin(profile_v_large,all_u+uLOS[i],model['vdop'])
            model['V'] += A_LOS_V[i]*B_LOS[2,i]*prof_V*B[i]

            prof_I = 1.0+ mu_LOS[i] * rav.localV.interpol_lin(profile_i_large,all_u+uLOS[i],model['vdop'])
            model['flux'] += A_LOS_I[i]*prof_I
        
    
        model['V'] = model['V']*value['lorentz']/conti_flux
        model['V'] = model['V']*np.pi/2
    
        model['flux'] = model['flux']/conti_flux
    
    return(model, ROT, LOS, MAG)

def analytical(param, verbose=True):
    
    #models the line profile by convolving the voigt fara function with the rotation profile
    rav.misc.check_req(param,'weak')
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
    
    if verbose: print('Max shift due to vsini: {} vdop'.format(urot))
    
    #sets up the u axis
    vel_range = np.max( [15, urot+15] )
    vrange = np.round(vel_range+1+urot)
    if verbose: print('Max velocity needed: {} vdop'.format(vel_range))
    
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
