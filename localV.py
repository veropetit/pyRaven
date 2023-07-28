import numpy as np
from numba import njit

@njit
def interpol_lin_numba(v, x, u):
    # V: The input vector can be any type except string.

    # Irregular grids:
    # X: The absicissae values for V.  This vector must have same # of
    # elements as V.  The values MUST be monotonically ascending
    # or descending.
    #
    # U: The absicissae values for the result.  The result will have
    #  the same number of elements as U.  U does not need to be
    #  monotonic.  If U is outside the range of X, then the
    #  closest two endpoints of (X,V) are linearly extrapolated.

    s = np.searchsorted(x,u)-1
    s = np.clip(s, 0, v.size-2)
    
    p = (u-x[s])*(v[s+1]-v[s])/(x[s+1] - x[s]) + v[s]

    return(p)

def interpol_lin(v, x, u):
    # V: The input vector can be any type except string.

    # Irregular grids:
    # X: The absicissae values for V.  This vector must have same # of
    # elements as V.  The values MUST be monotonically ascending
    # or descending.
    #
    # U: The absicissae values for the result.  The result will have
    #  the same number of elements as U.  U does not need to be
    #  monotonic.  If U is outside the range of X, then the
    #  closest two endpoints of (X,V) are linearly extrapolated.

    s = np.searchsorted(x,u)-1
    s = np.clip(s, 0, v.size-2)
    
    p = (u-x[s])*(v[s+1]-v[s])/(x[s+1] - x[s]) + v[s]


    return(p)


def unno_local_out_IV(ub, ua, pattern, small_u, w, all_u, sin_theta, cos_theta, sin_2chi, cos_2chi, kappa, bnu, mu):
    '''
    Calculate a local line profile in all Stokes paramters.
    
    :param ub: the lambda_B in doppler width units.
    :param ua: local radial velocity in doppler width units.
    :param pattern: a zeeman pattern dictionary returned by the zeeman_pattern function
    :param small_u: the uo array for the w array that contains the centered line profile function
    :param w: the line profile function centered at uo=0, calculated over small_u (the profile functions are shifted and interpolated for the addition of all the zeeman components, instead of recalculated, to same computation time).
    :param all_u: the full wavelength array in doppler width units. It need to be larger enough to accomodate the largest zeeman and rotational shifts.
    :param sin_theta: sin of the angle between the local magnetic field and the LOS.
    :param cos_theta: cos of the angle between the local magnetic field and the LOS.
    :param sin_2chi: sin of 2x the angle between the local magnetic field and the reference direction in the plane of the sky
    :param cos_2chi: cos of 2x the angle between the local magnetic field and the reference direction in the plane of the sky
    :param kappa: the line strength parameter
    :param bnu: the slope of the source function with respect to the vertical optical depth
    :param mu: the angle between the normal to the surface and the LOS. 
    

    '''

    eta_rho = np.zeros( all_u.size,
                          dtype=[('pi', complex), ('sigma_r', complex),('sigma_b', complex) ]  )

    for i in range(0, pattern['n_pi']):

        eta_rho['pi'] += pattern['pi'][i]['S'] * interpol_lin(w, small_u + ua - ub*(-1)*pattern['pi'][i]['split'], all_u)
        #Note, the signs of the u are inversed because it is the velocity grid that is shifted.

    eta_rho['pi'] *= 1.0/np.sqrt(np.pi)

    for i in range(0, pattern['n_sigma']):

        eta_rho['sigma_r'] += pattern['sigma_r'][i]['S'] * interpol_lin(w, small_u + ua - ub*(-1)*pattern['sigma_r'][i]['split'], all_u)
        eta_rho['sigma_b'] += pattern['sigma_b'][i]['S'] * interpol_lin(w, small_u + ua - ub*(-1)*pattern['sigma_b'][i]['split'], all_u)
        
    eta_rho['sigma_r'] *= 1.0/np.sqrt(np.pi)
    eta_rho['sigma_b'] *= 1.0/np.sqrt(np.pi)



    hr = np.zeros( eta_rho.size,
                          dtype=[('I', complex), ('Q', complex),('U', complex),('V', complex) ]  )

    hr['I'] = 0.5* (eta_rho['pi'] * sin_theta**2 + (eta_rho['sigma_b']+eta_rho['sigma_r'])/2.0 * (1+cos_theta**2) )

    hr['Q'] = 0.5* ( eta_rho['pi'] - (eta_rho['sigma_b']+eta_rho['sigma_r'])/2.0 ) * sin_theta**2 * cos_2chi

    hr['U'] = 0.5* ( eta_rho['pi'] - (eta_rho['sigma_b']+eta_rho['sigma_r'])/2.0 ) * sin_theta**2 * sin_2chi

    hr['V'] = 0.5* ( eta_rho['sigma_r'] - eta_rho['sigma_b'] ) * cos_theta


    kI = (kappa * hr['I'].real).astype(np.float64)
    kQ = (kappa * hr['Q'].real).astype(np.float64)
    kU = (kappa * hr['U'].real).astype(np.float64)
    kV = (kappa * hr['V'].real).astype(np.float64)
    fQ = (kappa * hr['Q'].imag).astype(np.float64)
    fU = (kappa * hr['U'].imag).astype(np.float64)
    fV = (kappa * hr['V'].imag).astype(np.float64)

    nabla = (1+kI)**4 + (1+kI)**2 * (fQ**2 + fU**2 + fV**2 - kQ**2 - kU**2 - kV**2) - (kQ*fQ + kU*fU + kV*fV)**2

    # For IQUV
    local_out = np.zeros( hr.size,
                         dtype=[('I', float), ('Q', float),('U', float),('V', float) ]  )
    # For IV only
    #local_out = np.zeros( hr.size,
    #                      dtype=[('I', float),('V', float) ]  )


    # Total
    local_out['I'] =  1.0 + bnu*mu * nabla**(-1) * (1.0+kI) * ( (1.0+kI)**2 + fQ**2 + fU**2 + fV**2)
    local_out['Q'] = -1*bnu*mu * nabla**(-1) * ( (1.0+kI)**2 * kQ - (1.0+kI) * (kU*fV - kV*fU) + fQ * (kQ*fQ + kU*fU + kV*fV) )
    local_out['U'] = -1*bnu*mu * nabla**(-1) * ( (1.0+kI)**2 * kU - (1.0+kI) * (kV*fQ - kQ*fV) + fU * (kQ*fQ + kU*fU + kV*fV) )
    local_out['V'] = -1*bnu*mu * nabla**(-1) * ( (1.0+kI)**2 * kV + fV*(kQ*fQ + kU*fU + kV*fV) )



    return local_out
