import numpy as np

def unno_local_out_IV(ub, ua, pattern, small_u, w, all_u, sin_theta, cos_theta, sin_2chi, cos_2chi, kappa, bnu, mu):


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
