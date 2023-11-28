import numpy as np

def voigt_fara(u, a):
    '''
    Calculation of Voigt and Faraday-Voigt profiles

        :param u: array of frequency in doppler width units.
        :param a: damping coefficient
        :return w: complex array where Voigt is the real component and Voigt-Fara is the imaginary component.

    '''

    # This function computes the H and L profiles.
    # The real part of w is H and the imaginary part is L (Landi p.163, eqs 5.45)
    # The implementation comes from http://adsabs.harvard.edu/abs/1982JQSRT..27..437H

    z = u + 1j*a
    t_arr = a -1j*u
    s = np.abs(u) + a

    w = np.empty(u.shape, dtype=complex)

    for i in range(0,u.size):
        t=t_arr[i]

        if (s[i] >= 15.0):
            #region 1
            #print i, 'region 1'
            w[i] = 0.5641896*t/(0.5 + t**2)
        else:
            if (s[i] >= 5.5):
                #region 2
                #print i, 'region 2'
                w[i] = t*(1.410474 + 0.5641896*t**2)/((3.0 + t**2)*t**2 + 0.75)
            else:
                if (z[i].imag >= (0.195*abs(z[i].real) - 0.176) ):
                    # region 3
                    #print i, 'région 3'
                    w[i] = (16.4955 + t*(20.20933 + t*(11.96482 + t*(3.778987 + t*0.5642236))))/ (16.4955 + t*(38.82363 + t*(39.27121 + t*(21.69274 + t*(6.699398 + t)))))
                else:
                    #région 4
                    #print i, 'région 4'
                    w[i] = np.exp(t**2) - t*(36183.31 - t**2* (3321.9905 - t**2* (1540.787 - t**2*(219.0313 - t**2* (35.76683 - t**2* (1.320522 - t**2*0.56419))))))/ (32066.6 - t**2* (24322.84 - t**2*(9022.228 - t**2* (2186.181 - t**2* (364.2191 - t**2*(61.57037 - t**2* (1.841439 - t**2)))))))

    return w

def voigt_fara_mat(u_matrix, a):
    '''
    Calculation of Voigt and Faraday-Voigt profiles and its derivative with respect to u with a arbritrary shaped arrray of u.
    This function uses indexations instead of a loop, 
    and is suitable for quick calculation of the Voigt profiles on large grids 
    for the weak field approximation. 

    NOTE: the returned Voigt and Faraday-Voigt is NOT normalized by sqrt(pi)

        :param u: X-D array of frequency in doppler width units.
        :param a: damping coefficient
        :return w: complex X-D array where Voigt is the real component and Voigt-Fara is the imaginary component.

    '''
    t = a -1j*u_matrix

    voigt_mat = np.empty(u_matrix.shape, dtype=complex)

    # Voigt region 1
    n = (np.abs(u_matrix)+a) >= 15.0
    voigt_mat[n] = 0.5641896*t[n]/(0.5 + t[n]**2)

    # Voigt region 2
    n = np.logical_and( (np.abs(u_matrix)+a) < 15.0, (np.abs(u_matrix)+a) > 5.5 )
    voigt_mat[n] = t[n]*(1.410474 + 0.5641896*t[n]**2)/((3.0 + t[n]**2)*t[n]**2 + 0.75)

    # Voigt region 3
    n = np.logical_and( (np.abs(u_matrix)+a) < 5.5, (u_matrix + 1j*a).imag >= (0.195*abs((u_matrix + 1j*a).real) - 0.176) )
    voigt_mat[n] = (16.4955 + t[n]*(20.20933 + t[n]*(11.96482 + t[n]*(3.778987 + t[n]*0.5642236))))/ (16.4955 + t[n]*(38.82363 + t[n]*(39.27121 + t[n]*(21.69274 + t[n]*(6.699398 + t[n])))))

    # Voigt region 4
    n = np.logical_and( (np.abs(u_matrix)+a) < 5.5, (u_matrix + 1j*a).imag < (0.195*abs((u_matrix + 1j*a).real) - 0.176) )
    voigt_mat[n] = np.exp(t[n]**2) - t[n]*(36183.31 - t[n]**2* (3321.9905 - t[n]**2* (1540.787 - t[n]**2*(219.0313 - t[n]**2* (35.76683 - t[n]**2* (1.320522 - t[n]**2*0.56419))))))/ (32066.6 - t[n]**2* (24322.84 - t[n]**2*(9022.228 - t[n]**2* (2186.181 - t[n]**2* (364.2191 - t[n]**2*(61.57037 - t[n]**2* (1.841439 - t[n]**2)))))))

    return voigt_mat
       
def get_voigt_weak(u_matrix, a):
    '''
    Calculation of Voigt profile and its derivative with respect to u with a arbritrary shaped arrray of u.
    This function uses indexations instead of a loop, 
    and is suitable for quick calculation of the Voigt profiles on large grids 
    for the weak field approximation.
    
    NOTE: the returned Voigt and its derivative IS normalized by sqrt(pi)

    :param u: X-D array of frequency in doppler width units.
    :param a: damping coefficient
    :return voigt_mat and dvoigt_mat: real X-D array with Voigt and the derivative with respect to u.
    '''

    # get the imaginary Voigt function
    w_mat = voigt_fara_mat(u_matrix, a)

    #From Landi & Landofi 2005, eq 5.58
    #dHdv = (2.*(-v*w4.real + anm*w4.imag))
    dvoigt_mat = 2*(-1*u_matrix*w_mat.real + a*w_mat.imag)/np.sqrt(np.pi)
    voigt_mat = w_mat.real/np.sqrt(np.pi)

    return(voigt_mat, dvoigt_mat)
