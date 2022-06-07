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

