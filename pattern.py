import numpy as np
import matplotlib.pyplot as plt



def zeeman_pattern(down, up, **lande_parameters):
    
# Input parameter
# up   - S, L, J of upper level
# down - S, L, J of down level
# g_u  - lande factor of upper level (if not given computed in LS coupling and returned)
# g_d  - lande factor of down level (if not given computed in LS coupling and returned)

# up and down levels

    s_u = float(up[0])
    s_d = float(down[0])

    l_u = float(up[1])
    l_d = float(down[1])

    j_u = float(up[2])
    j_d = float(down[2])
    
# calculate the Lande factor in LS coupling if not given
    
    if ('g_u' in lande_parameters):
        print('overwriting LS lande for up-level level ', lande_parameters['g_u'])
    else:
        if j_u == 0:
            g_u=0
        else:
            g_u = 3.0/2.0 + ( s_u * (s_u+1.0) - l_u * (l_u+1.0) ) / ( 2*j_u * ( j_u+1 ) )
        #print 'Calculating up-level lande in LS coupling: {}'.format(g_u)

        
    if ('g_d' in lande_parameters):
        print('overwriting LS lande for down-level level ', lande_parameters['g_d'])
    else:
        if j_d == 0:
            g_d=0
        else:
            g_d = 3.0/2.0 + ( s_d * (s_d+1.0) - l_d * (l_d+1.0) ) / ( 2*j_d * ( j_d+1 ) )
        #print 'Calculating down-level lande in LS coupling: {}'.format(g_d)
    
    geff = 0.5 * (g_u+g_d) + 0.25 * (g_u-g_d) * ( j_u*(j_u+1) - j_d*(j_d+1) )
    #print 'Triplet approximation geff: {}'.format(geff)

# which level has the lowest J
    j_min = min([j_u,j_d])
    #print 'j_min: {}'.format(j_min)

# create an array with the m for the level with lowest m (the only allaowed transitions are Delta_m = 0, +-1
    #m_min = indgen(j_min*2+1, /double)-j_min
    m_min = np.arange( -j_min, j_min+1 )
    
    m_min = m_min[::-1] # to go from the highest m to the lowest m
    #print 'm_min: should go from {} to {} in dereasing order: {}'.format(-j_min, j_min, m_min)
    
# number of transitions
    n_pi = int(2*j_min+1)
    n_sigma = int(j_u + j_d)
    #print '{} Pi transitions'.format(n_pi)
    #print '{} Sigma transitions'.format(n_sigma)
    
# pi
    
    pi = np.recarray( n_pi, dtype=[('split', float), ('S', float)] )
    sigma_r = np.recarray( n_sigma, dtype=[('split', float), ('S', float)] )
    sigma_b = np.recarray( n_sigma, dtype=[('split', float), ('S', float)] )
    
    if round(j_u - j_d) == 0:
        #print 'levels of same j'

        for i in range(0, n_pi):
            pi['split'][i] = m_min[i] * ( g_d - g_u )
            pi['S'][i] = m_min[i]**2
        for i in range(0, n_sigma):
            sigma_b['split'][i] = m_min[i] * ( g_d - g_u ) - g_d
            sigma_r['split'][i] = m_min[i+1] * ( g_d - g_u ) + g_d
            sigma_b['S'][i] = 0.5*(j_min+m_min[i])*(j_min-m_min[i]+1) # en partant du m le plus haut des transitions sigma b
            sigma_r['S'][i] = 0.5*(j_min-m_min[i+1])*(j_min+m_min[i+1]+1)
    
    elif round(j_u - j_d) == 1:
        #print 'j_min is the lower level'
        
        for i in range(0, n_pi):
            pi['split'][i] = m_min[i] * ( g_d - g_u )
            pi['S'][i] = (j_min+1)**2 - m_min[i]**2
        for i in range(0, n_sigma):
            sigma_b['split'][i] = m_min[i] * ( g_d - g_u ) - g_u
            sigma_r['split'][i] = m_min[i] * ( g_d - g_u ) + g_u
            sigma_b['S'][i] = 0.5*(j_min+m_min[i]+1)*(j_min+m_min[i]+2)
            sigma_r['S'][i] = 0.5*(j_min-m_min[i]+1)*(j_min-m_min[i]+2)
        
    elif round(j_u - j_d) == -1:
        #print 'JMIN is UP'

        for i in range(0, n_pi):
            pi['split'][i] = m_min[i] * ( g_d - g_u )
            pi['split'][i] = (j_min+1)**2 - m_min[i]**2
        for i in range(0, n_sigma):
            sigma_b['split'][i] = m_min[i] * ( g_d - g_u ) - g_d
            sigma_r['split'][i] = m_min[i] * ( g_d - g_u ) + g_d
            sigma_b['S'][i] = 0.5*(j_min-m_min[i]+1)*(j_min-m_min[i]+2)
            sigma_r['S'][i] = 0.5*(j_min+m_min[i]+1)*(j_min+m_min[i]+2)

        
    else:
        print('computation of S not supposed to failed….')
    
    pi['S'] = pi['S']/np.sum(pi['S'])
    sigma_b['S'] = sigma_b['S'] / np.sum(sigma_b['S'])
    sigma_r['S'] = sigma_r['S'] / np.sum(sigma_r['S'])
    
    #print '****'
    return { 'pi':pi, 'sigma_r':sigma_r, 'sigma_b':sigma_b, 'n_pi':pi.size, 'n_sigma':sigma_r.size, 'g_d':g_d, 'g_u':g_u  }
    
    
    
################################
################################
################################

def plot_zeeman_pattern(pattern):

    fig, ax = plt.subplots(1,1)

    ymax = np.max(pattern['pi']['S'])
    ymin = np.min(pattern['sigma_b']['S'])
    yrange = np.max([ymax, ymin])

#    ax.set_ylim(-1*yrange, yrange)
    
    xpi = np.max(np.abs(pattern['sigma_b']['split']))
    xsig = np.max(np.abs(pattern['sigma_b']['split']))
    
    xmax = np.max([xpi, xsig])*1.1
    
    ax.set_xlim(-1*xmax, xmax)

    for item in pattern['pi']:
        ax.plot( [item['split']]*2, [0, item['S']], color='green' )
        
    for item in pattern['sigma_r']:
        ax.plot( [item['split']]*2, [0, -1*item['S']], color='red' )

    for item in pattern['sigma_b']:
        ax.plot( [item['split']]*2, [0, -1*item['S']], color='blue' )

    ax.plot([-1*xmax, xmax], [0]*2, ls='--', c='k')
    
    ax.set_xlabel(r'$\Delta \lambda / \Delta_{\lambda_B}$')
    ax.set_ylabel('Normalized S')
    
#    print '****'
    return
    
