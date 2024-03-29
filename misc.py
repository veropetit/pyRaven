import mpl_toolkits
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.colors as cl
from scipy.special import erf
import numpy as np
import sys


def cm_plusmin():
    cols = []
    for x in np.linspace(0,1, 256):
        rcol = 0.237 - 2.13*x + 26.92*x**2 - 65.5*x**3 + 63.5*x**4 - 22.36*x**5
        gcol = ((0.572 + 1.524*x - 1.811*x**2)/(1 - 0.291*x + 0.1574*x**2))**2
        bcol = 1/(1.579 - 4.03*x + 12.92*x**2 - 31.4*x**3 + 48.6*x**4 - 23.36*x**5)
        cols.append((rcol, gcol, bcol))

    cm_plusmin = cl.LinearSegmentedColormap.from_list("PaulT_plusmin", cols)
    return(cm_plusmin)
    
    
def cm_linear():
    cols = []
    for x in np.linspace(1,0, 256):
        rcol = (1 - 0.392*(1 + erf((x - 0.869)/ 0.255)))
        gcol = (1.021 - 0.456*(1 + erf((x - 0.527)/ 0.376)))
        bcol = (1 - 0.493*(1 + erf((x - 0.272)/ 0.309)))
        cols.append((rcol, gcol, bcol))

    cm_linear = cl.LinearSegmentedColormap.from_list("PaulT_linear", cols)
    return(cm_linear)
    
def cm_rainbow():
    cols = []
    for x in np.linspace(0,1, 254):
        rcol = (0.472-0.567*x+4.05*x**2)/(1.+8.72*x-19.17*x**2+14.1*x**3)
        gcol = 0.108932-1.22635*x+27.284*x**2-98.577*x**3+163.3*x**4-131.395*x**5+40.634*x**6
        bcol = 1./(1.97+3.54*x-68.5*x**2+243*x**3-297*x**4+125*x**5)
        cols.append((rcol, gcol, bcol))

    #cols.append((1,1,1))
    cm_rainbow = cl.LinearSegmentedColormap.from_list("PaulT_rainbow", cols)
    return(cm_rainbow)
    
def viz( ROT, c, cmap=None, title='' ):
    fig = plt.figure(figsize=(10,10) )
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=90, azim=90)
    if cmap == None:
        cmap = cm_plusmin()

    sc = ax.scatter(ROT[0,:], ROT[1,:], ROT[2,:], s=30, c=c, cmap=cmap)
    cbar = plt.colorbar(sc)


    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.tight_layout()
    
    return(fig,ax)

def check_req(param,func):
      genreq=['lambda0','vsini','vdop','av','bnu','logkappa','ndop','Bpole','incl','beta','phase']
      unnoreq=['down','up']
      weakreq=['geff']
      gridreq=['Bpole_grid','incl_grid','beta_grid','phase_grid']
      loopreq=['lambda0','vsini','vdop','av','bnu','logkappa','ndop']
      if func=='unno':
            try:
                  param['general']
            except:
                  print('Missing `general` subdictionary')
            try:
                  param['unno']
            except:
                  print('Missing `unno` subdictionary')
            for i in range(len(genreq)):
                  if genreq[i] not in list(param['general'].keys()):
                        print('Missing {}'.format(genreq[i]))
                        sys.exit()
            for i in range(len(unnoreq)):
                  if unnoreq[i] not in list(param['unno'].keys()):
                        print('Missing {}'.format(unnoreq[i]))
                        sys.exit()

      if func=='weak':
            try:
                  param['general']
            except:
                  print('Missing `general` subdictionary')
            try:
                  param['weak']
            except:
                  print('Missing `weak` subdictionary')
            for i in range(len(genreq)):
                  if genreq[i] not in list(param['general'].keys()):
                        print('Missing {}'.format(genreq[i]))
                        sys.exit()
            for i in range(len(weakreq)):
                  if weakreq[i] not in list(param['weak'].keys()):
                        print('Missing {}'.format(weakreq[i]))
                        sys.exit()


      if func=='analytic':
            try:
                  param['general']
            except:
                  print('Missing `general` subdictionary')
            for i in range(len(loopreq)):
                  if loopreq[i] not in list(param['general'].keys()):
                        print('Missing {}'.format(loopreq[i]))
                        sys.exit()


      if func=='loop':
            try:
                  param['general']
            except:
                  print('Missing `general` subdictionary')
            try:
                  param['weak']
            except:
                  print('Missing `weak` subdictionary')
            try:
                  param['grid']
            except:
                  print('Missing `grid` subdictionary')
            for i in range(len(loopreq)):
                  if loopreq[i] not in list(param['general'].keys()):
                        print('Missing {}'.format(loopreq[i]))
                        sys.exit()
            for i in range(len(weakreq)):
                  if weakreq[i] not in list(param['weak'].keys()):
                        print('Missing {}'.format(weakreq[i]))
                        sys.exit()
            for i in range(len(gridreq)):
                  if gridreq[i] not in list(param['grid'].keys()):
                        print('Missing {}'.format(gridreq[i]))
                        sys.exit()