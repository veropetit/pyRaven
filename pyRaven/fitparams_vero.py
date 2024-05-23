import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
from scipy.stats import norm
import scipy
from statistics import mode
import copy

from . import diskint as rav_diskint

def fitdata(param,DataPacket,guess):
  '''
  This function fits a set of LSD profiles using scipy's curve fit function.

  Inputs:
    param - input parameter dictionary
    DataPacket - input DataPacket with real data
    guess - array of guess values for logkappa, vsini, and vmac. Ex: np.array([1.3,250,30])

   Outputs:
    parameters - array of fit parameters
    covariance - covariance matrix of the fit
    modelout - the best fit model

  '''
  ##################################
  def model(v,logkappa,vsini,vmac):
      '''
      This function creates the line profile model that will be fit to the observed profile

      Inputs:
        logkappa - value of logkappa that the walker is on in parameter space
        vsini - value of vsini that the walker is on in parameter space
        vmac - value of the vmac that the walker is on in parameter space
        v - velocity array of the actual data

      Outputs:
        f - line profile model using the weak field method at the walker's position in parameter space
      '''
      # using the param that is passed to the previous funciton as a global
      # making a deepcopy here as to not modify the original that was passed
      # to the function. 
      param_copy = copy.deepcopy(param)
      # updating the param_copy to the values passed by the curve_fit
      param_copy['general']['vsini']=vsini
      param_copy['general']['vmac']=vmac
      param_copy['general']['logkappa']=logkappa

      #pyRaven's weakfield disk integration function
      model=rav_diskint.analytical(param_copy,False)

      #interpolating the model to be size MCMC wants
      f=np.interp(v,model['vel'],model['flux'])
      return(f)
  ##################################
    
  x=DataPacket.scaled.lsds[0].vel#+DataPacket.vrad[0]
  y=DataPacket.scaled.lsds[0].specI
  sigma = DataPacket.scaled.lsds[0].specSigI
  if DataPacket.nobs!=1:
    for i in range(1,DataPacket.nobs):
      x=np.append(x,DataPacket.scaled.lsds[i].vel)#+DataPacket.vrad[i])
      y=np.append(y,DataPacket.scaled.lsds[i].specI)
      sigma=np.append(sigma,DataPacket.scaled.lsds[i].specSigI)

  parameters,covariance = scipy.optimize.curve_fit(model,x,y,p0=guess,sigma=sigma)

  modelout=model(x,parameters[0],parameters[1],parameters[2])
  modelout=modelout[:DataPacket.scaled.lsds[0].vel.size]
  param_best = copy.deepcopy(param)
  param_best['general']['logkappa'] = parameters[0]
  param_best['general']['vsini'] = parameters[1]
  param_best['general']['vmac'] = parameters[2]
  return param_best, covariance, modelout


def fitdata_novsini(param,DataPacket,guess):
  '''
  This function fits a set of LSD profiles using scipy's curve fit function.

  Inputs:
    param - input parameter dictionary
    DataPacket - input DataPacket with real data
    guess - array of guess values for logkappa and vmac. Ex: np.array([1.3,30])

   Outputs:
    parameters - array of fit parameters
    covariance - covariance matrix of the fit
    modelout - the best fit model

  '''
  def model(v,logkappa,vmac):
      '''
      This function creates the line profile model that will be fit to the observed profile

      Inputs:
        logkappa - value of logkappa that the walker is on in parameter space
        vmac - value of vmac that the walker is on in parameter space
        v - velocity array of the actual data

      Outputs:
        f - line profile model using the weak field method at the walker's position in parameter space
      '''
      # using the param that is passed to the previous funciton as a global
      # making a deepcopy here as to not modify the original that was passed
      # to the function. 
      param_copy = copy.deepcopy(param)
      param_copy['general']['vmac'] = vmac
      param_copy['general']['logkappa'] = logkappa

      #pyRaven's weakfield disk integration function
      model=rav_diskint.analytical(param_copy,False)

      #interpolating the model to be size MCMC wants
      f=np.interp(v,model['vel'],model['flux'])
      return(f)
    
  x=DataPacket.scaled.lsds[0].vel#+DataPacket.vrad[0]
  y=DataPacket.scaled.lsds[0].specI
  sigma = DataPacket.scaled.lsds[0].specSigI

  if DataPacket.nobs!=1:
    for i in range(1,DataPacket.nobs):
      x=np.append(x,DataPacket.scaled.lsds[i].vel)
      y=np.append(y,DataPacket.scaled.lsds[i].specI)
      sigma=np.append(sigma,DataPacket.scaled.lsds[i].specSigI)


  parameters,covariance = scipy.optimize.curve_fit(model,x,y,p0=guess,sigma=sigma)

  modelout=model(x,parameters[0],parameters[1])
  modelout=modelout[:DataPacket.scaled.lsds[0].vel.size]
  param_best = copy.deepcopy(param)
  param_best['general']['logkappa'] = parameters[0]
  param_best['general']['vmac'] = parameters[1]
  return param_best,covariance,modelout



def binary_fitting(xs,ys,guess, param1, param2):
  '''
  This function creates the line profile model for a binary system that will be fit to the observed profile

  Inputs:
  xs - list of velocity arrays of each observation of the observed profile
  ys - list of specI arrays of each observation of the observed profile
  guess - array with guess parameters. Format: np.array([kappa1,kappa2,vsini1,vsini2,vmac1,vmac2,vrad1_1,vrad2_1,vrad1_2,vrad2_2,...])
  param1 - initial lsd parameters for star 1
  param2 - initial lsd parameters for star 2

  Outputs:
  pout - array with the best fit parameters in the same format at guess
  pcov - corresponding covariance matrix
  binary_models - array of specI models of the binary at each observation
  star1_models - array of specI models of star 1 at each observation
  star2_models - array of specI models of star 2 at each observation
  '''
  
  x_data = np.hstack(xs)
  y_data = np.hstack(ys)

  # star 1 model
  def star1(v,kappa,vsini,vmac,vrad):
    '''
    This function creates the line profile model for star 1 that will be fit to the observed profile

    Inputs:
    v - velocity array of the actual data
    kappa - value of kappa that the walker is on in parameter space
    vsini - value of vsini that the walker is on in parameter space
    vmac - value of the macroturbulence velocity that the walker is on in parameter space
    vrad - value of the radial velocity that the walker is on in parameter space


    Outputs:
    f - line profile model using the weak field method at the walker's position in parameter space
    '''
    param1['general']['vsini']=vsini
    param1['general']['vmac']=vmac
    param1['general']['logkappa']=np.log(kappa)

    #pyRaven's weakfield disk integration function
    model=rav_diskint.analytical(param1,False)

    #interpolating the model to be size MCMC wants
    f=np.interp(v,model['vel']+vrad,model['flux'])
    return(f)

  #star 2 model
  def star2(v,kappa,vsini,vmac,vrad):
    '''
    This function creates the line profile model for star 2 that will be fit to the observed profile

    Inputs:
    v - velocity array of the actual data
    kappa - value of kappa that the walker is on in parameter space
    vsini - value of vsini that the walker is on in parameter space
    vmac - value of the macroturbulence velocity that the walker is on in parameter space
    vrad - value of the radial velocity that the walker is on in parameter space


    Outputs:
    f - line profile model using the weak field method at the walker's position in parameter space
    '''
    param2['general']['vsini']=vsini
    param2['general']['vmac']=vmac
    param2['general']['logkappa']=np.log(kappa)

    #pyRaven's weakfield disk integration function
    model=rav_diskint.analytical(param2,False)

    #interpolating the model to be size MCMC wants
    f=np.interp(v,model['vel']+vrad,model['flux'])
    return(f)



  # combines the star 1 and star 2 models into 1 model
  def binary(v,kappa1,kappa2,vsini1,vsini2,vmac1,vmac2,vrad1,vrad2):
    '''
    This function creates the line profile model that will be fit to the observed profile

    Inputs:
    v - velocity array of the actual data
    kappa1 - value of kappa for star 1 that the walker is on in parameter space
    vsini1 - value of vsini for star 1 that the walker is on in parameter space
    vmac1 - value of the macroturbulence velocity for star 1 that the walker is on in parameter space
    vrad1 - value of the radial velocity for the current observation of star 1 that the walker is on in parameter space
    kappa2 - value of kappa for star 2 that the walker is on in parameter space
    vsini2 - value of vsini for star 2 that the walker is on in parameter space
    vmac2 - value of the macroturbulence velocity for star 2 that the walker is on in parameter space
    vrad2 - value of the radial velocity for the current observation of star 2 that the walker is on in parameter space


    Outputs:
    binary - the combined binary star model
    '''
    f1=star1(v,kappa1,vsini1,vmac1,vrad1)
    f2=star2(v,kappa2,vsini2,vmac2,vrad2)
    binary=(f1+f2)-1
    return(binary)

  # the model used by curvefit
  def poly(x_, *p):
    kappa1=p[0]
    kappa2=p[1]
    vsini1=p[2]
    vsini2=p[3]
    vmac1=p[4]
    vmac2=p[5]

    lens=[0]
    for i in range(len(xs)):
        lens.append(lens[i]+len(xs[i]))
    
    models=[]
    for i in range(len(lens)-1):
        models.append(binary(x_[lens[i]:lens[i+1]],kappa1,kappa2,vsini1,vsini2,vmac1,vmac2,p[6+2*i],p[7+2*i]))

    model=np.hstack(models)  
    return model
  

  guess=guess
  
  # bounds for each parameter. kappa between 0 and infinity, vsini from 0 to 300 km/s, vmac from 0 to 40 vmac, vrads from -infinity to infinity
  bounds=([0,0,0,0,0,0],[np.inf,np.inf,300,300,40,40])
  for i in range(len(xs)):
      bounds[0].append(-np.inf)
      bounds[0].append(-np.inf)
      bounds[1].append(np.inf)
      bounds[1].append(np.inf)

  pout, pcov = scipy.optimize.curve_fit(poly,x_data,y_data,guess,bounds=bounds)
  
  binary_models=[]
  star1_models=[]
  star2_models=[]
  for n in range(len(xs)):
    binary_models.append([binary(xs[n],pout[0],pout[1],pout[2],pout[3],pout[4],pout[5],pout[6+2*n],pout[7+2*n])])
    star1_models.append([star1(xs[n],pout[0],pout[2],pout[4],pout[6+2*n])])
    star2_models.append([star2(xs[n],pout[1],pout[3],pout[5],pout[7+2*n])])

  return pout,pcov,binary_models,star1_models,star2_models