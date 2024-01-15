import numpy as np
import matplotlib.pyplot as plt
import pyRaven as rav
import emcee
import corner
from scipy.stats import norm
import scipy
from statistics import mode


def fitdata(param,DataPacket,guess):
  '''
  This function fits a set of LSD profiles using scipy's curve fit function.

  Inputs:
    param - input parameter dictionary
    DataPacket - input DataPacket with real data
    guess - array of guess values for kappa, vsini, and vmac. Ex: np.array([1.3,250,30])

   Outputs:
    parameters - array of fit parameters
    covariance - covariance matrix of the fit
    modelout - the best fit model

  '''
  def model(v,kappa,vsini,vmac):
      '''
      This function creates the line profile model that will be fit to the observed profile

      Inputs:
        kappa - value of kappa that the walker is on in parameter space
        vsini - value of vsini that the walker is on in parameter space
        vmac - value of the vmac that the walker is on in parameter space
        v - velocity array of the actual data

      Outputs:
        f - line profile model using the weak field method at the walker's position in parameter space
      '''
      param['general']['vsini']=vsini
      param['general']['vmac']=vmac
      param['general']['logkappa']=np.log(kappa)

      #pyRaven's weakfield disk integration function
      model=rav.diskint2.analytical(param,False)

      #interpolating the model to be size MCMC wants
      f=np.interp(v,model['vel'],model['flux'])
      return(f)
    
  x=DataPacket.scaled.lsds[0].vel#+DataPacket.vrad[0]
  y=DataPacket.scaled.lsds[0].specI
  if DataPacket.nobs!=1:
    for i in range(1,DataPacket.nobs):
      x=np.append(x,DataPacket.scaled.lsds[i].vel)#+DataPacket.vrad[i])
      y=np.append(y,DataPacket.scaled.lsds[i].specI)

  parameters,covariance = scipy.optimize.curve_fit(model,x,y,guess)

  modelout=model(x,parameters[0],parameters[1],parameters[2])
  modelout=modelout[:DataPacket.scaled.lsds[0].vel.size]
  return parameters,covariance,modelout


def fitdata_novsini(param,DataPacket,guess):
  '''
  This function fits a set of LSD profiles using scipy's curve fit function.

  Inputs:
    param - input parameter dictionary
    DataPacket - input DataPacket with real data
    guess - array of guess values for kappa and vmac. Ex: np.array([1.3,30])

   Outputs:
    parameters - array of fit parameters
    covariance - covariance matrix of the fit
    modelout - the best fit model

  '''
  def model(v,kappa,vmac):
      '''
      This function creates the line profile model that will be fit to the observed profile

      Inputs:
        kappa - value of kappa that the walker is on in parameter space
        vmac - value of vmac that the walker is on in parameter space
        v - velocity array of the actual data

      Outputs:
        f - line profile model using the weak field method at the walker's position in parameter space
      '''
      param['general']['vmac']=vmac
      param['general']['logkappa']=np.log(kappa)

      #pyRaven's weakfield disk integration function
      model=rav.diskint2.analytical(param,False)

      #interpolating the model to be size MCMC wants
      f=np.interp(v,model['vel'],model['flux'])
      return(f)
    
  param['general']['vsini']=DataPacket.vsini

  x=DataPacket.scaled.lsds[0].vel#+DataPacket.vrad[0]
  y=DataPacket.scaled.lsds[0].specI
  if DataPacket.nobs!=1:
    for i in range(1,DataPacket.nobs):
      x=np.append(x,DataPacket.scaled.lsds[i].vel)#+DataPacket.vrad[i])
      y=np.append(y,DataPacket.scaled.lsds[i].specI)

  parameters,covariance = scipy.optimize.curve_fit(model,x,y,guess)

  modelout=model(x,parameters[0],parameters[1])
  modelout=modelout[:DataPacket.scaled.lsds[0].vel.size]
  return parameters,covariance,modelout