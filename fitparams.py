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
    
  x=DataPacket.original.lsds[0].vel-DataPacket.vrad[0]
  y=DataPacket.original.lsds[0].specI
  if DataPacket.nobs!=1:
    for i in range(1,DataPacket.nobs):
      x=np.append(x,DataPacket.original.lsds[i].vel-DataPacket.vrad[i])
      y=np.append(y,DataPacket.original.lsds[i].specI)

  parameters,covariance = scipy.optimize.curve_fit(model,x,y,guess)

  modelout=model(x,parameters[0],parameters[1],parameters[2])
  modelout=modelout[:DataPacket.original.lsds[0].vel.size]
  return parameters,covariance,modelout



def fitdataMCMC(param,DataPacket,nsteps,guess):
  '''
  This function fits the LSD profile using MCMC

  Inputs:
    param - input parameter dictionary
    DataPacket - input DataPacket with real data
    nsteps - number of steps to run MCMC
    guess - array of guess values for kappa, vsini, and vmac. Ex: np.array([1.3,250,30])

  Outputs:
    kappa - The average fitted value of kappa
    vsini - The average fitted value of vsini
    vmac - The average fitted value of vmac
  '''
  #def model(v,c,a,b,Ic):
  #  return(-c*np.exp(-0.5*np.power(v-a,2)/b**2)+Ic)

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

  def lnprior(params):
    '''
    This function is used to set constraints on the parameter space the walkers are allowed in. I did this to try and save time, could probably use some tweaking.

    Inputs:
      params - list of walker parameters, in this code that is [kappa, vsini]

    Outputs:
      -infinity - if kappa and/or vsini are out of the specified ranges
      0 - otherwise
    '''
    kappa,vsini,vmac=params
    if kappa<=0.0 or kappa>=10.0 or vsini >= 500.0 or vsini<=0.0 or vmac<=0.0 or vmac>=100.0:
      return(-np.inf)
    else:
      return(0.0)

  def lnlike(params,v,I,Ierr):
    '''
    Inputs:
      params -list of walker parameters, in this code that is [kappa, vsini]
      v - velocity array of the data
      I - stokes I of the actual data
      Ierr - uncertainty in the stokes I of the actual data

    Outputs:
      The log likelihood using a gaussian function
    '''
    kappa,vsini,vmac= params
    m = model(v,kappa,vsini,vmac)
    sigma2 = Ierr**2 #+m**2*np.exp(2*log_f)
    return(-0.5 * np.sum((I - m) ** 2 / sigma2 + np.log(sigma2)))

  def lnprob(params,v,I,Ierr):
    '''
    Inputs:
      params - list of walker parameters, in this code that is [kappa, vsini]
      v - velocity array of the data
      I - stokes I of the actual data
      Ierr - uncertainty in the stokes I of the actual data

    Outputs:
      log probability. Used to determine how good a fit the model is
    '''
    prior=lnprior(params)
    if not np.isfinite(prior):
      return(-np.inf)
    else:
      return(prior+lnlike(params,v,I,Ierr))

  # Set up the convergence diagonstic plots. At the final step we want all the walkers to be very close together, i.e a straight line at the end.
  fig, ax = plt.subplots(3,1,figsize=(15,5))
  ax[0].set_title('Convergence Diagnostic Plots')



  fig1, ax1 = plt.subplots(1,1,figsize=(5,5)) #sets up the send set of plots

  #for i in range(1):
  kappa=np.array([])
  #log_f=np.array([])
  vsini=np.array([])
  vmac=np.array([])
  Ic=1.0 #defining the continuum value of the real data
  vsiniin=DataPacket.vsini #defining the vsini listed in the data packet


  v=DataPacket.original.lsds[0].vel-DataPacket.vrad[0] #defining the velocity array of the data
  I=DataPacket.original.lsds[0].specI #defining the stokes I array of the data
  Ierr=DataPacket.original.lsds[0].specSigI #defining the stokes I error of the data
  for i in range(1,DataPacket.nobs):
    v=np.append(v,DataPacket.original.lsds[i].vel)-DataPacket.vrad[i] #defining the velocity array of the data
    I=np.append(I,DataPacket.original.lsds[i].specI) #defining the stokes I array of the data
    Ierr=np.append(Ierr,DataPacket.original.lsds[i].specSigI) #defining the stokes I error of the data

  ndim = 3 #number of parameters to fit
  nwalkers= 10 * ndim #number of walkers (10/parameter)
  pguess = guess #initial guess for kappa and vsini
  positions = np.zeros((nwalkers,ndim)) #set up walker position array
  positions[:,0] = np.abs(np.random.randn(nwalkers)*pguess[0]*0.1+pguess[0]) #set the inital positions of the kappa walkers to be a random distribution around the guess
  positions[:,1] = np.random.randn(nwalkers)*pguess[1]*0.1+pguess[1] #set the initial positions of the vsini walkers
  positions[:,2] = np.random.randn(nwalkers)*pguess[2]*2

  sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob,args=(v,I,Ierr)) #set up MCMC. Note that the args keyword contains the real data arrays

  pos,prob,state = sampler.run_mcmc(positions, nsteps,progress=True) #runs MCMC for the specified number of steps

  #make the first set of plots
  res = [ax[j].plot(sampler.chain[:,:,j].T, '-', color='k', alpha=0.3) for j in range(3)]
  res = [ax[j].axhline(pguess[j]) for j in range(3)]

  #save the walker positions at each step (for diagnostics)
  #kappa=np.append(kappa,np.mean(sampler.flatchain[int(2*nsteps/3):], axis=0)[0])
  #vsini=np.append(vsini,np.mean(sampler.flatchain[int(2*nsteps/3):], axis=0)[1])
  #vmac=np.append(vmac,np.mean(sampler.flatchain[int(2*nsteps/3):], axis=0)[2])

  kappa=sampler.flatchain[int(2*100/3):][:,0]
  vsini=sampler.flatchain[int(2*100/3):][:,1]
  vmac=sampler.flatchain[int(2*100/3):][:,2]
  
  bins=20
  bin_means = (np.histogram(kappa, bins, weights=kappa)[0]/np.histogram(kappa, bins)[0])
  kappa=bin_means[np.histogram(kappa, bins)[0]==np.histogram(kappa, bins)[0].max()][0]

  bin_means = (np.histogram(vsini, bins, weights=vsini)[0]/np.histogram(vsini, bins)[0])
  vsini=bin_means[np.histogram(vsini, bins)[0]==np.histogram(vsini, bins)[0].max()][0]

  bin_means = (np.histogram(vmac, bins, weights=vmac)[0]/np.histogram(vmac, bins)[0])
  vmac=bin_means[np.histogram(vmac, bins)[0]==np.histogram(vmac, bins)[0].max()][0]

  #log_f=np.append(log_f,np.mean(sampler.flatchain, axis=0)[1])

  #make the second set of plots
  
  ax1.plot(DataPacket.original.lsds[0].vel-DataPacket.vrad[0],model(DataPacket.original.lsds[0].vel-DataPacket.vrad[0], kappa,vsini,vmac))
  ax1.plot(v,I)

  print('kappa: {} | vsini: {} | vmac: {}'.format(kappa,vsini,vmac))


  #make the corner plots
  flat_samples = sampler.get_chain(discard=0, thin=5, flat=True)
  labels = ["kappa","vsini",'vmac']
  corner.corner(
      flat_samples, labels=labels)

  return(kappa,vsini,vmac,sampler.flatchain)


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

  x=DataPacket.original.lsds[0].vel-DataPacket.vrad[0]
  y=DataPacket.original.lsds[0].specI
  if DataPacket.nobs!=1:
    for i in range(1,DataPacket.nobs):
      x=np.append(x,DataPacket.original.lsds[i].vel-DataPacket.vrad[i])
      y=np.append(y,DataPacket.original.lsds[i].specI)

  parameters,covariance = scipy.optimize.curve_fit(model,x,y,guess)

  modelout=model(x,parameters[0],parameters[1])
  modelout=modelout[:DataPacket.original.lsds[0].vel.size]
  return parameters,covariance,modelout