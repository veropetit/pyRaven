import numpy as np
import scipy
import copy
import pyRaven as rav

def update_params(theta_,param_to_fit,parameters,nobs,guess):
    '''
    This function makes a copy of the param object and updates it based on the input parameters.

    :param theta_: list of parameter values to update the param objects with. 
        The format is [[guess for star 1 parameter to fit,guess for star 1 parameter to fit,...],[guess for star 2 parameter to fit,...]].
        Note - MUST BE IN THE SAME ORDER AS param_to_fit. 
        Note - If vrad is specified in param_to_fit, the corresponding guess input must be a list with one value per observation, i.e. [vrad1,vrad2,...]
    :param param_to_fit: list of strings stating which parameters should be fit for each star.
        format - [[star 1 parameter to fit,star 1 parameter to fit,...],[star 2 parameter to fit,...],...].
        Note - Order of parameters in arrays doesn't matter, you can fit different parameters for each star. 
        Note - Any parameter not stated here will not be fit and will be taken to be its value in the param object. 
    :param parameters: list of param objects for each star.
        format - [param object 1,param object 2,...].
    :param nobs: number of observations.
    :param guess: list containing the initial guess values for each parameter to fit and each star. 
        format - [[guess for star 1 parameter to fit,guess for star 1 parameter to fit,...],[guess for star 2 parameter to fit,...]].
        Note - MUST BE IN THE SAME ORDER AS param_to_fit. 
        Note - If vrad is specified in param_to_fit, the corresponding guess input must be a list with one value per observation, i.e. [vrad1,vrad2,...].
    :return: a list of param objects containing the new parameters (in the same format as param)
    
    '''

    star=[]
    prev=0
    for j in range(len(param_to_fit)): #loop over stars
        k=0
        star.append(copy.deepcopy(parameters[j])) 
        for i,n in enumerate(param_to_fit[j]): #loop over parameters that are being fit
            if n != 'vrad':
                star[j]['general'][n]=theta_[prev+i+k] #this updates the param value for non-vrad entries
            if n == 'vrad':
                k=nobs-1
                vrads=[]
                for o in range(nobs):
                    vrads.append(theta_[i+prev+o]) #this appends the fit vrad entries to the vrad array
                star[j]['general']['vrad']=vrads #this updates the param value for vrad
         
        prev+=len(np.hstack(guess[j]))

    return(star)

def param_to_model(parameters,lsds):
    '''
    This function models the multi-star LSD profile given the input list of param objects

    :param parameters: list of param objects for each star.
        format - [param object 1,param object 2,...].
    :param lsds: a list containing the arrays of velocity,specI,and specSigI values for the lsd profiles to fit.
        format - [[velocity array for ob1, '' for ob2, ...],[specI array for ob1, '' for ob2, ...],[specSigI array for ob1, '' for ob2, ...]]
    :return: list of stokes I values for the fit model per observation
    '''
    vel=lsds[0]
    specI=lsds[1]
    specSigI=lsds[2]
    nobs=int(len(lsds[0]))

    #this creates a model for each star
    fitmodels=[]
    fys=[]
    for j in range(len(parameters)):
        model=rav.diskint.analytical(parameters[j],False) #creates the model lsd profile
        for o in range(nobs):
            fys.append(np.interp(vel[o],model['vel']+parameters[j]['general']['vrad'][o],model['flux'])) #for each observation, this interpolates the model to be the same size as the input lsd arrays
    
    #this combines the models in the case of spectroscopic binaries to get a single LSD profile
    for o in range(nobs): #loop over observations
        obs=np.zeros(len(fys[o]))
        for j in range(len(parameters)): #loop of number of stars being fit
            obs+=fys[o+(nobs)*j]
        fitmodels.append(obs-j)
    return(fitmodels)

def chi2(theta_,*p):
    '''
    Helper function that calculates the chi2 used by scipy.optimize.minimize

    :param theta_: list of parameter values to update the param objects with.
        format - [[guess for star 1 parameter to fit,guess for star 1 parameter to fit,...],[guess for star 2 parameter to fit,...]].
        Note - MUST BE IN THE SAME ORDER AS param_to_fit. 
        Note - If vrad is specified in param_to_fit, the corresponding guess input must be a list with one value per observation, i.e. [vrad1,vrad2,...]
    :param param_to_fit: list of strings stating which parameters should be fit for each star.
        format - [[star 1 parameter to fit,star 1 parameter to fit,...],[star 2 parameter to fit,...],...].
        Note - Order of parameters in arrays doesn't matter, you can fit different parameters for each star. 
        Note - Any parameter not stated here will not be fit and will be taken to be its value in the param object. 
    :param parameters: list of param objects for each star.
        format - [param object 1,param object 2,...].
    :param lsds: a list containing the arrays of velocity,specI,and specSigI values for the lsd profiles to fit.
        format - [[velocity array for ob1, '' for ob2, ...],[specI array for ob1, '' for ob2, ...],[specSigI array for ob1, '' for ob2, ...]].
    :param guess: list containing the initial guess values for each parameter to fit and each star. 
        format - [[guess for star 1 parameter to fit,guess for star 1 parameter to fit,...],[guess for star 2 parameter to fit,...]].
        Note - MUST BE IN THE SAME ORDER AS param_to_fit. 
        Note - If vrad is specified in param_to_fit, the corresponding guess input must be a list with one value per observation, i.e. [vrad1,vrad2,...].
    :return: the chi2 value 

    '''

    param_to_fit=p[0]
    parameters=p[1]
    lsds=p[2]
    guess=p[3]
    nobs=int(len(lsds[0]))

    ys=[]
    ys_err=[]

    for o in range(nobs):
        ys.append(lsds[1][o])
        ys_err.append(lsds[2][o])

    star=update_params(theta_,param_to_fit,parameters,nobs,guess)
    fitmodels=param_to_model(star,lsds)
    
    models=np.hstack(fitmodels) 
    chi2=(np.sum((models-np.hstack(ys))**2/np.hstack(ys_err)))
    return(chi2)

def fitting(param_to_fit,parameters,lsds,guess,bounds):
    '''
    The actual fitting routine

    :param param_to_fit: list of strings stating which parameters should be fit for each star.
        format - [[star 1 parameter to fit,star 1 parameter to fit,...],[star 2 parameter to fit,...],...].
        Note - Order of parameters in arrays doesn't matter, you can fit different parameters for each star. 
        Note - Any parameter not stated here will not be fit and will be taken to be its value in the param object. 
    :param parameters: list of param objects for each star.
        format - [param object 1,param object 2,...].
    :param lsds: a list containing the arrays of velocity,specI,and specSigI values for the lsd profiles to fit.
        format - [[velocity array for ob1, '' for ob2, ...],[specI array for ob1, '' for ob2, ...],[specSigI array for ob1, '' for ob2, ...]].
    :param guess: list containing the initial guess values for each parameter to fit and each star. 
        format - [[guess for star 1 parameter to fit,guess for star 1 parameter to fit,...],[guess for star 2 parameter to fit,...]].
        Note - MUST BE IN THE SAME ORDER AS param_to_fit. 
        Note - If vrad is specified in param_to_fit, the corresponding guess input must be a list with one value per observation, i.e. [vrad1,vrad2,...].
    :param bounds: list of tuples containing the bounds to use for each parameter to fit.
        format - [[(lower bound for star 1 guess,upper bound for star 1 guess),...],[(lower bound for star 2 guess,upper bound for star 2 guess)]].
        Note - MUST BE IN THE SAME ORDER AS param_to_fit.
        Note - If vrad is specified in param_to_fit, the corresponding bounds must be a list with one tuple per observation, i.e. [(lower 1,upper 1),(lower 2, upper 2),...].
    :return: the output from scipy.optimize.minimize and a list of param objects for each star containing the final fit values

    '''

    nobs=int(len(lsds[0]))
    bound=[]
    guesses=np.array([])
    for j in range(len(param_to_fit)):
        guesses=np.append(guesses,np.hstack(guess[j]))
        for i,n in enumerate(param_to_fit[j]):
            if n!='vrad':
                bound.append(bounds[j][i])
            if n=='vrad':
                for o in range(nobs):
                    bound.append(bounds[j][np.where(np.array(param_to_fit[j])=='vrad')[0][0]][o])

    res=scipy.optimize.minimize(chi2,guesses,args=(param_to_fit,parameters,lsds,guess),bounds=bound)

    star=update_params(res.x,param_to_fit,parameters,nobs,guess)
    return(res,star)

