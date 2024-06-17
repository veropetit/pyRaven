import numpy as np
import scipy
import copy
import pyRaven as rav

def fun(theta_,*p):
    '''
    This function makes a copy of the param object and updates it based on the input parameters.
    Inputs:
    :theta_: a list of lists containing parameters to update the param object with. The structure is as follows: [[star1 parameters],[star2 parameters],...]
    :*p: is a set of additional input parameters, listed below.
        :fitparam: list of lists of strings indicating which parameters to fit for which stars 
        :param: list of param objects for each star to fit
        :DataPacket: the datapacket for the object
        :guess: a list of lists containing the original guess parameters in the same order and format as theta_

    Outputs:
    :star: a list of param objects containing the new parameters (in the same format as param)
    
    '''

    fitparam=p[0]
    param=p[1]
    DataPacket=p[2]
    guess=p[3]

    star=[]
    prev=0
    for j in range(len(fitparam)): #loop over stars
        k=0
        star.append(copy.deepcopy(param[j])) 
        for i,n in enumerate(fitparam[j]): #loop over parameters that are being fit (skipping vrad for now)
            if n != 'vrad':
                star[j]['general'][n]=theta_[prev+i+k]
            if n == 'vrad':
                k=DataPacket.nobs-1
                vrads=[]
                for o in range(DataPacket.nobs):
                    vrads.append(theta_[i+prev+o])
                star[j]['general']['vrad']=vrads
         
        prev+=len(np.hstack(guess[j]))

    return(star)

def param_to_model(parameters,DataPacket):
    '''
    This function models the multi-star LSD profile given the input list of param objects

    Inputs:
    :parameters: a list of param objects to model
    :DataPacket: the datapacket for the object

    Outputs:
    :fitmodels: list of stokes I values for the fit model per observation
    '''


    fitmodels=[]
    fys=[]
    for j in range(len(parameters)):
        model=rav.diskint.analytical(parameters[j],False)
        for o in range(DataPacket.nobs):
            fys.append(np.interp(DataPacket.scaled.lsds[o].vel,model['vel']+parameters[j]['general']['vrad'][o],model['flux']))
    
    for o in range(DataPacket.nobs):
        obs=np.zeros(len(fys[0]))
        for j in range(len(parameters)):
            obs+=fys[o+(DataPacket.nobs)*j]
        fitmodels.append(obs-j)
    return(fitmodels)

def chi2(theta_,*p):
    '''
    Helper function that calculates the chi2 used by scipy.optimize.minimize

    Inputs:
    :theta_: list of lists containing the parameters to test
    :*p: is a set of additional input parameters, listed below.
        :fitparam: list of lists of strings indicating which parameters to fit for which stars 
        :param: list of param objects for each star to fit
        :DataPacket: the datapacket for the object
        :guess: a list of lists containing the original guess parameters in the same order and format as theta_

    Outputs:
    :chi2: the chi2 value 

    '''


    ys=[]
    ys_err=[]
    DataPacket=p[2]
    for o in range(DataPacket.nobs):
        ys.append(DataPacket.scaled.lsds[o].specI)
        ys_err.append(DataPacket.scaled.lsds[o].specSigI)

    star=fun(theta_,*p)
    fitmodels=param_to_model(star,DataPacket)
    
    models=np.hstack(fitmodels) 
    chi2=(np.sum((models-np.hstack(ys))**2/np.hstack(ys_err)))
    return(chi2)

def fitting(param_to_fit,parameters,DataPacket,guess,bounds):
    '''
    The actual fitting routine

    Inputs:
    :param_to_fit: list of list of strings stating which parameters should be fit for each star
    :parameters: list of param objects for each star
    :DataPacket: datapacket for the object
    :guess: list of lists containing the initial guess values for each parameter to fit and each star
    :bounds: list of lists of tuples containing the bounds to use for each parameter to fit

    Outputs:
    :res: the output from scipy.optimize.minimize
    :star: list of param objects for each star containing the final fit values

    '''


    bound=[]
    guesses=np.array([])
    for j in range(len(param_to_fit)):
        guesses=np.append(guesses,np.hstack(guess[j]))
        for i,n in enumerate(param_to_fit[j]):
            if n!='vrad':
                bound.append(bounds[j][i])
            if n=='vrad':
                for o in range(DataPacket.nobs):
                    bound.append(bounds[j][np.where(np.array(param_to_fit[j])=='vrad')[0][0]][o])

    res=scipy.optimize.minimize(chi2,guesses,args=(param_to_fit,parameters,DataPacket,guess),bounds=bound)

    star=fun(res.x,param_to_fit,parameters,DataPacket,guess)
    return(res,star)

