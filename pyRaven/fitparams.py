import numpy as np
import scipy
import copy
import pyRaven as rav

def fun(theta_,*p):
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

