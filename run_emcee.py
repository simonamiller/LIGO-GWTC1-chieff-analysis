# ---- Script to run emcee to find mu and sigma^2 ----
# ---- Simona Miller, created Nov. 2019, slight updates April 2020 ----


    
'''
Set-up 
'''
import numpy as np
import emcee as mc
import h5py
from scipy.stats import gaussian_kde
from scipy.special import erf
import sys

# -- Set prior bounds --
mu_min = -1.0
mu_max = 1.0
sigma2_min = 0
sigma2_max = 1.0

# -- Function to calculate the value of a Gaussian distribution normalized on [-1, 1] at a point 'x' with mean mu and variance sigma2 --
def calculate_Gaussian(x, mu, sigma2): 
    
    # Normalization coefficient:
    norm = np.sqrt(sigma2*np.pi/2)*(erf((1-mu)/np.sqrt(2*sigma2)) + erf((1+mu)/np.sqrt(2*sigma2))) 
    
    # Gaussian distribution! 
    y = (1.0/norm)*np.exp((-1.0*(x-mu)**2)/(2*sigma2)) 
    return y
  
    
    
'''
Load samples 
'''
# Dict with samples from individual events: 
sampleDict = np.load("./sampleDict.npy").item()

# Will Farr's "detected Chi data" -- used for selection effects
X_det = np.asarray(h5py.File('./selection-effects.h5', 'r')['chi_effs'])


    
'''
Function to calculate the log posterior for a given c=(mu, sigma2) sample
'''
def logposterior(c):

    # Read parameters
    mu = c[0]
    sigma2 = c[1]

    # Flat priors, reject samples past boundaries
    if mu<mu_min or mu>mu_max or sigma2<sigma2_min or sigma2>sigma2_max:
        return -np.inf
    
    # If sample in prior range, evaluate
    else:
        logP = 0.
                
        for event in sampleDict:

            # Grab samples
            X_samples = sampleDict[event]['Xeff']      # LALinf Xeff posterior samples
            X_prior = sampleDict[event]['Xeff_prior']  # LALinf Xeff prior
            weights = sampleDict[event]['weights']     # Weight of pASTRO to pLAL for {m1, m2, z}
            
            # Chi probability - Gaussian: P(chi_eff | mu, sigma2)
            p_Chi = calculate_Gaussian(X_samples, mu, sigma2)    
            
            # Evaluate marginalized likelihood
            nSamples = p_Chi.size
            pEvidence = np.sum(weights*(p_Chi/X_prior))/nSamples
            
            # Summation
            logP += np.log(pEvidence)
        
        
        # Then multiply by detection efficiency (logs add)
        nEvents = len(sampleDict)

        det_weights = calculate_Gaussian(X_det,mu,sigma2)
        Nsamp = np.sum(det_weights)/np.max(det_weights)
        if Nsamp<=4*nEvents:
            return -np.inf

        log_detEff_term = -nEvents*np.log(np.sum(det_weights))
        logP += log_detEff_term
         
        return logP
    
    

    
# -- Running mcmc --     
if __name__=="__main__":

    # Initialize walkers from random positions in mu-sigma2 parameter space
    nWalkers = 16
    initial_mus = np.random.random(nWalkers)*(mu_max-mu_min)+mu_min
    initial_sigma2s = np.random.random(nWalkers)*(sigma2_max-sigma2_min)+sigma2_min
    initial_walkers = np.transpose([initial_mus,initial_sigma2s])
    
    print('Initial walkers:')
    print(initial_walkers)
    
    # Dimension of parameter space
    dim = 2

    # Run
    nSteps = 10000
    sampler = mc.EnsembleSampler(nWalkers,dim,logposterior,threads=8)
    sampler.run_mcmc(initial_walkers,nSteps)
    np.save('./mu_sigma2_samples.npy',sampler.chain) # chain dimensions: # walkers, # steps, # variables

    # ** Post processing of sampler chain is in a different script ** 
