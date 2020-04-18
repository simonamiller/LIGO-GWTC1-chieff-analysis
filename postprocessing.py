# ---- Script to post-process mu and sigma^2 samples ----
# ---- Simona Miller, created Nov. 2019, slight updates April 2020 ----

import numpy as np
import acor


if __name__=="__main__":

    # Load sample chain
    fname = 'mu_sigma2_samples'
    sample_chain = np.load(fname+'.npy')
    
    print("Sample chain:")
    print sample_chain
    
    sample_chain_shape = np.shape(sample_chain)
    print("Shape of sample chain:")
    print sample_chain_shape
    
    # Dimension of parameter space, number of steps taken, and number of walkers used
    dim = sample_chain_shape[2]
    nSteps = sample_chain_shape[1]
    nWalkers = sample_chain_shape[0]
    
    # Burn first half of chain
    chainBurned = sample_chain[:,int(np.floor(nSteps/2.)):,:]
    
    print("Shape of burned chain:")
    print np.shape(chainBurned)

    # Get max correlation length (over all variables and walkers)
    corrTotal = np.zeros(dim)
    for i in range(dim):
        for j in range(nWalkers):
            (tau,mean,sigma) = acor.acor(chainBurned[j,:,i]) #acor = autocorrelation package
            corrTotal[i] += 2.*tau/(nWalkers)
    maxCorLength = np.max(corrTotal)
    
    print("Max correlation length:")
    print(maxCorLength)

    # Down-sample by twice the max correlation length
    chainDownsampled = chainBurned[:,::int(maxCorLength),:]

    print("Shape of downsampled chain:")
    print np.shape(chainDownsampled)

    # Flatten - we don't care about each individual walker anymore
    chainDownsampled = chainDownsampled.reshape((-1,len(chainDownsampled[0,0,:])))
    
    print("Shape of downsampled chain post-flattening:")
    print np.shape(chainDownsampled)
    
    # Save data 
    np.save(fname+'_postprocessed.npy', chainDownsampled)

