# --------------------------------------------
#
#   Script to generate posterior on Xeff using population distribution as the prior
#
#   (1) Make KDEs for Xeff posterior for each GW detection, weighted by p_astro/p_LAL 
#   (3) Make KDEs for Xeff prior for each GW detection
#   (3) Load mu, sigma^2 samples
#   (4) Calculate posterior at for each event at each Xeff value, marginalizing over (mu, sigma^2) (including evidence term)
#   (5) Save results in a dict
#
#   Simona Miller, created Nov. 2019, slight updates April 2020 
#
# --------------------------------------------



'''
Set-up 
'''
import numpy as np
from scipy.stats import gaussian_kde
import h5py 
from scipy.special import erf
import astropy.cosmology as cosmo
import astropy.units as u
from astropy.cosmology import Planck15

# -- Names of BBH events -- 
BBH_names = ['GW151012', 'GW170608', 'GW170729', 'GW151226', 'GW170814', 'GW150914', 'GW170104', 'GW170809', 'GW170818', 'GW170823']

# -- Constants we'll need -- 
c = 3.0e8          # m/s
H_0 = 67270.0      # m/s/MPc
Omega_M = 0.3156 # unitless
Omega_Lambda = 1.0-Omega_M

# -- Reference distributions for luminosity distance and redshift -- 
ref_z = np.linspace(0.,3.,1000)
ref_dl = Planck15.luminosity_distance(ref_z).to(u.Mpc).value


'''
Functions used in script
'''

# -- To calculate Chi-effective given component spin magnitudes, cos(tilt angle), and masses -- 
def calculate_Xeff(a1, m1, cost1, a2, m2, cost2):
    Xeff = (a1*m1*cost1 + a2*m2*cost2)/(m1 + m2)
    return Xeff

# -- Function to calculate the value of a Gaussian distribution normalized on [-1, 1] at a point 'x' with mean mu and variance sigma2 --
def calculate_Gaussian(x, mu, sigma2): 
    # Normalization coefficient:
    norm = np.sqrt(sigma2*np.pi/2)*(erf((1-mu)/np.sqrt(2*sigma2)) + erf((1+mu)/np.sqrt(2*sigma2))) 
    # Gaussian distribution! 
    y = (1.0/norm)*np.exp((-1.0*(x-mu)**2)/(2*sigma2)) 
    return y

# -- To calculate the LALinference prior for masses and redshift -- 
def calculate_pLAL(z, dl):
    dc = dl/(1.+z) # comoving distance 
    pLAL = np.power(1.+z,2.)*np.power(dl,2.)*(dc+c*(1.+z)/Hz(z))
    return pLAL

# -- To calculate the astrophysically motivated prior for masses and redshift -- 
def calculate_pASTRO(z, dl, m1):
    dc = dl/(1.+z) # comoving distance 
    dVc_dz = 4*np.pi*c*(dc**2.)/Hz(z) # comoving volume 
    pASTRO = np.power(1.+z,1.7)*dVc_dz/(m1/(1.+z))/(m1/(1.+z)-5.)
    return pASTRO

# -- To calculate the factor H(z) that appears in the two above prior calculations -- 
def Hz(z):
    return H_0*np.sqrt(Omega_M*(1.+z)**3.+Omega_Lambda)




'''
Doing posterior reweighting -- following numbers in header of script
'''

# Grid to evaluate posteriors/priors on
dX = 0.001
Xeff_grid = np.arange(-1, 1+dX, dX)


# (1) 
# -- Load LALinf BBH data, calculate Xeff POSTERIOR samples, and then make KDE for posterior --

Xeff_post_KDEs = {}

for name in BBH_names:
    
    # load posterior samples
    BBH_post = h5py.File('GWTC-1_event_samples/'+name+'_GWTC-1.hdf5', 'r')['Overall_posterior']
    m1 = BBH_post['m1_detector_frame_Msun']
    m2 = BBH_post['m2_detector_frame_Msun']
    a1 = BBH_post['spin1']
    a2 = BBH_post['spin2']
    cost1 = BBH_post['costilt1']
    cost2 = BBH_post['costilt2']
    dl = BBH_post['luminosity_distance_Mpc']
    
    # calculate redshift
    z = np.interp(dl, ref_dl, ref_z)
    
    # calculate {m1, m2, z} astro and LAL priors for weights
    p_astro = calculate_pASTRO(z, dl, m1)
    p_lal = calculate_pLAL(z, dl)
    weights = p_astro/p_lal
    
    # checking for negative weights (would come from m2 < 5)
    weights[weights<0] = 0
    
    # calculate Xeff posterior
    Xeff_post = calculate_Xeff(a1, m1, cost1, a2, m2, cost2)
    
    # calculate KDE for Xeff posterior
    Xeff_post_KDEs[name] = gaussian_kde(Xeff_post, weights=weights)
    
    print('LALinf Xeff POSTERIOR KDE generated for {}.'.format(name))  
print('...')


# (2)
# -- Making KDE for the LALinf generated PRIOR on Xeff -- 

Xeff_prior_KDEs = {}

for name in BBH_names:
    
    # load prior samples
    BBH_prior = h5py.File('GWTC-1_event_samples/'+name+'_GWTC-1.hdf5', 'r')['prior']
    m1_prior = BBH_prior['m1_detector_frame_Msun']
    m2_prior = BBH_prior['m2_detector_frame_Msun']
    a1_prior = BBH_prior['spin1']
    a2_prior = BBH_prior['spin2']
    cost1_prior = BBH_prior['costilt1']
    cost2_prior = BBH_prior['costilt2']
    
    # calculate Xeff prior
    Xeff_prior = calculate_Xeff(a1_prior, m1_prior, cost1_prior, a2_prior, m2_prior, cost2_prior)
    
    # calculate KDE for Xeff prior
    Xeff_prior_KDEs[name] = gaussian_kde(Xeff_prior) 

    print('LALinf Xeff PRIOR KDE generated for {}.'.format(name))
print('...')


# (3)
# -- Load mu, sigma^2 samples -- 
mu_sigma2_samples = np.load('mu_sigma2_samples_postprocessed.npy')

print('mu sigma^2 samples loaded.')
print('...')




# (4)
# -- Cycle through each BBH event and caluculate new posterior

results_dict = {}

for name in BBH_names: 
    
    # Evaluate LAL posterior on grid
    LAL_post_KDE = Xeff_post_KDEs[name]
    LAL_post = LAL_post_KDE(Xeff_grid)
    
    # Evaluate LAL prior on grid
    LAL_prior_KDE = Xeff_prior_KDEs[name]
    LAL_prior = LAL_prior_KDE(Xeff_grid)
    
    new_post = np.zeros(len(Xeff_grid))
    
    # Cycle through all mu, sigma^2 samples (to marginalize over them)
    for sample in mu_sigma2_samples: 
        
        mu = sample[0]
        sigma2 = sample[1]
        
        # Evaluate population informed Gaussian prior
        Gaussian_prior = calculate_Gaussian(Xeff_grid, mu, sigma2)
        
        # calculate evidence term
        evidence = np.sum((LAL_post/LAL_prior)*Gaussian_prior)*dX
        
        # reweighting posterior
        P = (LAL_post*Gaussian_prior)/(evidence*LAL_prior)
        
        new_post += P
    
    # Normalize 
    norm = np.sum(new_post)*dX
    new_post_normed = (1.0/norm)*new_post
    
    # Save re-weighted posterior into dict
    results_dict[name] = new_post_normed
    
    print('New posterior calculated for {}.'.format(name))
    
print('...')
    
    
# (5)
# -- Save results 
np.save('population_informed_chieff_posteriors.npy', results_dict)

print('Results saved!')