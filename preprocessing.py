# ---- Script to load LALinf. samples/data and make sampleDict object to be used in emcee runs ----
# ---- Simona Miller, created Nov. 2019, slight updates April 2020 ----


'''
Set-up
'''

# -- Imports -- 
import numpy as np
from scipy.stats import gaussian_kde
import h5py
import astropy.cosmology as cosmo
import astropy.units as u
from astropy.cosmology import Planck15

# -- Names of BBH events -- 
BBH_names = ['GW151012', 'GW170608', 'GW170729', 'GW151226', 'GW170814', 'GW150914', 'GW170104', 'GW170809', 'GW170818', 'GW170823']

# -- Create dictionary to store all data in -- 
sampleDict = {}

# -- Constants we'll need -- 
c = 3.0e8          # m/s
H_0 = 67270.0      # m/s/MPc
Omega_M = 0.3156 # unitless
Omega_Lambda = 1.0-Omega_M

# -- Reference distributions for luminosity distance and redshift -- 
z_ref = np.linspace(0.,3.,1000)
DL_ref = Planck15.luminosity_distance(z_ref).to(u.Mpc).value




'''
Functions used in script
'''

# -- To calculate Chi-effective given component spin magnitudes, cos(tilt angle), and masses -- 
def calculate_Xeff(a1, m1, cost1, a2, m2, cost2):
    Xeff = (a1*m1*cost1 + a2*m2*cost2)/(m1 + m2)
    return Xeff

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
Cycling through all ten events to ...
1. Calculate Xeff POSTERIOR samples given component spin magnitudes, cos(tilt angles), and masses
2. Calculate Xeff PRIOR samples given sampels from spin magnitude, cos(tilt angle), and mass priors
3. Generate Gaussian Kernel Density Estimate (KDE) from the Xeff prior samples
4. Evaluate the prior at the Xeff posterior samples
5. Calculate ratio of LALinference prior to astrophysically motivated prior for masses and redshift
6. Save all samples in a dictionary
'''

# Grid for normalization of KDE
dX = 0.01
grid = np.arange(-1.0, 1.0, dX)

for name in BBH_names:
    
    # -- POSTERIORS --
    
    # load posterior samples
    BBH_post = h5py.File('./GWTC-1_event_samples/'+name+'_GWTC-1.hdf5', 'r')['Overall_posterior']
    m1 = BBH_post['m1_detector_frame_Msun']
    m2 = BBH_post['m2_detector_frame_Msun']
    a1 = BBH_post['spin1']
    a2 = BBH_post['spin2']
    cost1 = BBH_post['costilt1']
    cost2 = BBH_post['costilt2']
    
    # calculate Xeff posterior samples
    Xeff_post_samples = calculate_Xeff(a1, m1, cost1, a2, m2, cost2)
    
    print('Xeff posterior samples generated for {}.'.format(name))
    
    
     # -- PRIORS --
    
    # load prior samples
    BBH_prior = h5py.File('GWTC-1_event_samples/'+name+'_GWTC-1.hdf5', 'r')['prior']
    m1_prior = BBH_prior['m1_detector_frame_Msun']
    m2_prior = BBH_prior['m2_detector_frame_Msun']
    a1_prior = BBH_prior['spin1']
    a2_prior = BBH_prior['spin2']
    cost1_prior = BBH_prior['costilt1']
    cost2_prior = BBH_prior['costilt2']
    
    # calculate Xeff prior samples
    Xeff_prior_samples = calculate_Xeff(a1_prior, m1_prior, cost1_prior, a2_prior, m2_prior, cost2_prior)
    
    print('Xeff prior samples generated for {}.'.format(name))
    
    # Make Gaussian KDE of prior samples
    Xeff_prior_KDE = gaussian_kde(Xeff_prior_samples)
    
    print('Prior KDE fxn generated for {}.'.format(name))
    
    # Calculate normalization coefficient 
    norm = dX*np.sum(Xeff_prior_KDE(grid))
    
    # Evalulate KDE at Xeff posterior samples
    Xeff_prior = (1.0/norm)*Xeff_prior_KDE(Xeff_post_samples)
    
    print('Prior KDE fxn evaluated on Xeff posterior samples for {}.'.format(name))
    
    
    # -- MASS AND REDSHIFT PRIOR REWEIGHTING --
    
    DL = BBH_post['luminosity_distance_Mpc']     # luminosity distance
    z = np.asarray(np.interp(DL, DL_ref, z_ref)) # redshift -- interpolate from z and DL reference distribution 
    
    # Calculate pASTRO
    pASTRO = calculate_pASTRO(z, DL, m1)
    pASTRO[pASTRO<0] = 0 # if pASTRO < 0, make pASTRO = 0
    
    # Calculate pLAL
    pLAL = calculate_pLAL(z, DL)
    
    # weights = ratio between the two 
    weights = pASTRO/pLAL
    
    print('Weights generated for {}.'.format(name))
    
    
    # -- SAVING RESULTS -- 
    
    indiv_event_dict = {\
                        'Xeff': Xeff_post_samples,\
                        'Xeff_prior': Xeff_prior,\
                        'weights':weights
                       }
    
    sampleDict[name] = indiv_event_dict
    
    print('Samples saved for {}.'.format(name))
    

print('...')

np.save('./sampleDict.npy', sampleDict) 


print('Done!')
