# LIGO GWTC-1 chi-eff population analysis
This respository contains examples of the code used for the analyses performed in my undergraduate thesis at Smith college. I hierarchically measure the mean $\mu$ and variance $\sigma^2$ of the effective-spin (chi-eff, $\chi_\mathrm{eff}$) distribution across binary black holes using LIGO's GWTC-1. 

This work also appears in my publication Miller et. al, The Low Effective Spin of Binary Black Holes and Implications for Individual Gravitational-Wave Events, published Jan 2020: https://arxiv.org/abs/2001.06051

The scripts, notebooks, and other files in this repository are as follows, with scripts to be updated in this order:
1. `GWTC-1_event_samples` -- Folder containing the posterior and prior samples for each of LIGO's ten binary black hole (BBH) detections. These samples are of course publically available, and can be downloaded [here](https://dcc.ligo.org/LIGO-P1800370/public). 
2. `preprocessing.py` -- Scipt to load all the samples from the `GWTC-1_event_samples` folder and organize them into a way that can be used in the way we desire with the MCMC package `emcee`. Specifically, this script loads the spin magnitudes, tilt angles, masses, and luminosity distances, and calculates $\chi_\mathrm{eff}$ posterior and prior samples, as well as factors to re-weight the mass, redshift, and luminosity distance priors. Outputs `sampleDict.npy` file which contains the dict with the $\chi_\mathrm{eff}$ samples for each of the ten BBH, with the key equal to the event's name (e.g. "GW150914"). 
3. `run_emcee.py` -- Script to run `emcee` to measure the mean and variance of the $\chi_\mathrm{eff}$ distribution. Outputs `mu_sigma2_samples.npy` which contains an array of all 16 raw sample chains for the mean and variance from the MCMC.
4. `postprocessing.py` -- Takes the raw samples in `mu_sigma2_samples.npy` and processes them into a way we can use them, but discarding early samples in the chains, downsampling, and then flattening the remaining samples into 1D arrays. Outputs`mu_sigma2_samples_postprocessed.npy` which contains the final, post-processed $\mu$ and $\sigma^2$ samples
5.  `posterior_reweighting_indiv_events.py` -- 
6. `plotting.ipynb` -- 
