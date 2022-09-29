import pandas as pd
from get_distribution_type import fitter_best_distribution
from scipy.stats import cauchy, lognorm, powerlaw, chi2, norm, uniform, exponpow
import scipy
import matplotlib.pyplot as plt
import numpy as np


def sampling_flexible_distribution(sample_size, **kwargs):

    sample_list = {}
    for Ordnung in kwargs.keys():

        dist_str = list(kwargs[Ordnung].keys())[0]
        dist_object = getattr(scipy.stats, dist_str)
        parameter = kwargs[Ordnung][dist_str]
        parameter["size"] = sample_size

        samples = dist_object.rvs(**parameter)
        sample_list[Ordnung] = samples

    return sample_list



if __name__ == "__main__":

    dicti = {'hri_O_0.21': {'lognorm': {'s': 0.008220691268158182, 'loc': -7.0297729615879145, 'scale': 7.555459544706255}}, 'hri_O_0.31': {'powerlaw': {'a': 2.0399471427612914, 'loc': 0.4027622066593996, 'scale': 0.2739058471438275}}, 'hri_O_0.42': {'powerlaw': {'a': 2.8806826777751438, 'loc': 0.480724870018821, 'scale': 0.16792922473365798}}}

    sample_list = sampling_flexible_distribution(sample_size=10000, **dicti)
    print(sample_list)
    
    for key in sample_list:
        
        data = sample_list[key]
        
        plt.hist(data, density=True, bins=30)  # density=False would make counts
        plt.ylabel('Probability')
        plt.xlabel('Data')
        plt.show()
        
        quit()

