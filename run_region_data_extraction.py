'''
Script to collect regional beta weights for each participant into single matrices for use in downstream analyses.
'''
import numpy as np
import pandas as pd
import os,sys,glob
import utils
import nibabel as nib
from scipy.stats import zscore
import nilearn
import pickle , argparse
from joblib import Parallel, delayed
from config import *

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--contrast')
    parser.add_argument('-e', '--cohort_name')
    p = parser.parse_args()
    
    subject_df = pd.read_csv(f'{PROJECT_DIR}/data/subjects_included.csv')
    if VERBOSE: print(f'df of shape {subject_df.shape}')
    if p.cohort_name == 'baseline': 
        subjects = subject_df['subjectkey'].values
    else: 
        subjects = subject_df[subject_df['include_longitudinal_cohort'] == 1]['subjectkey'].values
    subjects = utils.check_family_repeats(subjects)
    subjects=[s.replace('NDAR_',"") for s in subjects]
    utils.extract_region_data(subjects, p.cohort_name, p.contrast, REGIONS, outdir=f'{PROJECT_DIR}/data/neuroimaging_data_aggregated')
            
            