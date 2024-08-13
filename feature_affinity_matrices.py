'''
feature_affinity_matrices.py

This script is used to pre-compute affinity matrices among the environmental variables for use in the E-PHATE procedure.
Pre-computation is more efficient due to the large number of samples; EPHATE can take a pre-computed affinity matrix 
or raw feature matrix and compute affinities. Here, pairwise similarity can be measured with euclidean similarity, cosine similarity,
pearson's correlation, or spearman's correlation.

runs from command line as:
`python feature_affinity_matrices.py -e $EVENT_NAME -f $WHAT_FEATURES -m $WHAT_SIMILARITY_METRIC`

ELB 2024

'''

import numpy as np
import pandas as pd
import os,sys,glob,argparse,pickle
import utils
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore
from config import *
import numpy.linalg

def euclidean_similarity_func(u, v):
    dist = np.sqrt(np.sum((e1-e2)**2 for e1, e2 in zip(u,v)))
    return 1/(1+dist)

def cosine_similarity_func(u,v):
    cosine = np.dot(u,v)/(numpy.linalg.norm(u)*numpy.linalg.norm(v))
    return cosine

def pearsonr_func(u,v):
    return np.corrcoef(u,v)[0,1]

def spearmanr_func(u,v):
    return spearmanr(u,v).correlation

def build_feature_affinity_matrix(feature_data, sample_labels, metric='euclidean'):
    if metric == 'euclidean':
        aff = squareform(pdist(np.nan_to_num(feature_data),  euclidean_similarity_func)) 
    elif metric == 'cosine':
        aff = squareform(pdist(np.nan_to_num(feature_data),  cosine_similarity_func)) 
    elif metric == 'pearsonr':
        aff = squareform(pdist(np.nan_to_num(feature_data),  pearsonr_func)) 
    elif metric == 'spearmanr':
        aff = squareform(pdist(np.nan_to_num(feature_data),  spearmanr_func)) 
    else:
        print(f'{metric} not implemented')
        return None
    aff[np.diag_indices_from(aff)] = 1
    return pd.DataFrame(columns=sample_labels, index=sample_labels, data=aff)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--event', default='baseline')
    parser.add_argument('-f','--view2_features', default='5')
    parser.add_argument('-m','--distance_metric', default='euclidean')
    p = parser.parse_args()
    
    feature_names = FEATURES_PER_MAT[p.view2_features]
    variable_names = [PREDICTOR_TO_VARIABLE[f] for f in feature_names]
    
    sub_df = utils.get_included_subject_dataframe()
    sublist = sub_df['subjectkey'].values
    
    F = []
    for f,v in zip(feature_names,variable_names):
        f = zscore(utils.load_variables(sublist, p.event, [f], v, check_reverse=True))
        F.append(f)
    F = np.array(F)
    if VERBOSE: print(f'loaded features {feature_names} of shape {F.shape}')
    aff_mat_outname = AFFINITY_MATRICES[p.view2_features]
    if VERBOSE and p.distance_metric != 'euclidean': print(f'check your distance metric'); sys.exit(1)
    aff_mat = build_feature_affinity_matrix(F.T, sublist, p.distance_metric)
    aff_mat.to_csv(aff_mat_outname)
    if VERBOSE: print(f'Done with {aff_mat_outname}')
    
            


