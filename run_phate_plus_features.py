'''
run_phate_plus_features.py

This script is used to make embeddings by combining brain activation with environmental 
features by concatenating them, as opposed to the dual-diffusion procedure in EPHATE.  

runs from command line as:
`python run_phate_plus_features.py -e $EVENT_NAME -r $ROI_NAME -c $CONTRAST_NAME`


ELB 2024
'''

import numpy as np
import pandas as pd
import os,sys,glob,argparse,pickle
import utils
import phate
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore
import mds
from config import *

def make_diffusion_op(X, sample_labels, filename, save_diffmat=True, n_components=[2]):
    # X is the data to build PHATE off
    # sample labels are used to save
    phate_op = phate.PHATE(verbose=0, n_pca=500, decay=6, n_landmark=X.shape[0], t=4, knn=4)
    phate_op.fit(np.nan_to_num(X))
    D = phate_op.diff_op    
    if save_diffmat:
        temp = pd.DataFrame(columns=sample_labels, index=sample_labels, data=D)
        temp.to_csv(f'{filename}_phate_diffmat.csv')
    return D

def make_embeddings(D, n_components, filename):
    to_return = None
    for nc in n_components:
        embedding = mds.embed_MDS(D, ndim=nc)
        fn = f'{filename}_{nc}d_embedding.npy'
        if VERBOSE: print(fn)
        np.save(fn, embedding)
        if nc == 2: to_return = embedding
    return to_return

def run_phate_plus_features(X,  outfilename, n_components=[2]):
    D = make_diffusion_op(X, None, None, False)
    _ = make_embeddings(D, n_components, outfilename)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--event')
    parser.add_argument('-r','--ROI_name')
    parser.add_argument('-c','--contrast')
    parser.add_argument('-b','--rerun',default=True)
    p = parser.parse_args()
        
    if VERBOSE: print(f"running event={p.event}, contrast={p.contrast}, ROI={p.ROI_name}")
    
    os.makedirs(EMBEDDING_DIRECTORIES['PHATE_PLUS_FEATURES'], exist_ok=True)
    outfilename = os.path.join(EMBEDDING_DIRECTORIES['PHATE_PLUS_FEATURES'], f'{p.event}_{p.contrast}_{p.ROI_name}_PHATE_PLUS_FEATURES')
    sublist = utils.select_brain_data_subjects(p.event, p.contrast)
    if VERBOSE: print(f"loaded {len(sublist)} from sublist")
    data, sublist = utils.load_brain_data(sublist, p.event, p.contrast, p.ROI_name)
    if VERBOSE: print(f'updated subject list has {len(sublist)} and shape {data.shape}')

    # load the features
    feature_names = FEATURES_PER_MAT['5']
    variable_names = [PREDICTOR_TO_VARIABLE[f] for f in feature_names]
    F = []
    for f,v in zip(feature_names,variable_names):
        f = zscore(utils.load_variables(sublist, p.event, [f], v, check_reverse=True))
        F.append(f)
    F = np.array(F).T
    if VERBOSE: print(f'loaded features {feature_names} of shape {F.shape}')
    # concatenate the data
    data = np.concatenate((data, F),axis=1)

    if VERBOSE: print(f'Final data of shape {data.shape}; Will save to: {outfilename}')
    diffmat = make_diffusion_op(data, sublist, outfilename, n_components=N_COMPONENTS)
    embedding = make_embeddings(diffmat, N_COMPONENTS, outfilename) # embed the phate diffusion mat that was already made. returns 2d
            
