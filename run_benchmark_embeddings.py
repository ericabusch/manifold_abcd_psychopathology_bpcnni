'''
run_benchmark_embeddings.py

This script is used to make embeddings of brain activation data using PCA and UMAP, which are the two standard metrics we benchmark
PHATE and EPHATE against.

runs from command line as:
`python run_benchmark_embeddings.py -e $EVENT_NAME -r $ROI_NAME -c $CONTRAST_NAME`

ELB 2024
'''

import numpy as np
import pandas as pd
import os,sys,glob,argparse,pickle
from config import *
from sklearn.preprocessing import MinMaxScaler
import numpy.linalg 
import utils
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
import umap

def run_pca(X, outfilename, n_components=[2]):
    X_pca=PCA().fit_transform(X)
    for nc in n_components:
        np.save(f'{outfilename}_{nc}d_embedding.npy', X_pca[:,:nc])

def run_umap(X, outfilename, n_components=[2]):
    for nc in n_components:
        embedding = umap.UMAP(n_neighbors=5, n_components=nc).fit_transform(X)
        np.save(f'{outfilename}_{nc}d_embedding.npy', embedding) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--event')
    parser.add_argument('-r','--ROI_name')
    parser.add_argument('-c','--contrast')
    parser.add_argument('-b','--rerun', default=True)
    p = parser.parse_args()
    
    
    if VERBOSE: print(f"running event={p.event}, contrast={p.contrast}, ROI={p.ROI_name}")

    BENCHMARKS={'PCA':run_pca,'UMAP':run_umap}

    sublist = utils.select_brain_data_subjects(p.event, p.contrast)
    if VERBOSE: print(f"loaded {len(sublist)} from sublist")
    data, sublist = utils.load_brain_data(sublist, p.event, p.contrast, p.ROI_name)
    data = np.nan_to_num(data)
    print(f'nans? {np.sum(data!=data)}')
    if VERBOSE: print(f'updated subject list has {len(sublist)} and shape {data.shape}')
    
    for METHOD, FUNC in BENCHMARKS.items():
        os.makedirs(EMBEDDING_DIRECTORIES[METHOD], exist_ok=True)
        print(METHOD, EMBEDDING_DIRECTORIES[METHOD])
        outfilename = os.path.join(EMBEDDING_DIRECTORIES[METHOD], f'{p.event}_{p.contrast}_{p.ROI_name}_{METHOD}')
        if VERBOSE: print(f'Will save to: {outfilename}')
        FUNC(data, outfilename, n_components=N_COMPONENTS)

    
    
        