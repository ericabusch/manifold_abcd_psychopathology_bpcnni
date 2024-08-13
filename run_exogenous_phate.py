import numpy as np
import pandas as pd
import os,sys,glob,argparse,pickle
import phate, scprep
from scipy.spatial.distance import pdist, squareform
from scipy.stats import zscore, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import mds
from config import *
from sklearn.preprocessing import MinMaxScaler
import numpy.linalg 
import utils


def combine(diffmat1, diffmat2):
    return np.matmul(diffmat1.T, diffmat2)

def row_normalize(M):
    for row in M:
        if np.sum(row) == 0: # this should never be true
            continue
        row[:] /= np.sum(row)
    return M 

def make_diffusion_op(X, M_F, sample_labels, filename, save_diffmat=True, n_components=[2], return_diffop=False):
    # X is the data to build PHATE off
    # M_F is the second view affinity matrix
    # sample labels are used to save
    phate_op=phate.PHATE(verbose=0, n_pca=500, decay=100, n_landmark=X.shape[0], t=4, knn=4)
    phate_op.fit(np.nan_to_num(X))
    # row normalize
    M_F = row_normalize(M_F)

    # combine 
    D = np.nan_to_num(combine(phate_op.diff_op, M_F))

    if return_diffop: return D
    # embed
    embed2 = make_embeddings(D, n_components, filename)
    
    if save_diffmat:
        temp = pd.DataFrame(columns=sample_labels, index=sample_labels, data=D)
        temp.to_csv(f'{filename}_combined_diffmat.csv')
        temp = pd.DataFrame(columns=sample_labels, index=sample_labels, data=M_F)
        temp.to_csv(f'{filename}_feature_diffmat.csv')
        temp = pd.DataFrame(columns=sample_labels, index=sample_labels, data=phate_op.diff_op)
        temp.to_csv(f'{filename}_phate_diffmat.csv')
    return embed2

def make_embeddings(D, n_components, filename):
    to_return = None
    for nc in n_components:
        embedding = mds.embed_MDS(D, ndim=nc)
        fn = f'{filename}_{nc}d_embedding.npy'
        print(fn)
        np.save(fn, embedding)
        if nc == 2: to_return = embedding
    return to_return


def exogenous_phate(X, F, sample_labels, n_components=[2], metric='euclidean', outfilename=None):
    if F.shape[0] == F.shape[1]:  M_F = F
    else: M_F = build_feature_affinity_matrix(F, sample_labels, metric=metric)
    if outfilename == None:
        save = False
    embedding = make_diffusion_op(X, M_F, sample_labels, outfilename, save_diffmat=save, n_components=n_components)
    return embedding

def run_ephate(X, M_F, outfilename, n_components=[2]):
    D = make_diffusion_op(X, M_F, sample_labels=None, filename=None, save_diffmat=False, n_components=N_COMPONENTS, return_diffop=True)
    print(f"got diffop shape {D.shape}")
    _ = make_embeddings(D, N_COMPONENTS, outfilename)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--event')
    parser.add_argument('-r','--ROI_name')
    parser.add_argument('-c','--contrast')
    parser.add_argument('-f','--view2_features', default='5')
    parser.add_argument('-b','--rerun', default=True)
    p = parser.parse_args()
    
    
    
    if VERBOSE: print(f"running event={p.event}, contrast={p.contrast}, diff feature={p.view2_features}, ROI={p.ROI_name}")
    t=p.view2_features
    aff_fn = AFFINITY_MATRICES[t]
        
    F = f'EPHATE_{t}'
    os.makedirs(EMBEDDING_DIRECTORIES[F], exist_ok=True)
    outfilename = os.path.join(EMBEDDING_DIRECTORIES[F], f'{p.event}_{p.contrast}_{p.ROI_name}_{F}')
    sublist = utils.select_brain_data_subjects(p.event, p.contrast)
    if VERBOSE: print(f"loaded {len(sublist)} from sublist")
    data, sublist = utils.load_brain_data(sublist, p.event, p.contrast, p.ROI_name)
    if VERBOSE: print(f'updated subject list has {len(sublist)} and shape {data.shape}')
    if VERBOSE: print(f'Will save to: {outfilename}')
    
    M_F = pd.read_csv(aff_fn, index_col=0)
    # take just the subjects here
    M_F = M_F.loc[sublist, sublist].values
    if VERBOSE: print(f'after masking, M_F.nan={np.sum(M_F!=M_F)}, data.nan={np.sum(data!=data)}')
    embedding = make_diffusion_op(data, M_F, sublist, outfilename, n_components=N_COMPONENTS)
    
        