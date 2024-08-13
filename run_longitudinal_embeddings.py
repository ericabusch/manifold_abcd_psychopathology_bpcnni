'''
This script takes baseline brain data and embeds it, for the participants who are included in the 2-year follow up.
'''

import numpy as np
import argparse
import utils
from config import *
import pandas as pd
import run_vanilla_phate as vphate
import run_exogenous_phate as ephate
import run_benchmark_embeddings as benchmarks
import run_phate_plus_features as vphate_plus
from scipy.stats import zscore

def select_longitudinal_subjects(contrast):
    sublist_baseline =  utils.select_brain_data_subjects("baseline", contrast)
    df = utils.get_included_subject_dataframe()
    include_subs = df[df['include_longitudinal_cohort']==1]['subjectkey_fmt'].values
    # get overlap
    overlap = [s for s in sublist_baseline if s in include_subs]
    return overlap

def select_embedding_function(embedding_type):

    functions = {'vanilla_PHATE': vphate.run_vanilla_phate,
    'UMAP':benchmarks.run_umap,
    'PCA':benchmarks.run_pca,
    'PHATE_PLUS_FEATURES': vphate_plus.run_phate_plus_features,
    'EPHATE': ephate.run_ephate}
    return functions[embedding_type]

def load_data(ROI_name, contrast, additional_features=None, load_affinity_mat=False):
    
    data, sublist = utils.load_brain_data([], "longitudinal", p.contrast, p.ROI_name)
    sublist_nofam = select_longitudinal_subjects(p.contrast)
    # sublist_nofam = utils.check_family_repeats(overlapping_subs)

    if VERBOSE: print(f'there are {len(sublist_nofam)} with both years, 1per fam, {len(sublist)} originally')
    # get indices of included subjects
    indices, final_subs = [], []
    for i,s in enumerate(sublist):
        if s in sublist_nofam:
            indices.append(i)
            final_subs.append(s)
    data = np.nan_to_num(data[indices,:])
    sublist = [sublist[i] for i in indices]
    # also check family
    if VERBOSE: print(f'Sorted correctly? {sublist == final_subs}')
    if VERBOSE: print(f'updated subject list has {len(sublist)} and shape {data.shape}')
    PKG=None
    if additional_features:
        if load_affinity_mat:
            aff_fn = AFFINITY_MATRICES[additional_features]
            M_F = pd.read_csv(aff_fn, index_col=0)
            # take just the subjects here
            M_F = np.nan_to_num(M_F.loc[sublist, sublist].values)
            PKG=[data, sublist, M_F]
        else:
            # load the features
            feature_names = FEATURES_PER_MAT['5']
            variable_names = [PREDICTOR_TO_VARIABLE[f] for f in feature_names]
            F = []
            for f,v in zip(feature_names,variable_names):
                f = np.nan_to_num(zscore(utils.load_variables(sublist, 'baseline', [f], v, check_reverse=True)))
                F.append(f)
            F = np.array(F).T
            if VERBOSE: print(f'loaded features {feature_names} of shape {F.shape}')
            # concatenate the data
            data = np.concatenate((data, F),axis=1)
    if PKG==None:
        PKG=[data, sublist]
    outdir=f'{PROJECT_DIR}/data/neuroimaging_data_aggregated'
    # write out the subject list
    print(f'writing out {len(sublist)} to file {outdir}/sublist_{contrast}_longitudinal.txt')
    with open(f'{outdir}/sublist_{contrast}_longitudinal.txt','w') as f:
        for s in sublist:
            f.write(s+'\n')
    return PKG





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r','--ROI_name')
    parser.add_argument('-c','--contrast')
    parser.add_argument('-b', '--embedding_type')
    parser.add_argument('-o','--overwrite', default=True)
    p=parser.parse_args()
    EMBEDDING_TYPE=p.embedding_type
    ROI_name=p.ROI_name
    CONTRAST=p.contrast
    
    additional_features=None
    load_affmat=False
    if "EPHATE" in EMBEDDING_TYPE:
        addit = EMBEDDING_TYPE.split('_')
        if len(addit) > 2: 
            addit=f'{addit[1]}_{addit[2]}'
        else: 
            addit=addit[1] 
        additional_features=addit
        load_affmat=True

    elif EMBEDDING_TYPE=='PHATE_PLUS_FEATURES':
        additional_features=True
        load_affmat=False

    DATA_PACK=load_data(ROI_name,CONTRAST,additional_features=additional_features, load_affinity_mat=load_affmat)
    outfilename=os.path.join(EMBEDDING_DIRECTORIES[EMBEDDING_TYPE], f'longitudinal_{CONTRAST}_{ROI_name}_{EMBEDDING_TYPE}')
    if os.path.exists(outfilename+f'20d_embedding.npy'): print(f'already ran {outfilename}; ending'); sys.exit(0)
    if VERBOSE: print(f'will save to {outfilename}')
    if "EPHATE" in EMBEDDING_TYPE: 
        FUNC = select_embedding_function("EPHATE")
        FUNC(DATA_PACK[0], DATA_PACK[2], outfilename, N_COMPONENTS)
    else: 
        FUNC = select_embedding_function(EMBEDDING_TYPE)
        FUNC(DATA_PACK[0], outfilename, N_COMPONENTS)





