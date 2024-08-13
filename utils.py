import os, sys, glob
import numpy as np
import pandas as pd
from scipy.stats import zscore
import nibabel as nib
import pickle
# import feature_phate
from config import *
from scipy.stats import bootstrap



def reverse_score(y):
    ymax = np.nanmax(y)
    return ymax + 1 - y

def format_subjects(sublist):
    if sublist[0][:4]=='NDAR':
        return [s for s in sublist]
    return [f'NDAR_{f}' for f in sublist]

def check_family_repeats(sublist):
    # check for repeat from family
    rel_fam = load_variables(sublist, 'baseline','rel_family_id','FAM', check_reverse=False)
    uniq_fam = np.unique(rel_fam)
    idx_include = []
    for i,f in enumerate(uniq_fam):
        idx= np.where(rel_fam==f)[0]     
        idx_include.append(idx[0])
    sublist_final=[]
    for i,s in enumerate(sublist):
        if i in idx_include:
            sublist_final.append(s)
    rel_fam = load_variables(sublist_final, 'baseline','rel_family_id','FAM', check_reverse=False)
    assert len(rel_fam) == len(np.unique(rel_fam))
    return sublist_final

def select_brain_data_subjects(event, contrast):
    subject_filename = f'{MYDATA_DIR}/sublist_{event}_{contrast}.txt'
    with open(subject_filename, 'r') as f:
        subjects = f.readlines()
    subjects = [l.strip() for l in subjects]
    subjects = check_family_repeats(subjects)
    return subjects

def load_brain_data(sublist, event, contrast, ROI_name):
    subjects = select_brain_data_subjects(event, contrast)
    if VERBOSE: print(f'subjects shape : {len(subjects)}')
    if event == 'longitudinal': 
        filename = f'{MYDATA_DIR}/baseline_{contrast}_{ROI_name}_data.pkl'
    else:
        filename=f'{MYDATA_DIR}/{event}_{contrast}_{ROI_name}_data.pkl'
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    if VERBOSE: print(f'data shape : {data.shape}')

    if event == 'longitudinal':
        overall_subjects = select_brain_data_subjects('baseline', contrast)
        idx_include = []
        for i, s in enumerate(overall_subjects):
            if s in subjects:
                idx_include.append(i)
        idx_include=np.array(idx_include)
        data=data[idx_include]
        if VERBOSE: print(f'data shape : {data.shape}; subjects : {len(subjects)}')
    return data , subjects
    

def load_variables(sublist, event, column_variables, variable_name, check_reverse=True):
    sublist = format_subjects(sublist)
    df = pd.read_csv(VARIABLE_FILENAMES[variable_name], index_col=0, low_memory=False, sep='\t')
    df = df[(df['eventname'] == YEAR_CODES[event]) & (df['subjectkey'].isin(sublist))]
    vals = []
    for sub in sublist:
        val = df[df['subjectkey']==sub][column_variables].values
        if len(val) == 0: 
            val = np.array([0.0])
        try:
            val = float(val[0])
        except:
            val=val
        vals.append(val)
    try:
        vals = np.squeeze(vals)
    except:
        return vals
    if check_reverse and column_variables[0] in TO_REVERSE_SCORE:
        vals = reverse_score(vals)
    return vals

def load_atlas():
    nii = nib.load(DLABEL_FN).get_fdata()
    with open(LABEL_TXT_FN, 'r') as f:
        labels = f.readlines()
    labels = [l.strip() for l in labels]
    label_dict = {}
    for i in range(0, len(labels), 2):
        indices = labels[i+1].strip().split(' ')
        indices = [int(x) for x in indices] 
        label_dict[labels[i]] = indices
    return nii, label_dict

def return_brain_map_subject_exclusions(all_wb_data):
    exclude_idx, exclude_260, exclude_mean, exclude_sd = [],[],[],[]

    means = np.nanmean(all_wb_data, axis=1)
    mean_mu, mean_sd = np.nanmean(means), np.nanstd(means)
    stdevs = np.nanstd(all_wb_data, axis=1)
    stdevs_mu, stdevs_sd = np.nanmean(stdevs), np.nanstd(stdevs)

    for i in range(len(all_wb_data)):
        # more missing values than just the medial wall
        if np.sum(all_wb_data[i]!=all_wb_data[i]) > 260 * 2:
            exclude_260.append(i)
            exclude_idx.append(i)

        # mean outside of 3st+/-group mean
        elif (means[i] > mean_mu + 3*mean_sd) or (means[i] < mean_mu - 3*mean_sd):
            exclude_mean.append(i)
            exclude_idx.append(i)

        elif (stdevs[i] > stdevs_mu + 3*stdevs_sd) or (stdevs[i] < stdevs_mu - 3*stdevs_sd):
            exclude_sd.append(i)
            exclude_idx.append(i)

    if VERBOSE: print(f"excluding: {len(exclude_idx)} subjects")
    if VERBOSE: print(f'excluding based on: mean activation ({len(exclude_mean)}), std activation ({len(exclude_sd)}), missing values ({len(exclude_260)})')
    return np.array(exclude_idx)

def load_release4_data_sublist(subject_list, eventname, taskname, contrastname):
    data = []
    for s in subject_list:
        fn=f'{RELEASE4_CIFTI}/{s}_{eventname}/{taskname}/{contrastname}_beta.dscalar.nii'
        x=nib.load(fn).get_fdata()
        data.append(x)
    return np.array(data)

def get_included_subject_dataframe():
     return pd.read_csv(f'{PROJECT_DIR}/data/subjects_included.csv')


def extract_region_data(subject_list, eventname, contrastname, region_list, taskname='nBack', run_exclusions=True, outdir=None):
    data_by_region = {}
    # load all the beta maps in order for subjects in subject list
    wb_data = np.squeeze(load_release4_data_sublist(subject_list, 'baseline', taskname, contrastname))
    if VERBOSE: print(f'loaded data of shape {wb_data.shape} ')
    # load the atlas and label mapping
    atlas_nii, atlas_label_dict = load_atlas()
    if run_exclusions:
        exclude_idx = return_brain_map_subject_exclusions(wb_data)
        updated_subidx = np.setdiff1d(np.arange(len(wb_data)), exclude_idx)
        subject_list = [subject_list[i] for i in updated_subidx]
        wb_data = wb_data[updated_subidx]
        if VERBOSE: print(f'After excluding {len(exclude_idx)} subjects, WB data of shape {wb_data.shape}')

    # get the index of a region from the label dictionary - regardless of the subregion / network / hemisphere
    get_region_idx = lambda r : np.array([v[0] for k, v in atlas_label_dict.items() if r in k])
    region_index_mapping = {r : get_region_idx(r) for r in region_list}

    # get the vertices that belong to each region
    get_roi_atlas_vertices = lambda r : np.concatenate([np.where(atlas_nii == region_index_mapping[r][i])[1] for i in range(len(region_index_mapping[r]))])
    atlas_vertices = {r : get_roi_atlas_vertices(r) for r in region_list}

    for region, region_atlas_idx in region_index_mapping.items():
        vertices = atlas_vertices[region]
        if VERBOSE: print(f'{region} is index {region_atlas_idx} and has {len(vertices)} vertices')
        # now pull out that data
        data_by_region[region] = wb_data[:, vertices]
        if VERBOSE: print(f'After selection, {region} data is of shape {data_by_region[region].shape}')

    if outdir:
        # write out data
        for region, data in data_by_region.items():
            outfn = f'{outdir}/{eventname}_{contrastname}_{region}_data.pkl'
            with open(outfn,'wb') as f:
                pickle.dump(data, f)
            if VERBOSE: print(f'saved {outfn}')
        # write out subjects
        with open(f'{outdir}/sublist_{eventname}_{contrastname}.txt','w') as f:
            for s in subject_list:
                f.write(s+'\n')
        return
    return data_by_region


def load_diffmat_and_embedding(eventname, contrastname, ROI_name, embedding_type, ndim, return_subjects=True):
    dirname = EMBEDDING_DIRECTORIES[embedding_type]
    embedding_fn = f'{dirname}/{eventname}_{contrastname}_{ROI_name}_{embedding_type}_{ndim}d_embedding.npy'
    if embedding_type == 'vanilla_PHATE':
        diffmat_fn = f'{dirname}/{eventname}_{contrastname}_{ROI_name}_{embedding_type}_phate_diffmat.csv'
    else:
        diffmat_fn = f'{dirname}/{eventname}_{contrastname}_{ROI_name}_{embedding_type}_combined_diffmat.csv'
    
    a=diffmat_fn.split('/')[-1]
    b=embedding_fn.split('/')[-1]
    df = pd.read_csv(diffmat_fn, index_col=0)
    embed = np.load(embedding_fn)
    if VERBOSE: print(f'loaded {a}, embed={b}, return_subjects={return_subjects}')
    if VERBOSE: print(f'df of shape {df.shape}; embedding of shape: {embed.shape}')
    if return_subjects: return list(df.index), embed
    return df, embed

def load_embedding(cohort_name, contrast_name, ROI_name, embedding_type, ndim):
    dirname = EMBEDDING_DIRECTORIES[embedding_type]
    embedding_fn = f'{dirname}/{cohort_name}_{contrast_name}_{ROI_name}_{embedding_type}_{ndim}d_embedding.npy'
    embed = np.load(embedding_fn)
    return embed




