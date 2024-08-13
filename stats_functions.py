import numpy as np
import pandas as pd
import utils
import os, sys, glob
import itertools
import statsmodels.api as sm
from statsmodels.stats import multitest
from scipy.spatial.distance import cdist, pdist, squareform
from random import choices, choice
from sklearn.preprocessing import MinMaxScaler
import corrstats
from scipy import stats
from functools import reduce
from config import * 


def return_bootstrap_info(df1, stat_name='partial_corr_rho'):
    result_df = pd.DataFrame(columns=['ROI','contrast','embedding_type','predictor','lower','upper','mean'])
    for v in df1.predictor.unique():
        for meth in df1.embedding_type.unique():
            for roi in df1.ROI_name.unique():
                for con in df1.contrast.unique():
                    cols = ['predictor','ROI_name','embedding_type','contrast']
                    vals = [v,roi,meth,con]
                    df_here = extract_results(df1, cols, vals)
                    x = df_here[stat_name].values
                    res = stats.bootstrap((x,), np.mean, confidence_level=0.95, n_resamples=1000, random_state=0)
                    lower = np.round(res.confidence_interval.low,3)
                    upper = np.round(res.confidence_interval.high,3)
                    mean=np.round(np.mean(x),3)
                    
                    result_df.loc[len(result_df)] = {'ROI':roi, 'contrast':con, 'embedding_type':meth,  'predictor':v, 'lower':lower, 'upper':upper, 'mean':mean}
                    
    return result_df

def run_pairwise_stats(dataframe, yname, embeddings_to_compare, nperm=1000, alt=None):
    stat_df = pd.DataFrame(columns=['contrast','ROI_name', 'r2z_p','xy','ab','name_ab','name_xy','n_per_fold',
                                    'sorted_comparison','yname','perm_p','corrected_p','symbol','corrected_pperm', 'combostr'])
    # compares prediction of a certain attribute (already filtered) within contrast across ROIs and embedding methods and calcualtes / returns pvals
    cols=['contrast','ROI_name','embedding_type']
    for con in dataframe.contrast.unique():
        for roi in dataframe.ROI_name.unique():
            vals=[con,roi]
            result_dict = {}
            N = []
            for embed in embeddings_to_compare:
                result_dict[embed] = extract_results(dataframe, cols, vals+[embed], statistic=yname)
                N.append(extract_results(dataframe, cols, vals+[embed], statistic='n_test').mean())
            # print(embed,con,roi,result_dict.keys())
            for combo in itertools.combinations(list(result_dict.keys()), 2):
                combostr = sorted(list(combo))
                # print(combostr)
                ab = result_dict[combostr[0]]
                xy = result_dict[combostr[1]]
                r_ab = np.nanmean(ab)
                r_xy = np.nanmean(xy)
                if alt != 'two-sided':
                    if combostr[0] == 'brain': alt='greater'
                    elif combostr[0] == 'EPHATE_5': alt = 'less'
                    else: alt = 'greater'
                # alt='greater'
                p=compare_correlations_r2z(xy, ab, np.nanmean(N), alternative=alt)
                pp = compare_correlation_permutation(xy, ab, nperm, alternative=alt)                

                stat_df.loc[len(stat_df)] = {'contrast':con,
                                             'ROI_name':roi,
                                             'r2z_p':p, 
                                            'name_xy':combostr[1], 
                                             'name_ab':combostr[0],
                                             'xy':r_xy, 
                                             'ab':r_ab, 
                                             'n_per_fold':np.nanmean(N), 
                                            'sorted_comparison':combostr, 
                                             'combostr':f'{combostr[0]}_{combostr[1]}',
                                             'yname':yname, 
                                            'perm_p':pp,
                                            }
    # return stat_df
    indices = {c:stat_df[stat_df['contrast']==c].index for c in stat_df.contrast.unique()}
    uncorrected = {c:extract_results(stat_df, ['contrast'], [c], 'r2z_p') for c in indices.keys()}
    corrected = {c:correct_pvalues_FDR(v)[0] for c,v in uncorrected.items()} 
    symbols = {c:[determine_symbol(i) for i in corrected[c]] for c in corrected.keys()}
    uncorrected_pperm = {c:extract_results(stat_df, ['contrast'], [c], 'perm_p') for c in indices.keys()}
    corrected_pperm = {c:correct_pvalues_FDR(v, fdr=False)[0] for c,v in uncorrected_pperm.items()}
    corrected_p_arr,corrected_pperm_arr = np.zeros(len(stat_df)),np.zeros(len(stat_df))
    symbol_pperm_arr=['' for i in range(len(stat_df))]
    symbol_arr=['' for i in range(len(stat_df))]
    # print(len(stat_df))
    for c in stat_df.contrast.unique():
        idx = list(indices[c])
        # print(idx)
        cor = np.squeeze(corrected[c])
        symb = symbols[c]
        corp=np.squeeze(corrected_pperm[c])
        # print(len(corp))
        corrected_p_arr[idx] = cor
        corrected_pperm_arr[idx] = corp
        for k,j in enumerate(idx):
            symbol_pperm_arr[j] = determine_symbol(corp[k])
            symbol_arr[j]=determine_symbol(cor[k])

    stat_df['corrected_p'] = corrected_p_arr
    stat_df['symbol'] = symbol_arr
    stat_df['corrected_pperm']=corrected_pperm_arr
    stat_df['symbol_pperm'] = symbol_pperm_arr
    return stat_df


def permutation_test(data, n_iterations, alternative='greater'):
    """
    permutation test for comparing the means of two distributions 
    where the samples between the two distributions are paired
    
    """
    
    def less(null_distribution, observed):
        cmps = null_distribution <= observed + gamma
        pvalues = (cmps.sum(axis=0) + 1) / (n_iterations + 1)
        return pvalues

    def greater(null_distribution, observed):
        cmps = null_distribution >= observed - gamma
        pvalues = (cmps.sum(axis=0) + 1) / (n_iterations + 1)
        return pvalues

    def two_sided(null_distribution, observed):
        pvalues_less = less(null_distribution, observed)
        pvalues_greater = greater(null_distribution, observed)
        pvalues = np.minimum(pvalues_less, pvalues_greater) * 2
        return pvalues
    
    compare = {'less': less, 'greater': greater, 'two-sided': two_sided}
    n_samples = data.shape[1]
    observed_difference = data[0] - data[1]
    observed = np.mean(observed_difference)
    eps = 1e-14
    gamma = np.maximum(eps, np.abs(eps * observed))
    null_distribution = np.empty(n_iterations)
    for i in range(n_iterations):
        weights = [choice([-1, 1]) for d in range(n_samples)]
        null_distribution[i] = (weights*observed_difference).mean()
    pvalue = compare[alternative](null_distribution, observed)
    return observed, pvalue, null_distribution

def compare_correlations_r2z(r1, r2, n, alternative='greater'):
    ## assuming that hypothesize r1 > r2
    ab = np.mean(r1)
    xy = np.mean(r2)
    if alternative != 'two-sided':
        t, p = corrstats.independent_corr(xy, ab, n, twotailed=False)
    else:
        t, p = corrstats.independent_corr(xy, ab, n, twotailed=True)
    return p

def compare_correlation_permutation(r1, r2, nperm=1000, alternative='two-sided'):
    ## assumes hypothesis r1 > r2
    pn = np.stack((r1,r2))
    _,p,_ = permutation_test(pn, nperm, alternative)
    return p
    
    
def correct_pvalues_FDR(pvalues, q=0.05, fdr=True):
    if fdr:
        rejection, adjusted_pvalues = multitest.fdrcorrection(pvalues, alpha=q)
    else:
        rejection, adjusted_pvalues, _, _ = multitest.multipletests(pvalues, alpha=q, method='bonferroni')
    return adjusted_pvalues, rejection

def extract_results(dataframe, filter_columns, filter_values, statistic=None):
    ## takes a dataframe
    # filters columns to specific values
    # returns statistics
    indices = []
    for c,v in zip(filter_columns, filter_values):
        idx = list(dataframe.query(f"{c} == '{v}'").index)
        indices.append(idx)
    overlap_indices = reduce(np.intersect1d, indices)
    if statistic:
        return dataframe.iloc[overlap_indices][statistic].values
    return dataframe.iloc[overlap_indices].reset_index() 

def determine_symbol(pval):
    symbols = {'0.1':'~', '0.05':'*', '0.01':'**', '0.001':'***'}
    if pval < 0.001:
        return symbols['0.001']
    if pval < 0.01:
        return symbols['0.01']
    if pval < 0.05:
        return symbols['0.05']
    if pval < 0.1:
        return symbols['0.1']
    return None