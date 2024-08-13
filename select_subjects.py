'''
Select subjects with:
- baseline nBack contrasts
- ADI
- CBCL
- SSCEY
- NSC
- SSCEP
- imgincl==1
include additional columns for:
- has 2year nBack contrasts
- has 2year CBCL
- baseline 
	- ANT
	- EHIS 
    
ELB 1/8/24
'''
import numpy as np 
import utils
from config import *
import pandas as pd
import os, sys, glob

# get subjects with contrasts
subjects_by_event = {}
for event in ['baseline','2year']:
	dirnames = glob.glob(f'{RELEASE4_CIFTI}/*_{event}/')
	subjects = []
	for dn in dirnames:
		fns = glob.glob(f'{dn}/nBack/*')
		if len(fns) > 0:
			s = dn.split(RELEASE4_CIFTI)[-1].split(f'_{event}')[0]
			subjects.append(s)
	subjects_by_event[event]=subjects

baseline_subjects = subjects_by_event['baseline']
subject_df = pd.DataFrame({'subjectkey':baseline_subjects})

# grab values for these subjects for the ones if missing will exclude
for FEAT in EXCLUDE_IF_MISSING:
	VAR = PREDICTOR_TO_VARIABLE[FEAT]
	values = utils.load_variables(baseline_subjects, 'baseline', [FEAT], VAR, check_reverse=True)
	subject_df[f'{FEAT}_baseline'] = values

# now check img incl
values = utils.load_variables(baseline_subjects, 'baseline', ['imgincl_nback_include'], 'imgincl', check_reverse=True)
if VERBOSE: print(f'IMGINCL is missing {np.sum(values == 0)} values')
# replace 0 with nan for ease of filtering
values = np.where(values==0, np.nan, values)
subject_df[f'imgincl_nback_include_baseline'] = values

# if there are nans in a subject's row, we toss them
nan_rows = subject_df.isna().any(axis=1)
# how many nans?
nnans = np.sum(nan_rows)
if VERBOSE: print(f'of {len(subject_df)} baseline subjects, {nnans} have nans')

# drop out the nans
subject_df = subject_df.dropna().reset_index(drop=True)
baseline_subjects = subject_df['subjectkey'].values
if VERBOSE: print(f'now df has {baseline_subjects.shape} subjects')
columns = list(subject_df.columns)

# now get remaining values
for P,V in PREDICTOR_TO_VARIABLE.items():
	colstr = f'{P}_baseline'
	if colstr in columns:
		continue
	vals = utils.load_variables(baseline_subjects, 'baseline', [P], V, check_reverse=True)
	subject_df[colstr] = vals
	if VERBOSE: print(f'finished {colstr}')
	columns.append(colstr)

# now get the couple of things that we may want 2 year
# refilter 2 year subjects only in baseline
year2_subjects = [s for s in subjects_by_event['2year'] if s in baseline_subjects]
y=len(subjects_by_event['2year'])

if VERBOSE: print(f'after exclusions at baseline, {len(year2_subjects)}/{y} remain @ 2year')
# get values at 2year

wantyear2 = ['cbcl_scr_syn_totprob_t', 'cbcl_scr_syn_internal_t', 'cbcl_scr_syn_external_t']
temp = pd.DataFrame({'subjectkey':year2_subjects})
for w in wantyear2:
	colstr = f'{w}_2year'
	vals = utils.load_variables(year2_subjects, '2year', [w], 'CBCL', check_reverse=True)
	temp[colstr]=vals

# find missing
nan_rows = temp.isna().any(axis=1)
nnans = np.sum(nan_rows)
if VERBOSE: print(f'of {len(temp)} year2 subjects, {nnans} have nans')
nan_df = nan_rows.to_frame()
nan_df.index.name ='index'
nan_df=nan_df.rename(columns={0:'include'})
nan_df = nan_df.apply(lambda x: x.apply(lambda y: 0 if y else 1))
nan_df['subjectkey'] = year2_subjects

include_year2 = []
for b in baseline_subjects:
    if b not in year2_subjects:
        include_year2.append(0)
    elif nan_df[nan_df['subjectkey']==b]['include'].item()==1:
        include_year2.append(1)
    else:
        include_year2.append(0)

subject_df['include_year2']=include_year2
for w in wantyear2:
	colstr = f'{w}_2year'
	vals = utils.load_variables(baseline_subjects, '2year', [w], 'CBCL', check_reverse=True)
	subject_df[colstr]=vals


# save
subject_df.to_csv(f'{PROJECT_DIR}/data/subject_inclusion_step1.csv')

