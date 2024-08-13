'''
run_longitudinal_regression.py

This script is used to run a cross-validated linear regression between a given representation of brain data (voxel-wise data or embedding of some sort)
with a variable of interest, as collected at the 2-year follow up. 
It scores the regression with spearman's correlation or a partial correlation, to control for other things.

For all analyses presented in paper, we use $NDIM=20

runs from command line as:
`python run_longitudinal_regression.py -r $ROI_NAME -c $CONTRAST_NAME -ndim $NDIM`

ELB 2024
'''
import numpy as np
import pandas as pd
import os, sys, glob
import utils, argparse
import scipy.stats 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import  KFold 
from joblib import Parallel,delayed
import warnings
import numpy as np
import pingouin as pg
from config import *
warnings.filterwarnings('ignore')

def select_subjects(event, contrast):
	df = pd.read_csv(f'{utils.project_dir}/data/subject_nback_features_include_{event}.csv',index_col=0)
	df = df[df['present']==1]
	return df.index

def run_regression(X, y, covars, n_folds):
	outer_folder = KFold(n_splits=n_folds)

	#outer_folder = LeaveOneOut()
	results_df = pd.DataFrame(columns = ['test_spearman_r','test_spearman_p', 'partial_corr_rho', 'partial_corr_p', 'partial_corr_ci', 'fold', 'n_test'])

	# mask nans again
	if np.sum(y!=y) > 0:
		idx = np.where(y==y)
		y=y[idx]
		X=X[idx]
		covars=covars[idx]
		print(f'masked again; {y.shape}, {X.shape}')
	dataframe = pd.DataFrame({'true_y':y, 'predicted_y':np.zeros_like(y)})
	for icov in range(covars.shape[0]):
		dataframe[f'covariate_{icov:02d}'] = covars[icov]
	predicted_y = np.zeros_like(y)
	covariate_columns=[f'covariate_{icov:02d}' for i in range(covars.shape[0])]
	y = y.reshape(-1,1)
	# now do one with all the best parameters
	for f, (train_idx, test_idx) in enumerate(outer_folder.split(np.arange(len(X)))):
		regression = LinearRegression()
		X_train, y_train = X[train_idx], np.nan_to_num(y[train_idx])
		X_test, y_test = X[test_idx], np.nan_to_num(y[test_idx])
		clf = regression.fit(X_train, y_train)
		y_pred = np.squeeze(clf.predict(X_test))
		y_true = np.squeeze(y_test)
		spearman = scipy.stats.spearmanr(y_pred, y_true)
		spearman_r, spearman_p = spearman.correlation, spearman.pvalue
		temp = pd.DataFrame({'y_pred':y_pred, 'y_true': y_true})
		for icov in range(covars.shape[0]):
			temp[f'covariate_{icov:02d}'] = covars[icov, test_idx]
		r=pg.partial_corr(data=temp, x='y_pred', y='y_true', covar=covariate_columns, method='spearman')
		predicted_y[test_idx] =y_pred
		results_df.loc[len(results_df)] = {
											'test_spearman_r':spearman_r, 
											'test_spearman_p':spearman_p,
											'partial_corr_rho':r.iloc[0]['r'], 
											'partial_corr_p':r.iloc[0]['p-val'],
											'partial_corr_ci':r.iloc[0]['CI95%'],
											'n_test':len(y_true),
											'fold':f}                                      
	return results_df

def load_env_only(eventname):
	sub_df = utils.get_included_subject_dataframe()
	sublist = sub_df['subjectkey'].values
	feature_names = FEATURES_PER_MAT["5"]
	variable_names = [PREDICTOR_TO_VARIABLE[f] for f in feature_names]        
	F = []
	is_valid = np.ones_like(sublist)
	for f,v in zip(feature_names,variable_names):
		f = utils.load_variables(sublist, eventname, [f], v, check_reverse=True)
		F.append(f)
		for c in range(len(f)):
			if f[c] != f[c]:
				is_valid[c] = 0
	idx_valid = np.where(is_valid==1)[0]
	data = np.array(F).T
	sublist = [sublist[i] for i in idx_valid]
	data = scipy.stats.zscore(data[idx_valid],axis=0)
	return data,sublist

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-r','--ROI_name')
	parser.add_argument('-c','--contrast')
	parser.add_argument('-b', '--embedding_type')
	parser.add_argument('-n','--ndim',default=20, type=int)
	parser.add_argument('-k', '--n_folds', default=20, type=int)
	parser.add_argument('-v','--verbose',default=True)
	parser.add_argument('-o','--overwrite', default=True)
	p = parser.parse_args()

	outdir = RESULTS_DIR_INTERMEDIATE
	covariate='mri_info_deviceserialnumber'
	v0=0
	
	if p.embedding_type != 'brain':
		R = f'{outdir}/longitudinal_{p.contrast}_{p.ROI_name}_{p.embedding_type}_{p.ndim}d_LinearRegression_v{v0}.csv'
	else:
		R = f'{outdir}/longitudinal_{p.contrast}_{p.ROI_name}_brain_LinearRegression_v{v0}.csv'

	if os.path.exists(R) and not p.overwrite: print(f"Already ran {R}; returning"); sys.exit(0)

	# load in the embedding file and subject list
	if p.embedding_type != 'brain':
		sublist =  utils.select_brain_data_subjects('longitudinal', p.contrast)
		data = utils.load_embedding('longitudinal', p.contrast, p.ROI_name, p.embedding_type, p.ndim)
		assert len(data) == len(sublist)
	else: 
		data, sublist = utils.load_brain_data([], 'longitudinal', p.contrast, p.ROI_name)
		data = np.nan_to_num(data)


	if p.verbose: print(f"running longitudinal, contrast={p.contrast}, {p.embedding_type}, ROI={p.ROI_name}")
	if p.verbose: print(f"loaded data of shape {data.shape}; loaded sublist of len {len(sublist)}")
	SCANNER_ID = utils.load_variables(sublist, 'baseline', 'mri_info_deviceserialnumber', 'SCANNER', check_reverse=False)
	# map to numeric
	uniq_ids = np.unique(SCANNER_ID)
	mapping = {uniq_ids[i]:i for i in range(len(uniq_ids))}
	numeric_serials = [mapping[hashed] for hashed in SCANNER_ID]
	SCANNER_ID_NUM=np.array(numeric_serials)

	result_lists = []
	for to_predict in PREDICTORS_OF_INTEREST:
		header_file = PREDICTOR_TO_VARIABLE[to_predict]
		y_now = scipy.stats.zscore(utils.load_variables(sublist, '2year', [to_predict], header_file))
		y_baseline = scipy.stats.zscore(utils.load_variables(sublist, 'baseline', [to_predict], header_file))
        
		if len(y_now) != data.shape[0]:  
			print(f'labels and data are mismatched: y={y_now.shape}, date={data.shape}; skipping');  
			continue
		# create covariate matrix
		COVAR_MAT = np.array([SCANNER_ID_NUM, y_baseline])
		if COVAR_MAT.shape[0]!=2: COVAR_MAT=COVAR_MAT.T

		reg_results = run_regression(data, y_now, COVAR_MAT, p.n_folds)
		reg_results['predictor'] = to_predict
		reg_results['n_dimensions'] = data.shape[1]
		reg_results['embedding_type'] = p.embedding_type
		reg_results['ROI_name'] = p.ROI_name
		reg_results['contrast']=p.contrast
		reg_results['covariates']=f'scanner_id, {to_predict}_baseline'
		if p.verbose: 
			g = reg_results['partial_corr_rho'].values.mean()
			rho = reg_results['test_spearman_r'].values.mean()
			print(f'finished {to_predict}; mean pcor={g}, rho={rho}')
		result_lists.append(reg_results)
	result_lists=pd.concat(result_lists).reset_index(drop=True)
	result_lists.to_csv(R)
	if p.verbose: print(f"saved to {R}")















