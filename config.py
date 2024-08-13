'''
Various path settings and parameters for use across files
ELB 2024
'''

import os

VERBOSE=True

# Set a whole bunch of paths specific to this project & running analyses on milgram cluster
PROJECT_DIR = '/gpfs/milgram/pi/turk-browne/users/elb77/project_abs'
FINAL_REPO = f'{PROJECT_DIR}/manifold_abcd_psychopathology_bpcnni'
FINAL_RESULTS = f'{FINAL_REPO}/results'
FINAL_PLOTS = f'{FINAL_REPO}/plots'
MYDATA_DIR = f'{PROJECT_DIR}/data/neuroimaging_data_aggregated'
RESULTS_DIR_INTERMEDIATE = f'{PROJECT_DIR}/results/revised_may14'
RESULTS_DIR = f'{PROJECT_DIR}/results/'
NDA4 = '/gpfs/milgram/pi/casey/ABCD/ABCDstudyNDA400/'
RELEASE4_PATH = '/gpfs/milgram/pi/casey/ABCD/release4'
SES_STR = 'ses-baselineYear1Arm1/func'
RELEASE4_CIFTI = '/gpfs/milgram/pi/casey/ABCD/release4/cifti/'
SCRATCH_DIR = '/gpfs/milgram/scratch60/casey/elb77/abcd_manifold'


N_COMPONENTS=[5,10,20] # for comparisons
# contrasts of interest
CONTRASTS = ['nBack_2_back_vs_0_back', 'nBack_emotion_vs_neutface']
LABEL_TXT_FN = f'{PROJECT_DIR}/Tian2020MSA_v1.1/3T/Cortex-Subcortex/Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S1_label.txt'
DLABEL_FN = f'{PROJECT_DIR}/Tian2020MSA_v1.1/3T/Cortex-Subcortex/Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S1.dlabel.nii'
REGIONS = ['AMY','HIP','Cont','DorsAttn','SalVentAttn']
# formatting for plots
REGION_NAMES = ['Amygdala', 'Hippocampus', 'Frontoparietal', "Dorsal attention", 'Ventral attention']
AFFMAT_DIR = f'{PROJECT_DIR}/data/affinity_matrices' 
EXCLUDE_IF_MISSING = ['cbcl_scr_syn_totprob_t', 
'crpbi_y_ss_caregiver',
'nsc_p_ss_mean_3_items',
'reshist_addr1_adi_perc',
'fes_y_ss_fc', 
'neighborhood_crime_y',
'tfmri_nb_all_beh_ctotal_rate']

VARIABLE_FILENAMES = {'CBCL': f'{NDA4}/abcd_cbcls01.txt', 
                'ADI': f'{NDA4}/abcd_rhds01.txt', 
                'nihtbx':f'{NDA4}/abcd_tbss01.txt', 
                'imgincl':f'{NDA4}/abcd_imgincl01.txt', 
                'nback_beh':f'{NDA4}/abcd_mrinback02.txt', 
               'SSCEY':f'{NDA4}/abcd_sscey01.txt',
               'NSC':f'{NDA4}/abcd_nsc01.txt',
               'SSCEP':f'{NDA4}/abcd_sscep01.txt',
               'ANT':f'{NDA4}/abcd_ant01.txt',
               'EHIS':f'{NDA4}/abcd_ehis01.txt',
                     'DEMO':f'{NDA4}/pdem02.txt',
                     'FAM':f'{NDA4}/acspsw03.txt',
                     'SITE':f'{NDA4}/abcd_lt01.txt',
                  'SCANNER':f'{NDA4}/abcd_mri01.txt'} 
PREDICTOR_TO_VARIABLE = {'cbcl_scr_syn_anxdep_t':"CBCL", 
                         'cbcl_scr_syn_withdep_t':"CBCL", 
                         'cbcl_scr_syn_somatic_t':"CBCL", 
                         'cbcl_scr_syn_rulebreak_t':"CBCL", 
                         'cbcl_scr_syn_aggressive_t':"CBCL", 
                         "reshist_addr1_adi_perc":"ADI", # Residential history derived - Area Deprivation Index: national percentiles
                         'fes_y_ss_fc':"SSCEY", # Conflict Subscale from the Family Environment Scale Sum of Youth Report
                         'cbcl_scr_syn_totprob_t':"CBCL", 
                         'cbcl_scr_syn_internal_t':"CBCL",
                         'cbcl_scr_syn_external_t':"CBCL",
                         'nihtbx_pattern_uncorrected':"nihtbx",
                         'tfmri_nb_all_beh_ctotal_rate':"nback_beh",
                         'nihtbx_list_uncorrected':'nihtbx',
                         'neighborhood_crime_y':'NSC', # ABCD Youth Neighborhood Safety/Crime Survey (reverse score this) 
                         'nsc_p_ss_mean_3_items':'SSCEP', # Neighborhood Safety Protocol: Mean of Parent Report (reverse score this)
                         'crpbi_y_ss_caregiver':'SSCEY', # CRPBI - Acceptance Subscale Mean of Report by Secondary Caregiver by youth
                        'imgincl_nback_include':'imgincl',
                         'anthroheightcalc':'ANT',
                        'anthroweight1lb':"ANT",
                        'ehi_y_ss_scoreb':"EHIS",
                        'interview_age':'ANT',
                        'race_ethnicity': 'FAM',
                         'demo_comb_income_v2': 'DEMO',
                         'sex': 'DEMO',
                         'demo_prnt_ed_v2': 'DEMO',
                         'site_id_l': 'SITE',
                         'rel_family_id': 'FAM',
                        'mri_info_deviceserialnumber':'SCANNER',
                        }

PREDICTORS_OF_INTEREST = ['cbcl_scr_syn_anxdep_t', 
'cbcl_scr_syn_withdep_t', 
'cbcl_scr_syn_somatic_t', 
'cbcl_scr_syn_rulebreak_t', 
'cbcl_scr_syn_aggressive_t', 
'cbcl_scr_syn_totprob_t', 
'cbcl_scr_syn_internal_t', 
'cbcl_scr_syn_external_t', 
'tfmri_nb_all_beh_ctotal_rate']

TO_REVERSE_SCORE = ['neighborhood_crime_y', 'nsc_p_ss_mean_3_items', 'crpbi_y_ss_caregiver']

FEATURES_PER_MAT = {'control':['interview_age','ehi_y_ss_scoreb','anthroweight1lb','anthroheightcalc'], 
                   'ADI':['reshist_addr1_adi_perc'],
                   'SSCEY':['fes_y_ss_fc'],
                   'ADI_SSCEY':['fes_y_ss_fc','reshist_addr1_adi_perc'],
                   '5':['fes_y_ss_fc', 'reshist_addr1_adi_perc', 'neighborhood_crime_y', 'nsc_p_ss_mean_3_items', 'crpbi_y_ss_caregiver']}

EMBEDDING_DIRECTORIES = {'vanilla_PHATE': f'{SCRATCH_DIR}/vanilla_phate', 
                         'EPHATE_control':f'{SCRATCH_DIR}/ephate_control', 
                         'EPHATE_ADI':f'{SCRATCH_DIR}/ephate_adi',
                        'EPHATE_SSCEY':f'{SCRATCH_DIR}/ephate_sscey',
                        'PHATE_PLUS_FEATURES':f'{SCRATCH_DIR}/phate_plus_features',
                         'EPHATE_ADI_SSCEY':f'{SCRATCH_DIR}/ephate_adi_sscey',
                        'EPHATE_5':f'{SCRATCH_DIR}/ephate_5',
                        'PCA':f'{SCRATCH_DIR}/pca_embedding',
                        'UMAP':f'{SCRATCH_DIR}/umap_embedding'}

AFFINITY_MATRICES = {'control':f'{AFFMAT_DIR}/control_baseline_affinity_mat.csv', 
                    'ADI':f'{AFFMAT_DIR}/ADI_baseline_affinity_mat.csv', 
                    'ADI_SSCEY':f'{AFFMAT_DIR}/ADI_SSCEY_baseline_affinity_mat.csv',
                    '5':f'{AFFMAT_DIR}/5targets_baseline_affinity_mat.csv',
                    'SSCEY':f'{AFFMAT_DIR}/SSCEY_baseline_affinity_mat.csv'}

YEAR_CODES = {'baseline':'baseline_year_1_arm_1', '2year':'2_year_follow_up_y_arm_1'}





