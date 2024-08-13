'''
create a crazy number of dsq joblists
'''
import os, sys, glob
from config import *

# # make region extraction jobs
# string = f'python -u {PROJECT_DIR}/run_region_data_extraction.py'
# with open('dsq_files/joblist_region_extract.txt','w') as f:
#         for C in CONTRASTS:
#             for E in YEAR_CODES.keys():
#                 f.write(f'{string} -c {C} -e {E}\n')

# # make feature phate jobs
# string = f'python -u {PROJECT_DIR}/run_feature_phate.py'
# with open('dsq_files/joblist_fphate_embeddings.txt','w') as f:
#     for F in FEATURES_PER_MAT.keys():
#         for C in CONTRASTS:
#             for R in REGIONS:
#                 f.write(f'{string} -r {R} -f {F} -c {C} -e baseline\n')

# # make vanilla phate jobs
# string = f'python -u {PROJECT_DIR}/run_vanilla_phate.py'
# with open('dsq_files/joblist_vanilla_phate_embeddings.txt','w') as f:
#     for C in CONTRASTS:
#         for R in REGIONS:
#             f.write(f'{string} -r {R} -c {C} -e baseline\n')

# # make regression brain jobs
# string = f'python -u {PROJECT_DIR}/run_prediction.py -e baseline -b brain'
# with open('dsq_files/joblist_regression_brain.txt', 'w') as f:
#     for C in CONTRASTS:
#         for R in REGIONS:
#             for P in PREDICTORS_OF_INTEREST:
#                 f.write(f'{string} -r {R} -c {C} -p {P}\n')

# # make regression embeddings
# string = f'python -u {PROJECT_DIR}/run_cross_section_regression.py -e baseline'
# with open('dsq_files/joblist_regression_embeddings.txt', 'w') as f:
#     E='env_only'
#     f.write(f'{string} -r None -c None -n 1 -b {E}\n')
#     for C in CONTRASTS:
#         for R in REGIONS:
#             f.write(f'{string} -r {R} -c {C} -n 1 -b brain\n')
#     for C in CONTRASTS:
#         for R in REGIONS:
#             for E in EMBEDDING_DIRECTORIES.keys():
#                 for N in N_COMPONENTS:
#                 	f.write(f'{string} -r {R} -c {C} -n {N} -b {E}\n')
            

# # make year2year embeddings
string = f'python -u {PROJECT_DIR}/run_longitudinal_embeddings.py'
with open('dsq_files/joblist_longitudinal_embeddings.txt', 'w') as f:
    for C in CONTRASTS:
        for R in REGIONS:
            for E in EMBEDDING_DIRECTORIES.keys():
                print(E)
                if E == 'CCA': continue
                f.write(f'{string} -r {R} -c {C} -b {E}\n')


#     # # year2year prediction
# string = f'python -u {PROJECT_DIR}/run_longitudinal_regression.py'
# with open('dsq_files/joblist_temporal_prediction_embeddings.txt', 'w') as f:
#     E='env_only'
#     f.write(f'{string} -r None -c None -n 1 -b {E}\n')
#     for C in CONTRASTS:
#         for R in REGIONS:
#             f.write(f'{string} -r {R} -c {C} -n 1 -b brain\n')
#     for C in CONTRASTS:
#         for R in REGIONS:
#             for E in EMBEDDING_DIRECTORIES.keys():
#                 for N in N_COMPONENTS:
#                     f.write(f'{string} -r {R} -c {C} -n {N} -b {E}\n')

# # year2year prediction
# string = f'python -u {PROJECT_DIR}/run_temporal_prediction.py'
# with open('dsq_files/joblist_temporal_prediction_brain.txt', 'w') as f:
#     for C in CONTRASTS:
#         for R in REGIONS:
#     		for P in PREDICTORS_OF_INTEREST:
#         		f.write(f'{string} -r {R} -c {C} -p {P} -b brain\n')





