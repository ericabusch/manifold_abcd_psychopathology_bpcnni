## Manifold learning uncovers nonlinear interactions between the adolescent brain and environment that predict emotional and behavioral problems

## Analysis Scripts
This directory contains all analysis scripts used for embeddings, benchmarking, and predictions in the above manuscript. It also includes code for selecting participants and plotting results and running statistics. 
We used the script `run_exogenous_phate.py` to get E-PHATE embeddings for the analyses reported in this paper. Since then,  we have released [EPHATE](https://github.com/ericabusch/EPHATE) as a standalone package and have shown that it replicates the embeddings presented here.

For more information, check out our [paper](https://doi.org/10.1016/j.bpsc.2024.07.001), in press in Biological Psychiatry: Cognitive Neuroscience and Neuroimaging.

## Scripts & Functions
1. `select_subjects.py` : goes through ABCD subject list and selects subjects with all data present (baseline nBack contrasts, ADI, CBCL, SSCEY, NSC, SSCEP, imgincl==1, 2 year CBCL)
2. `inclusion.ipynb`: notebook outlining in more detail the inclusison/exclusion criteria, with print outs at each step. After exclusions for missing files, excludes based on:
   * missing more values in CIFTI than just the medial wall (>260 missing values)
   * mean activation outside +/- 3 times the standard deviation of the group mean activation
   * mean standard deviation of activation outside +/- 3 times the standard deviation of the group mean standard deviation activation
   * selects one subject per family
3. `run_region_data_extraction.py` : masks whole-brain CIFTIs into ROI data matrices, stacking subjects together - eventual output shape = (n_subjects , n_voxels)
4. `run_exogenous_phate.py` : runs EPHATE on ROI data matrices
5. `run_vanilla_phate.py` : runs PHATE on ROI data matrices
6. `run_benchmark_embeddings.py` : runs PCA and UMAP on ROI data matrices
7. `run_longitudinal_embeddings.py` : takes baseline brain data for subjects who have longitudinal behavioral data and re-runs ephate, vanilla phate, and benchmarks for them
8. `run_phate_plus_features.py` : runs additional variants of including env data into embeddings
9. `run_cross_sectional_regression.py` : predicts baseline behavioral/mental health scores from baseline brain / embedding data
10. `run_longitudinal_regression.py` : predicts 2-year behavioral/mental health scores from baseline brain / embedding data
