# Single-cell cell-type classifier

Code and data in this repository is for building and testing cell type classifiers based on the PBMC68K dataset.  The data used to build the single-cell cell-type classifiers comes from https://github.com/10XGenomics/single-cell-3prime-paper/tree/master/pbmc68k_analysis and concerns the publication by Zheng et al, 2017.  Running scripts in this repository requires R, a Unix shell and Python3.  I have run this code using Python v3.6.8 and using the Niagara supercomputer.  The section dependencies hightlights the dependencies required and the specific version with which the scripts were run.  Some scripts are written in a manner suited for parallel computing, when intense resources were found to be required.

Follow the following steps to reproduce the results I obtained.

## Reproduce results of technical report.
### 1. Obtain raw data

The overall data consists of the 1000 most highly-expressed genes from the 68K PBMC dataset.  Use the scripts from the folder scripts/preprocess to regenerate this data.  More specifically, first, run ``main_process_68k_pbmc.R`` from https://github.com/10XGenomics/single-cell-3prime-paper/tree/master/pbmc68k_analysis, then, run the R script ``transform_data_from_r_to_python.R`` found under scripts/preprocess folder.  The output ``count_matrix.in`` and ``class_labels.in`` generated are used for training and testing the classifiers in this project.  ``ordered_gene_names.in`` is used for data interpretation, and contains gene names by decreasing order of variability (per the procedure by Zheng et al, 2017).

### 2. Perform data splitting

Run ``scripts/preprocess/split_into_train_and_test.py`` to split the count matrix and labels into training, validation and test sets.  Specify the absolute path ``RAW_DATA`` of the input folder--the one containing ``count_matrix.in`` and ``class_labels.in``--and the absolute path to the output folder ``PROCESSED_DATA`` where the split data will be written.

```
split_into_train_and_test.py --input_folder RAW_DATA --output_folder PROCESSED_DATA  
````

### 3. 

## Additional data

My technical report refers to results of feature importance where I build feature vectors with a 1 for a feature/gene and 0 elsewhere, and assess the class predicted by the neural network classifier.  The file ``suppl/FEATURE_IMPORTANCE_gene_by_class.out`` contains the class ID predicted and the features/genes sorted in decreasing order of probability.  Therefore, the top entry indicates the gene most predictive of the class in question.  These results are preliminary and will require more systematic analyses before interpretation. 

## Dependencies

In addition to the dependencies required to run ``main_process_68k_pbmc.R``, the following Python modules are required (version used to run indicated).

- scipy (1.2.1)
- numpy (1.19.5)
- sklearn (0.20.3)
- tensorflow (2.4.1)
- matplotlib (3.0.3)

Tensorflow in my case was run without GPU, which however did not appear to affect performance in terms of speed.

## References

Ponce, M. et al., Deploying a Top-100 Supercomputer for Large Parallel Workloads: the Niagara Supercomputer. PEARC'19 Proceedings, 2019.

Zheng, G. X. Y. et al., Massively parallel digital transcriptional profiling of single cells. Nat Commun, 2017. 8:14049