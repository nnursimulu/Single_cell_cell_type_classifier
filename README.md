# Single-cell cell-type classifier

The data used to build the single-cell cell-type classifiers comes from https://github.com/10XGenomics/single-cell-3prime-paper/tree/master/pbmc68k_analysis.  Running scripts in this repository requires R, a Unix shell and Python3.  I have run this code using Python v3.6.8.

## Raw data

The overall data consists of the 1000 most highly-expressed genes from the 68K PBMC dataset.  Use the scripts from the folder scripts/preprocess to regenerate this data.  More specifically, first, run main_process_68k_pbmc.R from https://github.com/10XGenomics/single-cell-3prime-paper/tree/master/pbmc68k_analysis, then, run the R script transform_data_from_r_to_python.R found in this folder.  The output count_matrix.in and class_labels.in are used for training and testing the classifiers in this project.

## Data splitting

Run scripts/preprocess/split_into_train_and_test.py to split the count matrix and labels into training, validation and test sets.  Specify the absolute path RAW_DATA of the input folder--the one containing count_matrix.in and class_labels.in--and the absolute path to the output folder PROCESSED_DATA where the split data will be written.

split_into_train_and_test.py --input_folder RAW_DATA --output_folder PROCESSED_DATA  

## Dependencies

In addition to the dependencies required to run main_process_68k_pbmc.R, the following are required.

The following Python modules: scipy, numpy, sklearn
