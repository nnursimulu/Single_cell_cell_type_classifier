# Details of scripts for single-cell cell-type classifier

Run these scripts only after preprocessing the raw data.

Retain the values of ``RAW_DATA``--the path to the folder containing ``count_matrix.in`` and ``class_labels.in``--and the absolute path to the folder ``PROCESSED_DATA`` where the split data has been written for cross-validation and testing, as these are typically used during the scripts below.  Find parameters required to run each script using ``python3 script_name.py --help``.  For ease of understanding, define ``RESULTS`` as the path to the folder to contain results from the scripts.  Note that ``utils.py`` contains functions used frequently by the different scripts.

## Cross-validation for various classifiers

- k-nearest neighbour: Run ``cv_kNN.py`` with correct parameters. Produces ``$RESULTS/CV_knn.out``.
- random forest: Run ``cv_RF.py`` with correct parameters.  Produces ``$RESULTS/CV_RF_balanced.out`` and ``$RESULTS/CV_RF_None.out`` (results for balanced and not balanced RF)
- linear support machine: Optimized for HPC (SLURM).  Modify ``parallel_SVM_balanced_hpc.sh`` and ``parallel_SVM_not_balanced_hpc.sh`` and submit.  Each will run in parallel threads ``cv_SVM_hpc.py``, which trains SVM on one section of the data for one particular parameter.  When these processes done, ``summarize_accuracy_hpc.py`` will output the summarized value of performance metrics from ``CV_SVM_balanced_HPC.out`` or ``CV_SVM_not_balanced_HPC.out`` as ``CV_SVM_balanced_averaged.out`` or ``CV_SVM_not_balanced_averaged.out``
- neural network: Optimized for HPC (SLURM).  Modify ``parallel_NN_hpc.sh`` and submit.  Runs ``cv_NN_hpc.py`` in parallel threads, produces ``CV_NN_HPC.out`` and summarizes results in ``CV_NN_averaged.out`` using ``summarize_accuracy_hpc.py``.

## Visualize loss on optimal neural network

Run ``visualize_loss_best_NN.py`` to look at how training and validation loss evolve over epochs during cross-validation.  I used this script to determine the best epoch at which to stop learning.  I had to run this script on my computer to allow for graphic display, not supported by default by Niagara (using Python v3.10.6, tensorflow v2.13.0, matplotlib v3.7.2, numpy v1.24.2, sklearn v1.3.0).  In future, I will re-write these scripts to make sure they can be run on the same platform.

## Test on held-out test set

I wrote ``run_optimal_classifiers.py`` from hyperparameters from cross-validation that gave the highest macro-averaged F1 score over the entire validation set.  Running this script produces files of the form ``OVERALL_classifier_name.out`` in ``$RESULTS`` giving the accuracy, the macro-F1 score, the precision and recall by class, and a confusion matrix detailing the classes corresponding to mispredictions.  Raw predictions of NN, RF-balanced and kNN are written to files of the form ``$RESULTS/ALL_PREDs_on_test_classifier.out``.  These can be ignored as they are only considered as part of future work (see below).

## Analysis: ablation experiment with neural network

``run_ablation_NN.py`` trains alternate NN architecture on training data, and produces files of the form ``$RESULTS/ABLATION_NN_expt_name.out`` with details of accuracy, macro-f1, class-specific performance and confusion matrix.

## Analysis: Calculate feature importance of NN

``calculate_feature_importance_NN.py`` finds which genes are importance for predictions on which class and produces ``$RESULTS/FEATURE_IMPORTANCE_gene_by_class.out``, a copy of which is in the ``suppl`` directory in repository's main directory.

## Future work: Majority rule on combined classifiers

I started working on ``majority_rule_on_combined_classifiers.py``, an experimental ensemble classifier which takes the predictions by kNN and SVM if they agree with each other, otherwise favouring the NN predictions.  I however found that performance, while higher than kNN, was lower than NN's.  This is a result that is rather intriguing that I wish to investigate later.  

Perhaps, there are other ways of combining predictions between classifiers that work best?  Perhaps there are other classifiers that I should consider?  I did not use SVM here in the interest of time.  Perhaps, SVM would give additional insight as an expert on certain classes?


