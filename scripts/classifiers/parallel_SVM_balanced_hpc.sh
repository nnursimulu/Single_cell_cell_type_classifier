#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --job-name=SVM_balanced

# Runs cross-validation for balanced SVM.
module load conda3

MAIN_PROJECT=/path/to/Single_cell_cell_type_classifier # Specify path to main repo folder.

TRAINING_FOLDER=${MAIN_PROJECT}/processed_data # Where the training data is.
RESULTS_FILE=${MAIN_PROJECT}/results/CV_SVM_balanced_HPC.out # Where the results will be written
WORKDIR=${MAIN_PROJECT}/scripts/classifiers # Directory containing this script.
SUMMARIZED_RESULTS=${MAIN_PROJECT}/results/CV_SVM_balanced_averaged.out # Where the averaged results will be written

cd $WORKDIR

is_balanced=1
for fold_i in 0 1 2 3 4; do
	for c_value in 0.01 0.1 1 10 100; do
		for penalty in 0 1; do
			python3 cv_SVM_hpc.py --training_folder $TRAINING_FOLDER --results_file $RESULTS_FILE --c_value $c_value --is_l1 $penalty --subsection_of_data $fold_i --is_balanced $is_balanced &
		done
	done
done

wait;

python3 summarize_accuracy_hpc.py --input_file $RESULTS_FILE --output_file $SUMMARIZED_RESULTS