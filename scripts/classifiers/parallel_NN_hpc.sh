#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --job-name=NN_cv

# Runs cross-validation for neural net.
module load conda3

MAIN_PROJECT=/path/to/Single_cell_cell_type_classifier # Specify path to main repo folder.

TRAINING_FOLDER=${MAIN_PROJECT}/processed_data # Where the training data is.
RESULTS_FILE=${MAIN_PROJECT}/results/CV_NN_HPC.out # Where the results will be written
WORKDIR=${MAIN_PROJECT}/scripts/classifiers # Directory containing this script.
SUMMARIZED_RESULTS=${MAIN_PROJECT}/results/CV_NN_averaged.out # Where the averaged results will be written

cd $WORKDIR

for fold_i in 0 1 2 3 4; do
	for dropout_rate in 0.2 0.5 0.8; do
		for nn_setting in 100_50_25 100_50 100 50_25 50 25; do
			python3 cv_NN_hpc.py --training_folder $TRAINING_FOLDER --results_file $RESULTS_FILE --nn_setting $nn_setting --dropout_rate $dropout_rate --subsection_of_data $fold_i &
		done
	done
done

wait;

python3 summarize_accuracy_hpc.py --input_file $RESULTS_FILE --output_file $SUMMARIZED_RESULTS