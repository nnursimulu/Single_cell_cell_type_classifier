#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --job-name=SVM_balanced

# Runs cross-validation for balanced SVM.
#module load conda3

MAIN_PROJECT=/Users/nirvana/Documents/CV/Job_applications/UHN_Bioinformatics_analyst/Technical_interview/Single_cell_cell_type_classifier

TRAINING_FOLDER=${MAIN_PROJECT}/processed_data # Where the training data is.
RESULTS_FILE=${MAIN_PROJECT}/results/CV_NN_HPC.out # Where the results will be written
WORKDIR=${MAIN_PROJECT}/scripts/classifiers # Directory containing this script.
SUMMARIZED_RESULTS=${MAIN_PROJECT}/results/CV_NN_averaged.out # Where the averaged results will be written

cd $WORKDIR

is_balanced=1
for fold_i in 0 1 2 3 4; do
	for dropout_rate in 0.2 0.5 0.8; do
		for nn_setting in 100_50_25 100_50 100 50_25 50 25; do
			python3 cv_NN_hpc.py --training_folder $TRAINING_FOLDER --results_file $RESULTS_FILE --nn_setting $nn_setting --dropout_rate $dropout_rate --subsection_of_data $fold_i
		done
	done
done

wait;

python3 summarize_accuracy_hpc.py --input_file $RESULTS_FILE --output_file $SUMMARIZED_RESULTS