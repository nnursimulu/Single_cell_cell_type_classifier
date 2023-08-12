# Script to summarize accuracy over multiple parameters and over different data subsections.
# Meant to process output from high-performance computing.
from argparse import ArgumentParser
import utils

if __name__ == '__main__':

    parser = ArgumentParser(description="Reads cross-validation output from high-performance computing and summarizes it.")
    parser.add_argument("--input_file", type=str, help="Absolute path to file that contains cross-validation results",
                        required=True)
    parser.add_argument("--output_file", type=str, help="Absolute path to file that will contain summarized results",
                        required=True)

    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file

    # First read cross-validation results
    dict_k_to_section_to_accuracy = {}
    with open(input_file) as reader:
        for line in reader:
            line = line.strip()
            if line == "":
                continue
            fold_i, setting, accuracy = utils.parse_out_cv_accuracy_for_data_subsection(line)
            utils.update_accuracy_dict(dict_k_to_section_to_accuracy, setting, fold_i, accuracy)

    dict_k_to_accuracy = utils.summarize_accuracy_across_hyperparameters(dict_k_to_section_to_accuracy)
    utils.write_out_summarized_accuracy(dict_k_to_accuracy, output_file)