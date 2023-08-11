import os
import scipy
import numpy as np
from argparse import ArgumentParser
from sklearn.model_selection import StratifiedKFold
import pickle



if __name__ == '__main__':
    
    parser = ArgumentParser(description="Splits the scRNA count matrix and corresponding labels into training," + \
        "validation and test data set.")
    parser.add_argument("--input_folder", type=str, help="Folder containing class_labels.in and count_matrix.in.",
                        required=True)
    parser.add_argument("--output_folder", type=str, help="Folder to contain split data.",
                        required=True)

    args = parser.parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder

    # Read feature vectors and labels.  Text labels will be associated with IDs for easier manipulation via sklearn.
    count_matrix = scipy.io.mmread(input_folder + "/count_matrix.in").tocsr()
    feature_vectors = convert_to_fvs(count_matrix)
    ordered_ids, id_to_label = read_labels(input_folder + "/class_labels.in")
    
    # Write out the number of cells in each category
    print ("========Distribution of raw data:")
    print_freq_ids(ordered_ids, id_to_label)

    # Split into train and test set.  Have training be 80% and test 20% of the data.
    # Set a seed for reproducibility.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=999)
    train_indices, test_indices = next(skf.split(feature_vectors, ordered_ids))
    train_fvs, train_label_ids = get_fvs_from_indices(feature_vectors, ordered_ids, train_indices)
    test_fvs, test_label_ids = get_fvs_from_indices(feature_vectors, ordered_ids, test_indices)

    # Create the output directory if it does not exist yet.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Write out the train and test feature vectors (as pickled data structures).
    # Also pickle the ID to label dictionary for easy access when learning.
    pickle_fvs_and_labels(train_fvs, train_label_ids, "OVERALL_training", output_folder)
    pickle_fvs_and_labels(test_fvs, test_label_ids, "OVERALL_test", output_folder)
    pickle_object(id_to_label, output_folder + "/id_to_label.pkl")

    # Split training data for 5-fold cross-validation.
    skf_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=999)
    for i, (train_indices_cv, test_indices_cv) in enumerate(skf_cv.split(train_fvs, train_label_ids)):
        train_fvs_cv, train_label_ids_cv = get_fvs_from_indices(train_fvs, train_label_ids, train_indices_cv)
        test_fvs_cv, test_label_ids_cv = get_fvs_from_indices(train_fvs, train_label_ids, test_indices_cv)

        # Also write out train and test feature vectors, again as pickled data structures.
        pickle_fvs_and_labels(train_fvs_cv, train_label_ids_cv, "CV_" + str(i) + "_training", output_folder)
        pickle_fvs_and_labels(test_fvs_cv, test_label_ids_cv, "CV_" + str(i) + "_test", output_folder)

    print ("===========Data splitting done.")