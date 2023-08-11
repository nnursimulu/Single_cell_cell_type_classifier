import os
import scipy
import numpy as np
from argparse import ArgumentParser
from sklearn.model_selection import StratifiedKFold
import pickle


def convert_to_fvs(count_matrix):
    """Given m by n count matrix (m cells, n genes), return m feature vectors.
    
    Parameters:
    count_matrix: (ndarray) m by n matrix.
    
    Return:
    m feature vectors as ndarray
    """

    fvs = []
    (n_cells, n_genes) = count_matrix.get_shape()
    i = 0
    while i < n_cells:
        fv = count_matrix[i, :]
        fvs.append(fv)
        i += 1
    return np.array(fvs)


def read_labels(filename):
    """Read labels from file and return the ids--as number--as id to label dictionary.
    
    Parameters: filename containing labels, type str
    
    Return:
    ids: array of numbers (int array)
    id_to_label: dict with key id (int) and value label (str)
    """

    ids = []
    id_to_label = {}
    label_to_id = {}

    latest_id = -1
    with open(filename) as reader:
        for label in reader:
            label = label.strip()
            if label == "":
                continue
            if label in label_to_id:
                curr_id = label_to_id[label]
            else:
                latest_id += 1
                curr_id = latest_id
                id_to_label[latest_id] = label
                label_to_id[label] = latest_id
            ids.append(curr_id)
    return ids, id_to_label


def print_freq_ids(ids, id_to_label):
    """Purely for observation: print the frequency of each ID/label.
    
    Parameters:
    ids: list of int
    id_to_label: dict of int to str (id to label)
    """

    id_to_count = {}
    for curr_id in ids:
        if curr_id not in id_to_count:
            id_to_count[curr_id] = 1
        else:
            id_to_count[curr_id] += 1
    print ("ID\tcount")
    for curr_id, count in id_to_count.items():
        print (id_to_label[curr_id] + "\t" + str(count))


def get_fvs_from_indices(all_fvs, all_ids, indices_of_int):
    """Given all feature vectors, all associated IDs, the indices of interest, 
    and a dictionary of id to label, return the features of interest and the 
    corresponding label IDs.
    
    Parameters:
    all_fvs: ndarray of float, containing all feature vectors
    all_ids: array of int, containing corresponding IDs (class labels)
    indices_of_int: ndarray of int, containing the indices of interest 

    Return:
    fvs_of_int:  array of float, containing feature vectors of interest (individual arrays).
    label_ids_of_int:  array of int, containing label IDs of interest for fvs_of_int
    """

    fvs_of_int = []
    label_ids_of_int = []
    for curr_index in indices_of_int:
        fvs_of_int.append(all_fvs[curr_index])
        label_ids_of_int.append(all_ids[curr_index])
    return fvs_of_int, label_ids_of_int


def pickle_object(value, output_file):
    """Utility function that pickles an object value to an output file
    Parameter:
    value: an object of some kind
    output_file: a string ending with .pkl, the location where the object will be pickled.
    """

    with open(output_file, 'wb') as output:
        pickle.dump(value, output)


def pickle_fvs_and_labels(fvs, label_IDs, prefix_for_output_file, output_dir):
    """Pickle the feature vectors and labels to file; feature vectors to be written to 
    $output_dir/${prefix_for_output_file}_fvs.pkl and labels to be written to 
    $output_dir/${prefix_for_output_file}_labels.pkl
    
    Parameters:
    fvs: array of fvs (individual arrays)
    label_IDs: array of int (each element is a label ID)
    prefix_for_output_file: str to indicate prefix of filename where fvs and labels will be being written out
    output_dir: str, the directory where the feature vectors and labels will be written.
    """

    output_for_fvs = output_dir + "/" + prefix_for_output_file + "_fvs.pkl"
    output_for_labels = output_dir + "/" + prefix_for_output_file + "_label_IDs.pkl"
    pickle_object(fvs, output_for_fvs)
    pickle_object(label_IDs, output_for_labels)


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