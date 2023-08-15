import utils
import os
import numpy as np
from argparse import ArgumentParser

def read_pred_labels(file_name):
    """Read the predicted labels from file name and return.
    
    Parameter:
    file_name: (str) path to file
    
    Returns:
    list of int/IDs"""

    liste = []
    with open(file_name) as reader:
        for line in reader:
            line = line.strip()
            if line == "":
                continue
            liste.append(int(line.split()[0]))
    return liste


def get_argmax(liste):
    """Return most frequent item in a liste
    
    Parameter:
    liste: an array of elements

    Return:
    most frequent elems from liste in a set, and frequency with which element appears.
    """

    elem_to_freq = {}
    for elem in liste:
        if elem not in elem_to_freq:
            elem_to_freq[elem] = 1
        else:
            elem_to_freq[elem] += 1
    freq_to_elem = {}
    for elem, freq in elem_to_freq.items():
        if freq not in freq_to_elem:
            freq_to_elem[freq] = set()
        freq_to_elem[freq].add(elem)
    max_freq = max(freq_to_elem.keys())
    return freq_to_elem[max_freq], max_freq


if __name__ == '__main__':
    
    parser = ArgumentParser(description="Test majority rule on results from classifiers.")
    parser.add_argument("--input_folder", type=str, help="Absolute path to folder that contains the training and test data",
                        required=True)
    parser.add_argument("--results_folder", type=str, help="Absolute path to folder that contains results from classifiers",
                        required=True)

    args = parser.parse_args()
    input_folder = args.input_folder
    results_folder = args.results_folder

    # Load actual labels.
    test_label_IDs = utils.unpickle_object(input_folder + "/OVERALL_test_label_IDs.pkl")

    # First read the predictions of each classifier.
    classifiers = ["knn", "NN", "RF"]
    classifier_to_pred = {}
    for classifier in classifiers:
        results_file = results_folder + "/ALL_PREDs_on_test_" + classifier + ".out"
        classifier_to_pred[classifier] = read_pred_labels(results_file)

    # Calculate macro-F1 if combine classifiers by simple majority rule, favouring NN in case of no consensus.
    majority_rule_pred = []
    i = 0
    while i < len(test_label_IDs):
        knn_pred = classifier_to_pred["knn"][i]
        rf_pred = classifier_to_pred["RF"][i]
        nn_pred = classifier_to_pred["NN"][i]
        most_freq_elem, freq = get_argmax([knn_pred, rf_pred, nn_pred])
        most_freq_elem = list(most_freq_elem)[0] # Select first one for simplicity.
        if freq == 1: # favour NN in case no consensus
            majority_rule_pred.append(nn_pred)
        else:
            majority_rule_pred.append(most_freq_elem)
        i += 1

    print (majority_rule_pred)
    print (test_label_IDs)
    accuracy, macro_f1, _, _, _ = utils.compute_accuracy(majority_rule_pred, test_label_IDs)
    print ("Accuracy: " + str(accuracy))
    print ("Macro-F1: " + str(macro_f1))