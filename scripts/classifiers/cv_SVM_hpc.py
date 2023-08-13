import utils
import os
from argparse import ArgumentParser
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

if __name__ == '__main__':
    
    parser = ArgumentParser(description="Cross-validation of support vector machine; trains and tests on subsection of data with certain hyperparameters.")
    parser.add_argument("--training_folder", type=str, help="Absolute path to folder that contains the training data",
                        required=True)
    parser.add_argument("--results_file", type=str, help="Absolute path to file that will contain the results",
                        required=True)
    parser.add_argument("--c_value", type=float, help="Value of the C-regularization parameter; need string that can be converted to float",
                        required=True)
    parser.add_argument("--is_l1", type=int, help="1 to try l1 regularization, 0 otherwise",
                        required=True)
    parser.add_argument("--subsection_of_data", type=int, \
        help="On which subsection of data to train and test classifiers; specify int from 0 to 4.", required=True)
    parser.add_argument("--is_balanced", type=int, help="1 to build a balanced classifier, 0 otherwise.", required=True)

    args = parser.parse_args()
    training_folder = args.training_folder
    results_file = args.results_file
    c_value = args.c_value
    is_l1 = args.is_l1
    fold_i = args.subsection_of_data
    is_balanced = args.is_balanced

    if is_l1 not in [0, 1]:
        raise Exception("Incorrect value of is_l1")
    if fold_i not in [0, 1, 2, 3, 4]:
        raise Exception("Incorrect value of subsection_of_data")
    if is_balanced not in [0, 1]:
        raise Exception("Incorrect value of is_balanced")

    if is_l1==1:
        penalty_value = "l1"
    else:
        penalty_value = "l2"

    if is_balanced==1:
        class_weight_value = 'balanced'
    else:
        class_weight_value = None

    try:
        c_value = float(c_value)
    except:
        raise Exception("C-value cannot be converted to float.")

    # Create the results directory if it does not exist yet.
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    # For this subsection of the data, try particular value of hyperparameter and calculate accuracy.
    # Load the training feature vectors and corresponding labels for this round of cross-validation.
    prefix_file = training_folder + "/CV_" + str(fold_i)
    training_fvs = utils.unpickle_object(prefix_file + "_training_fvs.pkl")
    training_fvs = utils.convert_to_dense_repr(training_fvs)
    training_label_IDs = utils.unpickle_object(prefix_file + "_training_label_IDs.pkl")

    # Load the test feature vectors and corresponding labels for this round of cross-validation.
    test_fvs = utils.unpickle_object(prefix_file + "_test_fvs.pkl")
    test_fvs = utils.convert_to_dense_repr(test_fvs)
    test_label_IDs = utils.unpickle_object(prefix_file + "_test_label_IDs.pkl")

    # Train the SVM classifier, with normalized features.
    dual_val=True
    if penalty_value == "l1":
        dual_val=False # The combination of penalty='l1' and loss='squared_hinge' are not supported when dual=True
    # Increase maximum number of iterations because otherwise no convergence.
    clf = make_pipeline(StandardScaler(), LinearSVC(random_state=0, dual=dual_val, penalty=penalty_value, \
        C=c_value, class_weight=class_weight_value, max_iter=5000))
    clf.fit(training_fvs, training_label_IDs)

    # Predict labels.
    predicted_label_IDs = clf.predict(test_fvs)
    accuracy, macro_f1, _, _, _ = utils.compute_accuracy(predicted_label_IDs, test_label_IDs)
    
    # Write out results.
    setting = " ".join(["C="+str(c_value), "penalty="+str(penalty_value), "class_weight="+str(class_weight_value)])
    utils.append_cv_result_for_data_subsection(fold_i, setting, accuracy, macro_f1, results_file)