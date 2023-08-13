import utils
import os
from argparse import ArgumentParser
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':
    
    parser = ArgumentParser(description="Cross-validation of random forest classifier for different number of trees of different depths.")
    parser.add_argument("--training_folder", type=str, help="Absolute path to folder that contains the training data.",
                        required=True)
    parser.add_argument("--results_folder", type=str, help="Absolute path to folder that will contain the results",
                        required=True)
    parser.add_argument("--has_balanced_trees", type=int, help="Whether the trees of the forest are to be balanced or not; 1=balanced, 0=not",
                        required=True)

    args = parser.parse_args()
    training_folder = args.training_folder
    results_folder = args.results_folder
    has_balanced_trees = args.has_balanced_trees
    if has_balanced_trees not in [0, 1]:
        raise Exception("Invalid value for whether trees should be balanced or not")
    if has_balanced_trees==1:
        balance_value = "balanced"
    else:
        balance_value = None

    # Create the results directory if it does not exist yet.
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Number of trees to try.
    n_trees_values = [25, 50, 100, 150, 200, 300, 500]

    # Different maximum depths to try.
    max_depth_values = [5, 10, 20, None]

    # Dict with accuracy values for different values of k and sections of the data.
    dict_setting_to_section_to_accuracy = {}

    print ("========Running random forest cross-validation for different max_depth and n_estimators.")

    # For each subsection of the data, try different values of hyperparameters and see impact on accuracy.
    fold_i = 0
    while fold_i < 5:
        # Load the training feature vectors and corresponding labels for this round of cross-validation.
        prefix_file = training_folder + "/CV_" + str(fold_i)
        training_fvs = utils.unpickle_object(prefix_file + "_training_fvs.pkl")
        training_fvs = utils.convert_to_dense_repr(training_fvs)
        training_label_IDs = utils.unpickle_object(prefix_file + "_training_label_IDs.pkl")

        # Load the test feature vectors and corresponding labels for this round of cross-validation.
        test_fvs = utils.unpickle_object(prefix_file + "_test_fvs.pkl")
        test_fvs = utils.convert_to_dense_repr(test_fvs)
        test_label_IDs = utils.unpickle_object(prefix_file + "_test_label_IDs.pkl")

        # Try different number of trees.
        for num_trees in n_trees_values:
            print ("======Training for n_trees=" + str(num_trees) + " for section of the data:" + str(fold_i))
            # Try different max depth:
            for curr_max_depth in max_depth_values:
                print ("============Training for max_depth=" + str(curr_max_depth) + " for section of the data:" + str(fold_i))

                rf = RandomForestClassifier(max_depth=curr_max_depth, n_estimators=num_trees, \
                    class_weight=balance_value, random_state=0)
                rf.fit(training_fvs, training_label_IDs)

                predicted_label_IDs = rf.predict(test_fvs)
                accuracy, macro_f1, _, _, _ = utils.compute_accuracy(predicted_label_IDs, test_label_IDs)
                
                setting = "num_trees=" + str(num_trees) + "\tmax_depth=" + str(curr_max_depth)
                utils.update_accuracy_dict(dict_setting_to_section_to_accuracy, setting, fold_i, accuracy, macro_f1)
        fold_i += 1

    # Summarize the value of the accuracy across different sections of the data, and write out results.
    dict_k_to_accuracy = utils.summarize_accuracy_across_hyperparameters(dict_setting_to_section_to_accuracy)
    utils.write_out_summarized_accuracy(dict_k_to_accuracy, results_folder + "/CV_RF_" + str(balance_value) + ".out")