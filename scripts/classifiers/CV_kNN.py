import utils
import os
from argparse import ArgumentParser
from sklearn.neighbors import KNeighborsClassifier

if __name__ == '__main__':
    
    parser = ArgumentParser(description="Cross-validation of k-nearest neighbour for different values of k.")
    parser.add_argument("--training_folder", type=str, help="Absolute path to folder that contains the training data.",
                        required=True)
    parser.add_argument("--results_folder", type=str, help="Absolute path to folder that will contain the results",
                        required=True)

    args = parser.parse_args()
    training_folder = args.training_folder
    results_folder = args.results_folder

    # Create the results directory if it does not exist yet.
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Values of k to try.
    k_values = [5, 10, 25, 50, 75, 100]

    # Dict with accuracy values for different values of k and sections of the data.
    dict_k_to_section_to_accuracy = {}

    print ("========Running kNN cross-validation for different values of k.")

    # For each subsection of the data, try different values of k and see impact on accuracy.
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

        # Try different value of k for the kNN classifier.
        for k_value in k_values:
            print ("======Training for k=" + str(k_value) + ", section of the data:" + str(fold_i))
            knn = KNeighborsClassifier(n_neighbors=k_value)
            knn.fit(training_fvs, training_label_IDs)

            predicted_label_IDs = knn.predict(test_fvs)
            accuracy = utils.compute_accuracy(predicted_label_IDs, test_label_IDs)
            
            utils.update_accuracy_dict(dict_k_to_section_to_accuracy, str(k_value), fold_i, accuracy)
        fold_i += 1

    # Summarize the value of the accuracy across different sections of the data, and write out results.
    dict_k_to_accuracy = utils.summarize_accuracy_across_hyperparameters(dict_k_to_section_to_accuracy)
    utils.write_out_summarized_accuracy(dict_k_to_accuracy, results_folder + "/CV_knn.out")