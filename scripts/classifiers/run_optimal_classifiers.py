import utils
import os
import numpy as np
from argparse import ArgumentParser
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf

def write_raw_predictions(predicted_IDs, actual_IDs, results_file):
    """Write raw predictions and actual IDs out.
    
    Parameters:
    predicted_IDs: list of predicted IDs.
    actual_IDs: list of actual IDs.
    results_file: (str) path to where results will be written.
    """

    with open(results_file, "w") as writer:
        for pred, actual in zip(predicted_IDs, actual_IDs):
            writer.write(str(pred) + "\t" + str(actual) + "\n")



if __name__ == '__main__':
    
    parser = ArgumentParser(description="Train various classifiers on training data and test; write out the results.")
    parser.add_argument("--input_folder", type=str, help="Absolute path to folder that contains the training and test data",
                        required=True)
    parser.add_argument("--results_folder", type=str, help="Absolute path to folder that will contain the results",
                        required=True)

    args = parser.parse_args()
    input_folder = args.input_folder
    results_folder = args.results_folder

    # Create the results directory if it does not exist yet.
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # First load the training data.
    training_fvs = utils.unpickle_object(input_folder + "/OVERALL_training_fvs.pkl")
    training_fvs = utils.convert_to_dense_repr(training_fvs)
    training_label_IDs = utils.unpickle_object(input_folder + "/OVERALL_training_label_IDs.pkl")

    # Now load the test data.
    test_fvs = utils.unpickle_object(input_folder + "/OVERALL_test_fvs.pkl")
    test_fvs = utils.convert_to_dense_repr(test_fvs)
    test_label_IDs = utils.unpickle_object(input_folder + "/OVERALL_test_label_IDs.pkl")

    # Load id to label
    id_to_label = utils.unpickle_object(input_folder + "/id_to_label.pkl")

    ############################################
    # First, for kNN classifier.
    knn_results_file = results_folder + "/OVERALL_knn_results.out"
    knn = KNeighborsClassifier(n_neighbors=75)
    knn.fit(training_fvs, training_label_IDs)
    predicted_knn_label_IDs = knn.predict(test_fvs)
    utils.write_out_optimal_results(predicted_knn_label_IDs, test_label_IDs, id_to_label, "knn", knn_results_file)
    write_raw_predictions(predicted_knn_label_IDs, test_label_IDs, results_folder + "/ALL_PREDs_on_test_knn.out")

    ############################################
    # Now, for RF classifier-balanced.
    rf_bal_results_file = results_folder + "/OVERALL_RF_balanced_results.out"
    rf_bal = RandomForestClassifier(max_depth=20, n_estimators=500, \
                    class_weight="balanced", random_state=0)
    rf_bal.fit(training_fvs, training_label_IDs)
    predicted_rf_bal_label_IDs = rf_bal.predict(test_fvs)
    utils.write_out_optimal_results(predicted_rf_bal_label_IDs, test_label_IDs, id_to_label, "RF-balanced", rf_bal_results_file)

    ############################################
    # Now, for RF classifier-not balanced.
    rf_bal_results_file = results_folder + "/OVERALL_RF_None_results.out"
    rf_bal = RandomForestClassifier(max_depth=None, n_estimators=300, \
                    class_weight=None, random_state=0)
    rf_bal.fit(training_fvs, training_label_IDs)
    predicted_rf_not_bal_label_IDs = rf_bal.predict(test_fvs)
    utils.write_out_optimal_results(predicted_rf_not_bal_label_IDs, test_label_IDs, id_to_label, "RF-not balanced", rf_bal_results_file)
    write_raw_predictions(predicted_rf_not_bal_label_IDs, test_label_IDs, results_folder + "/ALL_PREDs_on_test_RF.out")

    ############################################
    # Now, for SVM-balanced
    svm_bal_results_file = results_folder + "/OVERALL_SVM_bal_results.out"
    svm_bal = make_pipeline(StandardScaler(), LinearSVC(random_state=0, dual=False, penalty="l1", \
        C=0.1, class_weight="balanced", max_iter=5000))
    svm_bal.fit(training_fvs, training_label_IDs)
    predicted_svm_bal_label_IDs = svm_bal.predict(test_fvs)
    utils.write_out_optimal_results(predicted_svm_bal_label_IDs, test_label_IDs, id_to_label, "SVM-balanced", svm_bal_results_file)

    ############################################
    # Now, for SVM-not balanced
    svm_not_bal_results_file = results_folder + "/OVERALL_SVM_None_results.out"
    svm_not_bal = make_pipeline(StandardScaler(), LinearSVC(random_state=0, dual=False, penalty="l1", \
        C=0.1, class_weight=None, max_iter=5000))
    svm_not_bal.fit(training_fvs, training_label_IDs)
    predicted_svm_not_bal_label_IDs = svm_not_bal.predict(test_fvs)
    utils.write_out_optimal_results(predicted_svm_not_bal_label_IDs, test_label_IDs, id_to_label, "SVM-not balanced", svm_not_bal_results_file)

    ############################################
    # Now, for neural net.
    # First, need to adjust the loaded data.
    # Normalize the data.
    nn_results_file = results_folder + "/OVERALL_NN_results.out"
    scaler = StandardScaler()
    training_fvs = np.array(training_fvs)
    training_fvs = scaler.fit_transform(training_fvs)
    training_label_IDs = np.array(training_label_IDs)

    # Normalize test data accordingly.
    test_fvs = np.array(test_fvs)
    test_fvs = scaler.transform(test_fvs) # Consequent to normalizing training, deal with test data accordingly.
    test_label_IDs = np.array(test_label_IDs)

    # Create architecture of the network.
    tf.random.set_seed(0)
    layers = []
    for num in [50, 25]:
        layers.append(tf.keras.layers.Dense(units=num, activation='relu'))
        layers.append(tf.keras.layers.Dropout(0.2))
    # The last layer is one for prediction.
    layers.append(tf.keras.layers.Dense(units=11, activation='linear'))
    model = tf.keras.Sequential(layers)

    # Use adaptive moment estimation for faster learning, and use logits to deal with numerical error.
    # Stop learning at the 8th epoch.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    model.fit(training_fvs, training_label_IDs, epochs=8)
    logits = model.predict(test_fvs)
    predicted_nn_labels = utils.transform_from_nn_pred(logits)
    utils.write_out_optimal_results(predicted_nn_labels, test_label_IDs, id_to_label, "NN", nn_results_file)
    write_raw_predictions(predicted_nn_labels, test_label_IDs, results_folder + "/ALL_PREDs_on_test_NN.out")