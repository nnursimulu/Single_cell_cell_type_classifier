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

def write_2d_matrix(dict_matrix, catID_to_category, writer):
    """Write a 2D matrix out.
    
    Parameter:
    dict_matrix: matrix of the form 'A': {'A': 'a_a', 'B': 'a_b', 'C': 'a_c'}, 
                                            'B': {'A': 'b_a', 'B': 'b_b', 'C': 'b_c'}, 
                                                'C': {'A': 'c_a', 'B': 'c_b', 'C': 'c_c'}
    catID_to_category: (dict) id of category to name (eg: 'A' may stand for 'apple')
    writer: open writer stream

    Writes in the form 
            A	B	C
        A	a_a	a_b	a_c
        B	b_a	b_b	b_c
        C	c_a	c_b	c_c
    If key is missing in inner dict, writes 0 in that cell.
    """

    categoryIDs = dict_matrix.keys()
    header = "\t"
    for cat_2 in categoryIDs:
        header += "\t" + catID_to_category[cat_2]
    writer.write(header + "\n")
    for cat_1 in categoryIDs:
        info = [catID_to_category[cat_1]]
        for cat_2 in categoryIDs:
            if cat_2 not in dict_matrix[cat_1]:
                info.append("0")
            else:
                info.append(str(dict_matrix[cat_1][cat_2]))
        writer.write("\t" + "\t".join(info) + "\n")


def write_out_results(predicted_label_IDs, test_label_IDs, id_to_label, classifier_name, results_file):
    """Calculate and write out performance results for a particular classifier.

    Parameter:
    predicted_label_IDs: list of (int) predicted label IDs/classes
    test_label_IDs: list of (int) actual label IDs/classes
    id_to_label: (dict) id of a label ID to name of label
    classifier_name: (str) name of classifier in question
    results_file: (str) file where results will be written.
    """
    
    accuracy, macro_f1, id_to_precision, id_to_recall, observed_to_predicted_label = \
        utils.compute_accuracy(predicted_label_IDs, test_label_IDs)
    with open(results_file, "w") as writer:
        writer.write("===============\nResults for " + classifier_name + "\n\n")
        writer.write("*Accuracy: " + str(accuracy) + "\n")
        writer.write("*Macro-F1: " + str(macro_f1) + "\n")
        writer.write("*Label to precision and recall:\n")
        writer.write("\tLabel\tPrecision\tRecall\n")
        for curr_id, label in id_to_label.items():
            precision, recall = "NA", "NA"
            if curr_id in id_to_precision:
                precision = str(id_to_precision[curr_id])
            if curr_id in id_to_recall:
                recall = str(id_to_recall[curr_id])
            info = [label, precision, recall]
            writer.write("\t" + "\t".join(info) + "\n")
        writer.write("\n\n*Confusion matrix (row is actual num, column is predicted num):\n")
        write_2d_matrix(observed_to_predicted_label, id_to_label, writer)
    print ("Finished calculating performance for " + classifier_name)



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
    write_out_results(predicted_knn_label_IDs, test_label_IDs, id_to_label, "knn", knn_results_file)

    ############################################
    # Now, for RF classifier-balanced.
    rf_bal_results_file = results_folder + "/OVERALL_RF_balanced_results.out"
    rf_bal = RandomForestClassifier(max_depth=20, n_estimators=500, \
                    class_weight="balanced", random_state=0)
    rf_bal.fit(training_fvs, training_label_IDs)
    predicted_rf_bal_label_IDs = rf_bal.predict(test_fvs)
    write_out_results(predicted_rf_bal_label_IDs, test_label_IDs, id_to_label, "RF-balanced", rf_bal_results_file)

    ############################################
    # Now, for RF classifier-not balanced.
    rf_bal_results_file = results_folder + "/OVERALL_RF_None_results.out"
    rf_bal = RandomForestClassifier(max_depth=None, n_estimators=300, \
                    class_weight=None, random_state=0)
    rf_bal.fit(training_fvs, training_label_IDs)
    predicted_rf_not_bal_label_IDs = rf_bal.predict(test_fvs)
    write_out_results(predicted_rf_not_bal_label_IDs, test_label_IDs, id_to_label, "RF-not balanced", rf_bal_results_file)

    ############################################
    # Now, for SVM-balanced
    svm_bal_results_file = results_folder + "/OVERALL_SVM_bal_results.out"
    svm_bal = make_pipeline(StandardScaler(), LinearSVC(random_state=0, dual=False, penalty="l1", \
        C=0.1, class_weight="balanced", max_iter=5000))
    svm_bal.fit(training_fvs, training_label_IDs)
    predicted_svm_bal_label_IDs = svm_bal.predict(test_fvs)
    write_out_results(predicted_svm_bal_label_IDs, test_label_IDs, id_to_label, "SVM-balanced", svm_bal_results_file)

    ############################################
    # Now, for SVM-not balanced
    svm_not_bal_results_file = results_folder + "/OVERALL_SVM_None_results.out"
    svm_not_bal = make_pipeline(StandardScaler(), LinearSVC(random_state=0, dual=False, penalty="l1", \
        C=0.1, class_weight="balanced", max_iter=5000))
    svm_not_bal.fit(training_fvs, training_label_IDs)
    predicted_svm_not_bal_label_IDs = svm_not_bal.predict(test_fvs)
    write_out_results(predicted_svm_not_bal_label_IDs, test_label_IDs, id_to_label, "SVM-balanced", svm_not_bal_results_file)

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
    layers = []
    for num in [50, 25]:
        layers.append(tf.keras.layers.Dense(units=num, activation='relu'))
        layers.append(tf.keras.layers.Dropout(0.2))
    # The last layer is one for prediction.
    layers.append(tf.keras.layers.Dense(units=11, activation='linear'))
    model = tf.keras.Sequential(layers)

    # Use adaptive moment estimation for faster learning, and use logits to deal with numerical error.
    # Stop learning at the 4th epoch.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    model.fit(training_fvs, training_label_IDs, epochs=4)
    logits = model.predict(test_fvs)
    predicted_nn_labels = utils.transform_from_nn_pred(logits)
    write_out_results(predicted_nn_labels, test_label_IDs, id_to_label, "NN", nn_results_file)