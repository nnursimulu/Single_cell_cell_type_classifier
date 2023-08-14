import utils
import os
from argparse import ArgumentParser
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import numpy as np

if __name__ == '__main__':
    
    parser = ArgumentParser(description="Cross-validation of neural networks; " + \
        "trains and tests on subsection of data with different number of layers and nodes.")
    parser.add_argument("--training_folder", type=str, help="Absolute path to folder that contains the training data",
                        required=True)
    parser.add_argument("--results_file", type=str, help="Absolute path to file that will contain the results",
                        required=True)
    parser.add_argument("--nn_setting", type=str, help="String specifying the number of relu units in each layer (separated by dropout layers);" + \
        " formatted as X_Y_Z where X nodes in the first layer, followed by drop-out layer, Y in second, followed by drop-out layer, Z next" + \
            "with another drop-out layer.  Adds by default 11 unit linear layer at the end.",
                        required=True)
    parser.add_argument("--dropout_rate", type=float, help="Dropout rate, a number between 0 and 1.",
                        required=True)
    parser.add_argument("--subsection_of_data", type=int, \
        help="On which subsection of data to train and test classifiers; specify int from 0 to 4.", required=True)

    args = parser.parse_args()
    training_folder = args.training_folder
    results_file = args.results_file
    nn_setting = args.nn_setting
    dropout_rate = args.dropout_rate
    fold_i = args.subsection_of_data

    if (dropout_rate > 1) or (dropout_rate < 0):
        raise Exception("Incorrect value of dropout_rate")
    if fold_i not in [0, 1, 2, 3, 4]:
        raise Exception("Incorrect value of subsection_of_data")
    
    # Process the number of nodes in each layer.
    num_of_nodes_str = nn_setting.strip().split("_")
    num_of_nodes_list = []
    for num_str in num_of_nodes_str:
        if utils.is_int(num_str):
            num_of_nodes_list.append(int(num_str))
        else:
            raise Exception("Invalid value for nn_setting")

    # Create the results directory if it does not exist yet.
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    # For this subsection of the data, try particular value of hyperparameter and calculate accuracy.
    # Load the training feature vectors and corresponding labels for this round of cross-validation.
    scaler = StandardScaler()
    prefix_file = training_folder + "/CV_" + str(fold_i)
    training_fvs = utils.unpickle_object(prefix_file + "_training_fvs.pkl")
    training_fvs = utils.convert_to_dense_repr(training_fvs)
    training_fvs = np.array(training_fvs)
    training_fvs = scaler.fit_transform(training_fvs) # Normalize data for faster learning.

    training_label_IDs = utils.unpickle_object(prefix_file + "_training_label_IDs.pkl")
    training_label_IDs = np.array(training_label_IDs)

    # Load the test feature vectors and corresponding labels for this round of cross-validation.
    test_fvs = utils.unpickle_object(prefix_file + "_test_fvs.pkl")
    test_fvs = utils.convert_to_dense_repr(test_fvs)
    test_fvs = np.array(test_fvs)
    test_fvs = scaler.transform(test_fvs) # Consequent to normalizing training, deal with test data accordingly.

    test_label_IDs = utils.unpickle_object(prefix_file + "_test_label_IDs.pkl")
    test_label_IDs = np.array(test_label_IDs)

    # Create architecture of the network.
    layers = []
    for num in num_of_nodes_list:
        layers.append(tf.keras.layers.Dense(units=num, activation='relu'))
        layers.append(tf.keras.layers.Dropout(dropout_rate))
    # The last layer is necessarily one for prediction and always added.
    layers.append(tf.keras.layers.Dense(units=11, activation='linear'))
    model = tf.keras.Sequential(layers)

    # Use adaptive moment estimation for faster learning, and use logits to deal with numerical error.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

    # Stop the training when there is no improvement in validation loss for 10 consecutive epochs.
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(training_fvs, training_label_IDs, epochs=100, callbacks=[callback], \
        validation_data=(test_fvs, test_label_IDs))
    # For debugging only, write out the maximum number of epochs to make sure keeps best model (doesn't if run till end).
    print("DEBUG_epoch: " + nn_setting + ": " + str(len(history.history['loss'])))

    # Predict labels.
    logits = model.predict(test_fvs)
    predicted_label_IDs = utils.transform_from_nn_pred(logits)
    accuracy, macro_f1, _, _, _ = utils.compute_accuracy(predicted_label_IDs, test_label_IDs)
    
    # Write out results.
    setting = " ".join(["nn_setting="+nn_setting, "dropout_rate="+str(dropout_rate)])
    utils.append_cv_result_for_data_subsection(fold_i, setting, accuracy, macro_f1, results_file)