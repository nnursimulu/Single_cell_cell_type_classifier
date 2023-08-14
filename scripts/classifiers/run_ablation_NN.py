import utils
import os
from argparse import ArgumentParser
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import numpy as np

def create_model(layers_num_units, dropout_rate):
    """Create NN and return it.
    
    Parameters:
    layers_num_units: list of int, indicating how many relu units in each layer.
    dropout_rate: dropout rate.

    Return:
    model of desired architecture, also compiled.
    """

    # Create architecture of the network.
    layers = []
    for num in layers_num_units:
        layers.append(tf.keras.layers.Dense(units=num, activation='relu'))
        layers.append(tf.keras.layers.Dropout(dropout_rate))
    # The last layer is one for prediction.
    layers.append(tf.keras.layers.Dense(units=11, activation='linear'))
    model = tf.keras.Sequential(layers)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    return model


def predict_and_write_out(model, test_fvs, output_file):

    logits = model.predict(test_fvs)
    predicted_nn_labels = utils.transform_from_nn_pred(logits)
    utils.write_out_optimal_results(predicted_nn_labels, test_label_IDs, id_to_label, output_file, output_file)


if __name__ == '__main__':
    
    parser = ArgumentParser(description="Runs ablation experiment on optimal neural net architecture.")
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

    tf.random.set_seed(0)
    
    ####################
    # Experiment 1: Do not normalize the data.
    training_fvs = np.array(training_fvs)
    training_label_IDs = np.array(training_label_IDs)

    test_fvs = np.array(test_fvs)
    test_label_IDs = np.array(test_label_IDs)
    
    model_not_normalized = create_model([50, 25], 0.2)
    model_not_normalized.fit(training_fvs, training_label_IDs, epochs=8)
    predict_and_write_out(model_not_normalized, test_fvs, results_folder + "/ABLATION_NN_not_normalized.out")

    ####################
    # Normalize the feature vectors for next experiments.
    scaler = StandardScaler()
    training_fvs = scaler.fit_transform(training_fvs)
    test_fvs = scaler.transform(test_fvs) # Consequent to normalizing training, deal with test data accordingly.

    # Experiment 2: Without early stopping based on validation (optimal epoch determined to be around 4).
    # However, stop when training loss stops improving.
    model_not_early = create_model([50, 25], 0.2)
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
    model_not_early.fit(training_fvs, training_label_IDs, epochs=100, callbacks=[callback])
    predict_and_write_out(model_not_early, test_fvs, results_folder + "/ABLATION_NN_no_early_stop.out")

    ####################
    # Experiment 3: Without dropout.

    model_no_dropout = create_model([50, 25], 0)
    model_no_dropout.fit(training_fvs, training_label_IDs, epochs=8)
    predict_and_write_out(model_no_dropout, test_fvs, results_folder + "/ABLATION_NN_no_dropout.out")

    ####################
    # Experiment 4: Without second layer.

    model_shallow = create_model([50], 0.2)
    model_shallow.fit(training_fvs, training_label_IDs, epochs=8)
    predict_and_write_out(model_shallow, test_fvs, results_folder + "/ABLATION_NN_shallow.out")

    ####################
    # Experiment 4: With fewer units in first layer.

    model_fewer_unit_1 = create_model([40, 25], 0.2)
    model_fewer_unit_1.fit(training_fvs, training_label_IDs, epochs=8)
    predict_and_write_out(model_fewer_unit_1, test_fvs, results_folder + "/ABLATION_NN_fewer_unit_1.out")