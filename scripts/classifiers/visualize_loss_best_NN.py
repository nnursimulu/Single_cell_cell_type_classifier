import utils
import os
from argparse import ArgumentParser
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

def plot_loss_over_folds(plt, fold_i, loss_values, val_loss_values):
    """Plot training and validation loss over various cross-validation subsets of data.
    
    Parameter:
    plt: The plot object
    fold_i: (int) particular section of data
    loss_values: array of training loss vs epoch
    val_loss_values: array of validation loss vs epoch
    """

    plt.plot(loss_values, label='training')
    plt.plot(val_loss_values, label='validation')
    plt.ylabel('loss for round ' + str(fold_i))
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")


if __name__ == '__main__':
    
    parser = ArgumentParser(description="Generate training and validation loss plot over epochs for best hyperparameters for neural net.")
    parser.add_argument("--training_folder", type=str, help="Absolute path to folder that contains the training data",
                        required=True)

    args = parser.parse_args()
    training_folder = args.training_folder

    # Best hyperparameter settings found: dropout rate and num of nodes in each layer
    dropout_rate = 0.2
    num_of_nodes_list = [50, 25]

    # Prepare variables to keep track of loss.
    fold_to_loss = {}
    fold_to_val_loss = {}

    fold_i = 0
    while fold_i < 5:
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
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        history = model.fit(training_fvs, training_label_IDs, epochs=100, callbacks=[callback], \
            validation_data=(test_fvs, test_label_IDs))

        fold_to_loss[fold_i] = history.history['loss']
        fold_to_val_loss[fold_i] = history.history['val_loss']

        fold_i += 1

    # Prepare plot to display loss.
    fig = plt.figure()
    axes = fig.subplots(nrows=5, ncols=1)
    fig.suptitle("Training and validation loss over all data")

    fold_i = 0
    while fold_i < 5:
        plt.subplot(5, 1, fold_i+1)
        plot_loss_over_folds(plt, fold_i ,fold_to_loss[fold_i], fold_to_val_loss[fold_i])
        fold_i += 1
    plt.show()