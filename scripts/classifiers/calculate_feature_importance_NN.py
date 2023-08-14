import utils
import os
import numpy as np
from argparse import ArgumentParser
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

def read_by_order(filename):
    """Reads contents of file, and returns line number to element at that line.
    
    Parameter
    filename: (str) path to file.
    
    Return:
    dict of line number (int) to content of line (str)
    """

    id_to_elem = {}
    with open(filename) as reader:
        for i, line in enumerate(reader):
            id_to_elem[i] = line.strip()
    return id_to_elem


if __name__ == '__main__':
    
    parser = ArgumentParser(description="Train various classifiers on training data and test; write out the results.")
    parser.add_argument("--input_folder", type=str, help="Absolute path to folder that contains the training and test data",
                        required=True)
    parser.add_argument("--raw_folder", type=str, help="Absolute path to folder that contains the raw data",
                        required=True)
    parser.add_argument("--results_folder", type=str, help="Absolute path to folder that will contain the results",
                        required=True)

    args = parser.parse_args()
    input_folder = args.input_folder
    raw_folder = args.raw_folder
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

    # Set up neural net.
    # First, need to adjust the loaded data.
    # Normalize the data.
    scaler = StandardScaler()
    training_fvs = np.array(training_fvs)
    training_fvs = scaler.fit_transform(training_fvs)
    training_label_IDs = np.array(training_label_IDs)

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
    
    # Create test vectors to find the importance of each feature.
    # Each vector has a 1 in a single column.
    feature_test_fvs = []
    num_features = len(training_fvs[0])
    i = 0
    while i < num_features:
        curr_fv = np.zeros(num_features)
        curr_fv[i] = 1
        feature_test_fvs.append(curr_fv)
        i += 1
    feature_test_fvs = np.array(feature_test_fvs)

    # Now, get predictions of the model.
    logits = model.predict(feature_test_fvs)
    predicted_probs = tf.nn.softmax(logits)
    predicted_nn_labels = utils.transform_from_nn_pred(logits)

    # Now, get the label to the probability to the gene name.
    label_to_prob_to_gene = {}
    id_to_gene = read_by_order(raw_folder + "/ordered_gene_names.in")
    i = 0
    while i < num_features:
        gene_name = id_to_gene[i] # the gene in question.
        predicted_id = predicted_nn_labels[i]
        predicted_label = id_to_label[predicted_id] # predicted label if this gene.
        probability = predicted_probs[i][predicted_id].numpy() # probability with which this label was predicted.

        if predicted_label not in label_to_prob_to_gene:
            label_to_prob_to_gene[predicted_label] = {}
        if probability not in label_to_prob_to_gene[predicted_label]:
            label_to_prob_to_gene[predicted_label][probability] = set()
        label_to_prob_to_gene[predicted_label][probability].add(gene_name)
        i += 1

    # Write out the top 10 genes associated with a particular label (sorted by decreasing probability).
    top_x = 10
    result_file = results_folder + "/FEATURE_IMPORTANCE_gene_by_class.out"
    with open(result_file, "w") as writer:
        writer.write("Class\tProbability_pred\tGenes\n")
        for label, prob_to_gene in label_to_prob_to_gene.items():
            probs = sorted(prob_to_gene.keys(), reverse=True)
            for curr_prob in probs[:top_x]:
                curr_genes = list(prob_to_gene[curr_prob])
                info_to_write = [label, str(curr_prob), " ".join(curr_genes)]
                writer.write("\t".join(info_to_write) + "\n")