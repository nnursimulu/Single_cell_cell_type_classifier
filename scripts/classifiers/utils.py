import pickle

def parse_out_cv_accuracy_for_data_subsection(string):
    """Read out cross-validation result for a data subsection from string.
    
    Parameters:
    string: (str) line written out by append_cv_result_for_data_subsection

    Returns:
    fold_i: (int) data subsection index.
    setting: (str) hyperparameter setting.
    accuracy: (float) accuracy for this hyperparameter setting over this data subsection.
    """

    split = string.split("\t")
    fold_i = int(split[0].split("=")[1])
    setting = "\t".join(split[1:-1])
    accuracy = float(split[-1])
    return fold_i, setting, accuracy


def append_cv_result_for_data_subsection(fold_i, setting, accuracy, filename):
    """Write out cross-validation result for a data subsection.
    
    Parameters:
    fold_i: (str) data subsection index.
    setting: (str) hyperparameter setting.
    accuracy: (float) accuracy for this hyperparameter setting over this data subsection.
    filename: (str) path to file where this result will be written.
    """

    with open(filename, "a") as writer:
        info = "\t".join(["Subsection="+str(fold_i), setting, str(accuracy)])
        writer.write(info + "\n")


def write_out_summarized_accuracy(dict_setting_to_accuracy, output_file):
    """Write out summarized accuracy across different settings to output file
    
    Parameters:
    dict_setting_to_accuracy: dict mapping hyperparameter setting (str) to averaged accuracy across all sections of data (float)
    output_file: str denoting absolute path to output file
    """

    with open(output_file, "w") as writer:
        writer.write("Setting\tAccuracy\n")
        for setting, accuracy in dict_setting_to_accuracy.items():
            writer.write(setting + "\t" + str(accuracy) + "\n")


def update_accuracy_dict(dict_setting_to_section_to_accuracy, setting, fold_i, accuracy):
    """Update dictionary of accuracy with accuracy for this particular setting (set of hyperparameters) and 
    this section of the data.
    
    Parameters:
    dict_setting_to_section_to_accuracy: dict mapping setting of hyperparameters to section of data to accuracy.
    setting: A string representing the particular combination of hyperparameters used.
    fold_i: an int, denoting a different section of the training data for cross-validation.
    accuracy: float, computed accuracy for this section of the data and this set of hyperparameters.
    """

    if setting not in dict_setting_to_section_to_accuracy:
        dict_setting_to_section_to_accuracy[setting] = {}
    dict_setting_to_section_to_accuracy[setting][fold_i] = accuracy


def summarize_accuracy_across_hyperparameters(dict_setting_to_section_to_accuracy):
    """Return dict with summarize accuracy across different hyperparameter setting.
    
    Parameters:
    dict_setting_to_section_to_accuracy: dict mapping hyperparameter setting (str) to section of data (int) to accuracy (float).

    Returns:
    dict_setting_to_accuracy: dict mapping hyperparameter setting (str) to averaged accuracy across all sections of data (float)
    """

    dict_setting_to_accuracy = {}
    for setting, section_to_accuracy in dict_setting_to_section_to_accuracy.items():
        accuracy_array = []
        for section, accuracy in section_to_accuracy.items():
            accuracy_array.append(accuracy)
        avg_accuracy = sum(accuracy_array)/len(accuracy_array)
        dict_setting_to_accuracy[setting] = avg_accuracy
    return dict_setting_to_accuracy


def unpickle_object(filename):
    """Unpickle and return object.
    
    Parameter:
    filename: str denoting absolute path to an object.
    
    Return:
    Unpickled object.
    """

    with open(filename, 'rb') as input_data:
        var = pickle.load(input_data)
    return var


def compute_accuracy(predicted_IDs, actual_IDs):
    """Compute overall accuracy based on predicted and actual labels.
    Accuracy is computed as (number of correct predictions)/(total number of labels)
    
    Parameter:
    predicted_IDs: list of int (IDs) indicating predicted labels of feature vectors.
    actual_IDs: list of int (IDs) indicating actual labels of feature vectors. 

    Return:
    A float representing accuracy.
    """

    num_correct = 0
    for pred_ID, actual_ID in zip(predicted_IDs, actual_IDs):
        if pred_ID == actual_ID:
            num_correct += 1
    accuracy = float(num_correct)/len(actual_IDs)
    return accuracy


def convert_to_dense_repr(sparse_feature_vectors):
    """
    Convert an array of sparse feature vectors to an array of array.

    Parameter:
    sparse_feature_vectors: array of sparse feature vectors (matrix objects)

    Return:
    An array of arrays, each representing a feature vector.
    """

    dense_feature_vectors = []
    for fv in sparse_feature_vectors:
        dense_fv = fv.toarray().tolist()[0]
        dense_feature_vectors.append(dense_fv)
    return dense_feature_vectors