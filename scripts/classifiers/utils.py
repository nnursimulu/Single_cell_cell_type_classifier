import pickle
import tensorflow as tf

ACCURACY = "ACCURACY"
MACRO_F1 = "MACRO_F1"

def transform_from_nn_pred(logits):
    """Get neural net predictions from logits
    
    Parameters:
    logits: (numpy.ndarray) output logits from neural net

    Return:
    Class labels
    """

    list_tensors = tf.argmax(input=logits, axis=1)
    list_values = []
    for tensor in list_tensors:
        list_values.append(tensor.numpy())
    return list_values


def is_int(string):
    """Returns True is string is an integer.
    
    Parameter:
    string: str
    
    Returns:
    True iff string is an integer"""

    try:
        int(string)
        return True
    except:
        return False


def parse_out_cv_accuracy_for_data_subsection(string):
    """Read out cross-validation result for a data subsection from string.
    
    Parameters:
    string: (str) line written out by append_cv_result_for_data_subsection

    Returns:
    fold_i: (int) data subsection index.
    setting: (str) hyperparameter setting.
    accuracy: (float) accuracy for this hyperparameter setting over this data subsection.
    macro-f1: (float) macro-f1 for this hyperparameter setting over this data subsection.
    """

    split = string.split("\t")
    fold_i = int(split[0].split("=")[1])
    setting = "\t".join(split[1:-2])
    accuracy = float(split[-2])
    macro_f1 = float(split[-1])
    return fold_i, setting, accuracy, macro_f1


def append_cv_result_for_data_subsection(fold_i, setting, accuracy, macro_f1, filename):
    """Write out cross-validation result for a data subsection.
    
    Parameters:
    fold_i: (str) data subsection index.
    setting: (str) hyperparameter setting.
    accuracy: (float) accuracy for this hyperparameter setting over this data subsection.
    macro_f1: (float) macro-F1 for this hyperparameter setting over this data subsection.
    filename: (str) path to file where this result will be written.
    """

    with open(filename, "a") as writer:
        info = "\t".join(["Subsection="+str(fold_i), setting, str(accuracy), str(macro_f1)])
        writer.write(info + "\n")


def write_out_summarized_accuracy(dict_setting_to_accuracy, output_file):
    """Write out summarized accuracy and macro-f1 across different settings to output file
    
    Parameters:
    dict_setting_to_accuracy: dict mapping hyperparameter setting (str) to averaged accuracy across all sections of data (float)
    output_file: str denoting absolute path to output file
    """

    with open(output_file, "w") as writer:
        writer.write("Setting\tAccuracy\tMacro_F1\n")
        for setting, perf_dict in dict_setting_to_accuracy.items():
            accuracy = perf_dict[ACCURACY]
            macro_f1 = perf_dict[MACRO_F1]
            writer.write(setting + "\t" + str(accuracy) + "\t" + str(macro_f1) + "\n")


def update_accuracy_dict(dict_setting_to_section_to_accuracy, setting, fold_i, accuracy, macro_f1):
    """Update dictionary of accuracy with accuracy/macro-F1 for this particular setting (set of hyperparameters) and 
    this section of the data.
    
    Parameters:
    dict_setting_to_section_to_accuracy: dict mapping setting of hyperparameters to section of data to accuracy.
    setting: A string representing the particular combination of hyperparameters used.
    fold_i: an int, denoting a different section of the training data for cross-validation.
    accuracy: float, computed accuracy for this section of the data and this set of hyperparameters.
    """

    if setting not in dict_setting_to_section_to_accuracy:
        dict_setting_to_section_to_accuracy[setting] = {}
    dict_setting_to_section_to_accuracy[setting][fold_i] = {ACCURACY: accuracy, MACRO_F1: macro_f1}


def summarize_accuracy_across_hyperparameters(dict_setting_to_section_to_accuracy):
    """Return dict with summarize accuracy across different hyperparameter setting.
    
    Parameters:
    dict_setting_to_section_to_accuracy: dict mapping hyperparameter setting (str) to section of data (int) to accuracy and macro-f1 (float).

    Returns:
    dict_setting_to_accuracy: dict mapping hyperparameter setting (str) to averaged accuracy and macro-f1 across all sections of data (float)
    """

    dict_setting_to_accuracy = {}
    for setting, section_to_accuracy in dict_setting_to_section_to_accuracy.items():
        accuracy_array = []
        macro_f1_array = []
        for section, perf_dict in section_to_accuracy.items():
            accuracy_array.append(perf_dict[ACCURACY])
            macro_f1_array.append(perf_dict[MACRO_F1])
        avg_accuracy = sum(accuracy_array)/len(accuracy_array)
        avg_macro_f1 = sum(macro_f1_array)/len(macro_f1_array)
        dict_setting_to_accuracy[setting] = {ACCURACY: avg_accuracy, MACRO_F1: avg_macro_f1}
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
    """Compute overall accuracy and macro-f1 based on predicted and actual labels.
    Accuracy is computed as (number of correct predictions)/(total number of labels)
    
    Parameter:
    predicted_IDs: list of int (IDs) indicating predicted labels of feature vectors.
    actual_IDs: list of int (IDs) indicating actual labels of feature vectors. 

    Return:
    accuracy: A float representing accuracy.
    macro_f1: A float representing macro-averaged F1.
    label_to_precision: (dict) label (int) to precision (float)
    label_to_recall: (dict) label to recall (float)
    observed_to_predicted_label: (dict) observed label to predicted label mapping; 
                how many of the examples in an actual category are predicted in another category.
    """

    num_correct = 0
    observed_to_predicted_label = {}
    predicted_to_observed_label = {}

    # Compute accuracy.
    for pred_ID, actual_ID in zip(predicted_IDs, actual_IDs):
        if pred_ID == actual_ID:
            num_correct += 1
        update_dict_freq(observed_to_predicted_label, actual_ID, pred_ID)
        update_dict_freq(predicted_to_observed_label, pred_ID, actual_ID)
    accuracy = float(num_correct)/len(actual_IDs)

    label_to_precision = {}
    label_to_recall = {}

    # Compute macro-avg precision.
    num_pred_labels = 0
    for predicted_label, observed_label_to_count in predicted_to_observed_label.items():
        tp = 0
        fp = 0
        for observed_label, count in observed_label_to_count.items():
            if predicted_label == observed_label:
                tp = count
            else: # class predicted for elements of another class are false positives.
                fp += count
        precision = float(tp)/(tp + fp)
        label_to_precision[predicted_label] = precision
        num_pred_labels += 1
    macro_precision = macro_average(label_to_precision, num_pred_labels)

    # Compute macro-avg recall.
    num_obs_labels = 0
    for observed_label, predicted_label_to_count in observed_to_predicted_label.items():
        tp = 0
        fn = 0
        for predicted_label, count in predicted_label_to_count.items():
            if observed_label == predicted_label:
                tp = count
            else: # actual class members predicted of another class are false negatives.
                fn += count
        recall = float(tp)/(tp + fn)
        label_to_recall[observed_label] = recall
        num_obs_labels += 1
    macro_recall = macro_average(label_to_recall, num_obs_labels)

    macro_f1 = 2 * (macro_precision * macro_recall)/(macro_precision + macro_recall)
    return accuracy, macro_f1, label_to_precision, label_to_recall, observed_to_predicted_label


def macro_average(label_to_perf, denominator):
    """Calculate the macro-averaged performance value and return.
    
    Parameter:
    label_to_perf: (dict) label, an int, to a performance measure (float)
    denominator: (int) number to macro-average by.

    Returns:
    A float representing macro-averaged value.
    """

    perf_sum = 0.0
    for label, perf in label_to_perf.items():
        perf_sum += perf
    return perf_sum/denominator


def update_dict_freq(dict_A_to_B, a, b):
    """Update how many times elements of category A (eg: actual label) are found alongside category B (eg: predicted label).

    Parameter:
    dict_A_to_B: dict of freq of category A elements to category B elements (eg: frequency of observed label/A to predicted label/B)
    a: (int) element of category A (eg: actual label)
    b: (int) element of category B (eg: predicted label)
    """

    if a not in dict_A_to_B:
        dict_A_to_B[a] = {}
    if b not in dict_A_to_B[a]:
        dict_A_to_B[a][b] = 0
    dict_A_to_B[a][b] += 1


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