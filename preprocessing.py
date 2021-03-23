from data_handling import *

def get_vectorized_input_per_descriptor(df):
    """ Shapes the input as vectors of different descriptor values.

    Parameters
    ----------
    df : DataFrame
        Dataset.
        
    Returns
    -------
    Numpy array
        Reshaped dataset
    """
    numerical = df.drop(get_descriptive_col_names(), axis=1).values
    arr = []
    for row in range(numerical.shape[0]):
        arr.append(np.split(numerical[row], [12, 12+64, 12+64+80, 12+64+80+62]))
    return np.vstack(arr)

def pad_input(attr_lengths, X, n_of_descriptors=5):
    """ Pads the input in regards to the descriptor values.
    Since the descriptor's values have different lengths, we need to pad them to
    the greatest one to feed them into the NN. `special_value` will be passed into
    network's Masking layer and ignored in the trainin process (no additional noise).

    Parameters
    ----------
    attr_lengths : List
        List of descriptor lengths
    X : Numpy array
        Input      
    Returns
    -------
    Numpy array
        Reshaped dataset
    Int
        Special value marking the pad
    """
    special_value = -10.0
    max_seq_len = max(attr_lengths)
    padded_X = []
    for row in X:
        Xpad = np.full((n_of_descriptors, max_seq_len), fill_value=special_value)
        for s, x in enumerate(row):
            seq_len = x.shape[0]
            Xpad[s, 0:seq_len] = x
        padded_X.append(Xpad)
    return np.asarray(padded_X), special_value

def scale_per_descriptor(df, labels, descriptor_value_counts):
    """
    Scales the descriptor values per descriptors using sklearn.preprocessing.scale 
     - centers to the mean and ensures unit variance.
    'Per descriptor' means that there are 5 descriptors of different lengths in CoPhIR dataset.
    The scaling is done individually per all of these 5.

    Parameters
    ----------
    df : DataFrame
        Dataset.
    descriptor_value_counts : list
        Number of individual values in each descriptor.
        
    Returns
    -------
    DataFrame
        Normalized dataset.
    """
    col_pos = 0
    normalized = []
    numerical = df.drop(labels+["object_id"], axis=1).values
    for descriptor_value in descriptor_value_counts:
        current = numerical[:, col_pos:col_pos+descriptor_value]
        normalized.append(preprocessing.scale(current))
        col_pos += descriptor_value
    df = df.drop(df.columns.difference(labels+["object_id"]), 1)
    df = pd.concat([df, pd.DataFrame(np.hstack((normalized)))], axis=1)
    return df