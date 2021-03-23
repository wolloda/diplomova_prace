from imports import *
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import numpy as np
def get_index_col_names():
    return ['first_lvl_pivot_id', 'second_lvl_pivot_id']

def get_descriptive_col_names():
    return get_index_col_names() + ['object_id']

def get_descriptive_col_names_with_predictions():
    return get_descriptive_col_names() + ['first_lvl_pivot_id_pred'] + ['second_lvl_pivot_id_pred']

def split_data(df, random_state=21):
    """Splits the data to 80% training, 15% testing, 5% validation.
    """
    X = df.drop(get_descriptive_col_names(), axis=1)
    y = df[get_index_col_names()].values
    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.2, random_state=random_state)
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.25)
    print(X_train.shape, X_test.shape, X_val.shape, y_train.shape, y_test.shape, y_val.shape)
    return X_train, X_test, X_val, y_train, y_test, y_val

def encode_data(y):
    y_shape = 0
    if np.isnan(y).all():
        y_shape=y.shape
        y = [np.nan]
    encoder = LabelEncoder()
    #encoder = OrdinalEncoder()
    y = encoder.fit_transform(y)
    #y = encoder.transform(y)
    if y_shape != 0:
        y = np.zeros(y_shape, dtype=int)
    return y, encoder

def shuffle_and_encode_data(df):
    df_ = df.sample(frac=1)
    X = df_.drop(["first_lvl_pivot_id", "second_lvl_pivot_id","object_id"], axis=1).values
    y = df_[["first_lvl_pivot_id", "second_lvl_pivot_id"]].values
    y, encoder = encode_data(y[:, 0])
    return X, y, encoder

def get_training_data(df, labels, random_state=21, should_shuffle=True):
    """Splits the dataset into training data and labels, shuffles it.

    Parameters
    ----------
    df: Pandas DataFrame
        DF of all the values
    random_state (optional): int
        random state to fix the randomness of shuffle 

    Returns
    -------
    list
        training data, labels for L1, labels for L2, object ids
    """
    X = df.drop(labels + ["object_id"], axis=1)
    y = df[labels + ["object_id"]].values
    if should_shuffle:
        X, y = shuffle(X,y, random_state=random_state)
    y_1 = y[:, 0]; y_2 = y[:, 1]; y_obj_id = y[:, 2]
    print(X.shape, y_1.shape, y_2.shape)
    return X, y_1, y_2, y_obj_id

def create_splits(split_dfs, L1, L2, feature_values=282):
    """Splits the original df to training data and labels

    Parameters
    ----------
    split_dfs: List of Pandas DataFrames
        list of DFs split according to L1 and L2 (unique split for each unique combination)

    Returns
    -------
    list of dicts
        training values, labels for L1, labels for L2, object ids 
    """
    split_data = []
    for name, df_ in split_dfs:
        X = df_.drop([L1, L2, f"{L1}_pred", f"{L2}_pred", "object_id"], axis=1, errors='ignore')
        assert X.shape[1] == feature_values
        y = df_[[L1, L2, "object_id"]].values
        split_data.append({'X': X, 'y_2': y[:,1], 'y_1': y[:,0], 'object_id': y[:,2]})
    return split_data

def create_splits_L2(split_dfs, n_classes_L1, L1, L2, feature_values=282):
    """Splits the original df to training data and labels

    Parameters
    ----------
    split_dfs: List of Pandas DataFrames
        list of DFs split according to L1 and L2 (unique split for each unique combination)

    Returns
    -------
    list of dicts
        training values, labels for L1, labels for L2, object ids 
    """
    split_data = [[] for i in range(n_classes_L1)]
    for name, df_ in split_dfs:
        X = df_.drop([L1, L2, f"{L1}_pred", f"{L2}_pred", "object_id"], axis=1, errors='ignore')
        assert X.shape[1] == feature_values
        y = df_[[L1, L2, "object_id"]].values
        split_data[name[0]-1].append({'X': X, 'y_2': y[:,1], 'y_1': y[:,0], 'object_id': y[:,2]})
    return split_data

def get_split_info(split_dfs):
    """ Collects information about splits, used for knowing the number of objects in a split. 

    Parameters
    ----------
    split_dfs: List of Pandas DataFrames
        list of DFs split according to L1 and L2 (unique split for each unique combination)

    Returns
    -------
    split_info: List of Tuples
        Tuple: number of objects present in the split, L1 and L2 of the split
    """
    split_info = []
    for i, s_d in enumerate(split_dfs):
        n_obj = int(s_d.shape[0])
        if n_obj != 0:
            l1 = int(s_d["first_lvl_pivot_id_pred"].mode().values[0])
            l2 = int(s_d["second_lvl_pivot_id_pred"].mode().values[0])
        else:
            l1 = 0; l2 = 0
        split_info.append((n_obj, l1, l2))
    return split_info