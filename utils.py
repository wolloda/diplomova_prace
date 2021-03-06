from os import listdir
from os.path import isfile, join
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

def one_hot_frequency_encode(l1_labels, n_cats=18):
    frequency_labels = []
    for l in l1_labels:
        labels, counts = np.unique(l, return_counts=True)
        curr = np.zeros(n_cats)
        for l,c in zip(labels, counts):
            curr[int(l)-1] = c
        frequency_labels.append(curr)

    frequency_labels = np.vstack((frequency_labels))
    return frequency_labels

def label_encode_data(y):
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    return y, encoder

def label_encode_vectors(l1_labels):
    labels_unique = np.concatenate(np.array([np.unique(v) for v in l1_labels]), axis=0).astype(np.int64)
    le = LabelEncoder()
    le.fit(labels_unique)
    mapper_dict = dict(zip(labels_unique, le.transform(labels_unique)))
    def mp(entry):
        return mapper_dict[entry] if entry in mapper_dict else entry
    mp = np.vectorize(mp)
    encoded = [mp(v) for v in l1_labels]
    return le, encoded 

def get_class_weights_dict(y):
    return dict(enumerate(compute_class_weight('balanced', np.unique(y), y)))

def get_bucket_occupancy(df, labels, mtree_ratio=None, constant = 0.95):
    # TBD
    pass

def get_knn_objects(path="./queries.data", should_be_int=True):
    knn_object_ids = []
    with open(path) as f:
        for line in f.readlines():
            z_1 = re.findall(r"AbstractObjectKey ([\d\-_]+)", line)
            if z_1:
                if should_be_int:
                    knn_object_ids.append(int(z_1[0]))
                else:
                    knn_object_ids.append(z_1[0])
    if should_be_int:
        return np.array(knn_object_ids, dtype=np.int64)
    else:
        return np.array(knn_object_ids)

def get_sample_1k_objects(df_res, path = "/storage/brno6/home/tslaninakova/learned-indexes/datasets/queries.data"):
    should_be_int = True
    if path in ["/storage/brno6/home/tslaninakova/learned-indexes/datasets/mocap-queries.data", "/storage/brno6/home/tslaninakova/learned-indexes/datasets/mocap-queries-even.data", "/storage/brno6/home/tslaninakova/learned-indexes/datasets/mocap-queries-odd.data"]:
        should_be_int = False

    return df_res[df_res["object_id"].isin(get_knn_objects(path=path, should_be_int=should_be_int))]
