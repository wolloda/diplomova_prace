import json
import logging
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from skmultilearn.problem_transform import LabelPowerset, BinaryRelevance, ClassifierChain
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
from skmultilearn.adapt import MLkNN

logging.basicConfig(datefmt='%d-%m-%y %H:%M', format='%(asctime)-15s%(levelname)s: %(message)s', level=logging.INFO)

def make_2nn_from_30nn(gt_knns, n=2):
    gt_2nns = {}
    for key,values in gt_knns.items():
        s = {k: v for i, (k, v) in enumerate(sorted(values.items(), key=lambda item: item[1])) if i < n}
        gt_2nns[key] = s
    return gt_2nns

def get_1M_gts_all(f="1M-GT-30NN-all.json"):
    logging.info(f"Loading gts from {f}")
    with open(f) as json_file:
        gt_knns_all = json.load(json_file)
    return gt_knns_all

def create_multilabel_dataset(df, gt_2nns_all):
    L1_2NNs = []; L2_2NNs = []
    for i, o_id in enumerate(df["object_id"].values):
        if i % 10000 == 0:
            logging.info(f"Creating dataset, row: {i}")
        for i, (k,v) in enumerate(gt_2nns_all[str(o_id)].items()):
            if i != 0:
                L1_2NNs.append(df[df["object_id"] == int(k)]["L1"].values[0])
                L2_2NNs.append(df[df["object_id"] == int(k)]["L2"].values[0])
    df["L1_2NN"] = L1_2NNs
    df["L2_2NN"] = L2_2NNs
    df['L1_labels'] = df[['L1','L1_2NN']].apply(tuple, axis=1)
    df['L2_labels'] = df[['L2','L2_2NN']].apply(tuple, axis=1)
    return df

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

def one_hot_frequency_encode(l1_labels, n_cats=18):
    frequency_labels = []
    for l in l1_labels:
        labels, counts = np.unique(l, return_counts=True)
        curr = np.zeros(n_cats)
        for l,c in zip(labels, counts):
            curr[int(l)-1] = c
        frequency_labels.append(curr)

    #if len(frequency_labels)
    frequency_labels = np.vstack((frequency_labels))
    print(f"shape: {frequency_labels.shape}")
    if frequency_labels.shape[1] == 1:
        frequency_labels = np.hstack((frequency_labels, np.zeros(frequency_labels.shape)))
    return frequency_labels

def load_knn_multilabel_dataset(f="../knn-datasets/cophir-1M-2000-30nn.csv"):
    logging.info(f"Loading dataset from {f}")
    df = pd.read_csv(f)
    l1_labels = []
    for i in range(1, 9):
        if f"L{i}_labels" in df.columns:
            df[f"L{i}_labels"] =  df[f"L{i}_labels"].apply(lambda x: eval(x))
    #for l in df["L1_labels"].values:
    #    l1_labels.append(eval(l))
    #l2_labels = []
    #for l in df["L2_labels"].values:
    #    l2_labels.append(eval(l))
    #df["L1_labels"] = l1_labels
    #df["L2_labels"] = l2_labels
    #if "cophir-1M-200-" in f:
    #    l3_labels = []
    #    for l in df["L3_labels"].values:
    #        l3_labels.append(eval(l))
    #    df["L3_labels"] = l3_labels
    if "first_lvl_pivot_id" in df.columns:
        df.drop(["L1", "L2"], axis=1, errors="ignore", inplace=True)
    return df

def load_2nn_multilabel_dataset(f="nn-datasets/cophir-1M-2000-2nn.csv"):
    return load_knn_multilabel_dataset(k=2)

def one_hot_encode(data):
    logging.info(f"One-hot encoding data of shape {data.shape}")
    one_hot = MultiLabelBinarizer()
    one_hot_encoded_L1 = one_hot.fit_transform(data)
    return one_hot, one_hot_encoded_L1

def train_L1(groupby_df, df):
    stack_l1 = []; preds_l1 = []; obj_ids = []; probas = []
    one_hot_encoded_L2s = {}
    for name, group in groupby_df:
        obj_ids.extend(group["object_id"].values)
        X = group.drop(["L1","L2","L1_2NN","L2_2NN","L1_labels","L2_labels", "object_id", "L1_pred"], axis=1, errors="ignore").values
        assert X.shape[1] == 282
        logging.info(f"Fitting L1: {name} : {X.shape}")
        one_hot = MultiLabelBinarizer()
        one_hot_encoded_L2 = one_hot.fit_transform(np.array(group["L2_labels"].values))
        #print(one_hot_encoded_L2.shape); print(X.shape)
        one_hot_encoded_L2s[name] = one_hot
        #y = sparse.lil_matrix(one_hot_encoded_L2)
        #print(y[:10])
        c = LabelPowerset(LogisticRegression(solver='newton-cg'))
        #c = MLkNN()
        c.fit(X, one_hot_encoded_L2)
        stack_l1.append(c)
        proba = c.predict_proba(X)
        probas.append(proba)
        #preds_l1.extend(c.predict_proba(X))
        argmax_preds = [one_hot.classes_[np.argmax(p)] for p in proba.toarray()]
        preds_l1.extend(argmax_preds)
    df_l2 = pd.DataFrame(np.array([preds_l1, obj_ids]).T, columns=["L2_pred"] + ["object_id"])
    df_l1 = df.merge(df_l2, on="object_id")
    return stack_l1, one_hot_encoded_L2s, df_l1

def train_cophir_1M(df):
    one_hots = []; stack = [[],[]]; probas = []
    one_hot, one_hot_encoded_L1 = one_hot_encode(np.array(df["L1_labels"].values))
    root = LabelPowerset(LogisticRegression(solver="newton-cg"))
    #root = MLkNN()
    X = df.drop(["L1","L2","L1_2NN","L2_2NN","L1_labels","L2_labels", "object_id"], axis=1).values
    assert X.shape[1] == 282
    logging.info(f"Fitting root model, data: {X.shape}")
    root.fit(X, one_hot_encoded_L1)
    preds_proba = root.predict_proba(X); preds = root.predict(X)
    argmax_preds = [one_hot.classes_[np.argmax(p)] for p in preds_proba.toarray()]
    logging.info(f"Fitted and predicted, accuracy: {accuracy_score(one_hot_encoded_L1,preds)}")
    one_hots.append(one_hot); stack[0].append(root); probas.append(preds_proba)

    df_l1 = pd.DataFrame(np.array([argmax_preds, df["object_id"]]).T, columns=[f"L1_pred", "object_id"])
    #print(df_l1.head())
    df_root = df.merge(df_l1, on="object_id")

    stack_l1, one_hot_encoded_L2s, df_res = train_L1(df_root.groupby([f"L1_pred"]), df)
    stack[1] = stack_l1
    return stack, [one_hot_encoded_L1, one_hot_encoded_L2s], df_res