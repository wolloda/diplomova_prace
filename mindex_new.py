from imports import *
from parsing import *
import json
import time
import re
import math
from searching import get_classification_probs_per_level
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
import random
from tensorflow.keras.models import clone_model
from sklearn.utils import shuffle
import logging
logging.basicConfig(datefmt='%d-%m-%y %H:%M', format='%(asctime)-15s%(levelname)s: %(message)s', level=logging.INFO)
import warnings
from sklearn.preprocessing import LabelEncoder

from preprocessing import scale_per_descriptor
from sklearn.mixture import GaussianMixture


from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.optimizers import Adam
from data_handling import encode_data
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.exceptions import DataConversionWarning, ConvergenceWarning
from searching import approximate_search_mindex
from classification import get_classification_probs_per_level_new, estimate_distance_of_best_deepest_path, get_mindex_distance, get_wspd
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)
from tensorflow.keras.metrics import categorical_accuracy
from multilabel_training import one_hot_frequency_encode, label_encode_vectors
from network import construct_mlp, construct_fully_connected_model, construct_fully_connected_model_1M, to_categorical, get_baseline_model, compile_baseline, get_simple_baseline_model
from tensorflow.keras import backend as K
import tensorflow as tf
jobs = 10
tf.config.threading.set_intra_op_parallelism_threads(jobs)
tf.config.threading.set_inter_op_parallelism_threads(jobs)

def custom_softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    return numerator / denominator

class BarebonesLogisticRegression(LogisticRegression):

    def predict_proba_single(self, x):
        x = x[0]
        #val = np.dot(self.coef_, x.reshape(-1, 1)) + self.intercept_
        #print(self.intercept_.shape); print(np.dot(self.coef_, x).shape)#; print(val.shape)
        return custom_softmax(np.dot(self.coef_, x) + self.intercept_)

#from network import construct_fully_connected_model_1M
class MIndex(object):

    def __init__(self, PATH="/storage/brno6/home/tslaninakova/learning-indexes/mindex-knn/", mindex="mindex-cophir1M-leaf2000-dump/"):
        self.dir = PATH + mindex
        self.mindex = mindex
        self.knn_gts_file = f"{self.dir}/knn_gt.json"
        if "Profiset-leaf2000" in self.dir:
            self.labels = ["L1", "L2"]
        elif "leaf2000" in self.dir or "Profiset-leaf200" in self.dir:
            self.labels = ["L1", "L2", "L3", "L4", "L5", "L6"]
        else:
            self.labels = ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"]
        self.stack = []
        self.mapping = []
        self.encoders = []
        self.objects_in_buckets = {}
        self.class_encoders = []

    def get_descriptive_col_names_2(self):
        return ["L1", "L2"]

    def get_descriptive_col_names(self):
        if "leaf2000" in self.dir:
            return ["L1", "L2", "L3", "L4", "L5", "L6"]
        else:
            return ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"]

    def get_descriptive_col_names_pred(self):
        if "leaf2000" in self.dir:
            return ["L1", "L2", "L3", "L4", "L5", "L6", "L1_pred", "L2_pred", "L3_pred", "L4_pred", "L5_pred", "L6_pred"]
        else:
            return ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L1_pred", "L2_pred", "L3_pred", "L4_pred", "L5_pred", "L6_pred", "L7_pred", "L8_pred"]
            
    def train_LMI_level(self, df, groupby_df, param_dict, level=1, na_label=None):
        stack_l1 = []; preds_l1 = []; object_ids = []; mapping = []; y_enc = []
        y_label = self.labels[level]
        L2_max = df[y_label].max()
        #print(len(self.encoders))
        #for e in self.encoders:
        #    print(len(e))
        self.encoders.append([])
        for name, group in groupby_df:
            if "NNMult" in param_dict:
                labels_l = [f"L{l}_labels" for l in range(1, len(self.labels)+1)]
                X = group.drop(self.labels + [f"{l}_pred" for l in self.labels] + labels_l + ["object_id"], axis=1, errors='ignore').values
                y_based_on_pred = []
                prev_label = name
                if type(name) != int and type(name) != float:
                    prev_label = name[-1]
                prev_label = float(prev_label)
                max_y = 0
                for val_l1, val_l2 in zip(group[f"L{level}"+"_labels"].values, group[f"L{level+1}"+"_labels"].values):
                    pos = np.where(np.array(val_l1) == prev_label)
                    if pos[0].shape[0] != 0:
                        subarray = np.array(val_l2)[pos]
                        if subarray.max() > max_y:
                            max_y = subarray.max()
                        #if subarray.size != 0:
                        y_based_on_pred.append(subarray)
                    else:
                        subarray = np.array(val_l2)
                        if subarray.max() > max_y:
                            max_y = subarray.max()
                        y_based_on_pred.append(subarray)
                #encoder, encoded = label_encode_vectors(y_based_on_pred)
                #y = group[f"L{level}"+"_labels"].values
                y = y_based_on_pred

                y = one_hot_frequency_encode(y, n_cats=int(max_y))
            else:
                X = group.drop(self.labels + [f"{l}_pred" for l in self.labels] + [f"{l}_enc" for l in self.labels] + ["object_id"], axis=1, errors="ignore").values
                y = group[y_label].values
                #y = np.nan_to_num(arr, copy=True, nan=label_for_nan)
                #y_str = []
                #for prev_label, curr_label in zip(group[self.labels[level-1]].values, group[y_label].values):
                #    y_str.append(f"{int(prev_label)}.{int(curr_label)}")
                #le = LabelEncoder()
                #y = le.fit_transform(y_str)

            if X.shape[0] == 1:
                pass
            print(X.shape)
            assert X.shape[1] == 282 or X.shape[1] == 4096
            object_ids.extend(group["object_id"].values)
            orig_shape = None
            if "LogReg" in param_dict:
                logging.info(f"({name}) : L{level} LogReg on {X.shape} for {param_dict['LogReg'][level]['ep']} epochs")
                y, encoder = encode_data(y)
                if np.unique(y, return_counts=True)[1].shape[0] == 1:
                    #logging.info(f"{name}, not fitting, merging {name}.{y[0]}")
                    clf = DecisionTreeClassifier()
                    #d_class_weights = dict(enumerate(compute_class_weight('balanced', np.unique(y), y)))
                    #clf = BarebonesLogisticRegression(max_iter=param_dict["LogReg"][level]["ep"], class_weight=d_class_weights)
                    clf.fit(X, y)
                    preds = clf.predict(X)
                    #preds = [preds[0]]
                    #clf = None
                    #y, encoder = encode_data(np.array([np.nan]*len(y)))
                    #preds = [np.nan]*len(y)
                    preds = encoder.inverse_transform(preds)
                    #print(f"Passing {print(preds.shape)}")
                    logging.info(f"{name}, not fitting, merging {name}.{y[0]} | preds={preds}")
                    self.encoders[-1].append(encoder)
                else:
                    #d_class_weights = dict(enumerate(compute_class_weight('balanced', np.unique(y), y)))
                    clf = BarebonesLogisticRegression(max_iter=param_dict["LogReg"][level]["ep"])#, class_weight=d_class_weights)
                    clf.fit(X, y)
                    preds = clf.predict(X)
                    preds = encoder.inverse_transform(preds)
                    self.encoders[-1].append(encoder)
            elif "RF" in param_dict:
                y, encoder = encode_data(y)
                logging.info(f'({name}) : L{level} RF on {X.shape}: max_depth={param_dict["RF"][level]["depth"]} | n_est: {param_dict["RF"][level]["n_est"]}')
                clf = self.get_forest(max_depth=param_dict["RF"][level]["depth"], n_estimators=param_dict["RF"][level]["n_est"], n_jobs=10)
                clf.fit(X, y)
                preds = clf.predict(X)
                preds = encoder.inverse_transform(preds)
                self.encoders[-1].append(encoder)
            elif "NN" in param_dict:
                logging.info(f"({name}) : L{level} NN on {X.shape} with model= | opt={str(param_dict['NN'][level]['opt'])} | ep={param_dict['NN'][0]['ep']}")
                y, encoder = encode_data(y)
                if np.unique(y, return_counts=True)[1].shape[0] == 1:
                    logging.info("Training dec tree")
                    clf = DecisionTreeClassifier()
                    clf.fit(X, y)
                    preds = clf.predict(X)
                    preds = encoder.inverse_transform(preds)
                    #logging.info(f"{name}, not fitting, merging {name}.{y[0]} | preds={preds}")
                    self.encoders[-1].append(encoder)
                else:
                    clf = param_dict['NN'][level]['model'](input_data_shape=X.shape[1], output_data_shape=max(y)+1) #param_dict['NN'][level]['model'](output_data_shape=y.shape[1]) #clone_model(param_dict['NN'][level]['model'])
                    clf.compile(loss='sparse_categorical_crossentropy', metrics=[categorical_accuracy], optimizer=param_dict["NN"][level]["opt"])
                    clf.fit(X, y, epochs=param_dict["NN"][level]["ep"],  verbose=True)
                    preds = [np.argmax(p) for p in clf.predict(X)]
                    preds = encoder.inverse_transform(preds)
                    self.encoders[-1].append(encoder)
            elif "NNMult" in param_dict:
                #d_class_weights = dict(enumerate(compute_class_weight('balanced', np.unique(np.argmax(y, axis=1)), np.argmax(y, axis=1))))
                logging.info(f"({name}) : L{level} Multilabel | {X.shape} samples | model={param_dict['NNMult'][level]['model']().name} | opt={str(param_dict['NNMult'][level]['opt'])} | ep={param_dict['NNMult'][0]['ep']}")
                #print(y, y.shape, np.unique(y, return_counts=True)[1].shape[0] == 1)
                if y.shape[0] == 1:
                    y = np.argmax(y[0])
                    #y, encoder = encode_data([y])
                    #print(f"Original y shape: 1, {y} {y_label}")
                    clf = DecisionTreeClassifier()
                    clf.fit(X, [y])
                    preds = clf.predict(X)
                    #preds = [preds[0] + 1]
                    #preds = encoder.inverse_transform(preds)
                    preds = [preds[0] + 1]
                    #self.encoders[-1].append(encoder)
                    #print(preds)
                else:
                    clf = param_dict['NNMult'][level]['model'](input_data_shape=X.shape[1],output_data_shape=y.shape[1]) #clone_model(param_dict['NNMult'][level]['model'])
                    loss = "categorical_crossentropy"
                    #should_adjust_pred = False
                        #loss = "sparse_categorical_crossentropy"
                    clf.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer=param_dict["NNMult"][level]["opt"])
                    #y_1 = to_categorical(y)
                    #print(X.shape, y.shape)
                    #self.encoders[-1].append(encoder)
                    clf.fit(X, y, epochs=param_dict["NNMult"][level]["ep"], verbose=True)# class_weight=d_class_weights)
                    
                    preds_full = [np.argmax(p) for p in clf.predict(X)]
                    #print(f"classes: {encoder.classes_}")
                    #print(f"predictions: {preds_full}")
                    preds = [f+1 for f in preds_full]
                    #preds = encoder.inverse_transform(preds_full) + 1
                    #print(preds)
                    #preds = [o for p in preds]
                if na_label:
                    #print(f"here, before {preds}")
                    preds = np.array(preds, dtype=np.float64)
                    np.put(preds, np.where(preds == na_label), np.nan)
                    np.put(preds, np.where(preds == na_label+1), np.nan)
                    #print(f"here, after {preds}")
                        #print(preds)
                    #if should_adjust_pred:
                    #    preds = preds[0]
                    #    print(f"Final preds: {preds}")
                    #preds = encoder.inverse_transform(preds)
            elif "GMM" in param_dict:
                if np.unique(y, return_counts=True)[1].shape[0] == 1 and X.shape[0] == 1:
                    orig_shape = X.shape
                    #logging.info(f"{name}, not fitting, merging {name}.{y[0]}")
                    #clf = DecisionTreeClassifier()
                    #d_class_weights = dict(enumerate(compute_class_weight('balanced', np.unique(y), y)))
                    #clf = BarebonesLogisticRegression(max_iter=param_dict["LogReg"][level]["ep"], class_weight=d_class_weights)
                    #try:
                    #    clf.fit(X, y)
                    #    preds = clf.predict(X)
                    #except:
                    #    print(X); print(y)
                    X = np.vstack((X,X))
                    #preds = [preds[0]]
                    #clf = None
                    #y, encoder = encode_data(np.array([np.nan]*len(y)))
                    #preds = [np.nan]*len(y)
                    #preds = encoder.inverse_transform(preds)
                    #print(f"Passing {print(preds.shape)}")
                    #logging.info(f"{name}, not fitting, merging {name}.{y[0]} | preds={preds}")
                n_comp  = param_dict['GMM'][level]['comp']
                if X.shape[0] <= n_comp:
                    n_comp = X.shape[0] // 2
                logging.info(f"({name}) :GMM | n_comp={n_comp}")
                #clf = GaussianMixture(n_components=n_comp, covariance_type='diag', max_iter=1)
                clf = GaussianMixture(n_components=n_comp, covariance_type='spherical', init_params='kmeans',max_iter=1)
                #try:
                clf.fit(X)
                preds = clf.predict(X)
                if np.unique(y, return_counts=True)[1].shape[0] == 1:
                    preds = [preds[0]]
                if orig_shape:
                    preds = preds[:orig_shape[0]]
                while len(preds) < group["object_id"].values.shape[0]:
                    preds.append(preds[-1])
                #print(group["object_id"].values.shape[0], len(preds))
                assert group["object_id"].values.shape[0] == len(preds)
                #print(len(preds), orig_shape)
                    #print(len(y), len(preds))
                #except:
                #print(len(preds)); print(len(y))
                #    preds = y

            if clf:
                if type(name) is float or type(name) is int:
                    mapping.append([int(name)])
                else:
                    mapping.append([int(n) for n in name])
                stack_l1.append(clf)
            #preds[preds == label_for_nan]=np.nan
            #if len(self.encoders) == 3:
            #    for e in self.encoders[-1]:
            #        print(e.classes_)
            preds_l1.extend(preds)
            #self.class_encoders.append(le)
            #y_enc.extend(y)

        print(len(preds_l1), len(object_ids))
        df_l2 = pd.DataFrame(np.array([preds_l1, object_ids]).T, columns=[y_label+"_pred"] + ["object_id"])
        df_l2[y_label+"_pred"] = df_l2[y_label+"_pred"].astype('float')
        #df_l2[y_label+"_enc"] = y_enc
        df_l1 = df.merge(df_l2, on="object_id")
        return stack_l1, df_l1, mapping


    def train_LMI(self, df, param_dict, pretrained_root=False, na_label=None, return_root=False, return_input=False, alg2=False):
        self.stack = []; self.mapping = []; self.encoders = []; self.objects_in_buckets = {}
        df_ = df.sample(frac=1)
        encoded = None
        if "NNMult" in param_dict:
            df_ = df
            labels = [f"L{i}_labels" for i in range(1, len(self.labels)+1)]
            X = df_.drop(self.labels + ["object_id"] + labels, axis=1).values
            #print(df_[labels[0]].values)
            encoder, encoded = label_encode_vectors(df_[labels[0]].values)
            #print(encoder.classes_)
            #print(f"Original n_cats={df_[self.labels[0]].max()}, current n_cats={len(encoder.classes_)}")

            #y = one_hot_frequency_encode(df_[labels[0]], n_cats=df_[self.labels[0]].max())
            y = one_hot_frequency_encode(encoded, n_cats=len(encoder.classes_))
        else:
            X = df_.drop(self.labels + ["object_id"], axis=1).values
            y = df_[self.labels[0]].values

        y_obj_id = df_["object_id"].values
        
        if "RF" in param_dict:
            y, encoder = encode_data(y)
            logging.info(f'Training root RF model on {X.shape}: max_depth={param_dict["RF"][0]["depth"]} | n_est: {param_dict["RF"][0]["n_est"]}')
            root = self.get_forest(max_depth=param_dict["RF"][0]["depth"], n_estimators=param_dict["RF"][0]["n_est"], n_jobs=10)
            assert X.shape[1] == 282 or X.shape[1] == 4096
            root.fit(X, y)
            preds = root.predict(X)
            preds = encoder.inverse_transform(preds)
            self.encoders.append([encoder])
        elif "LogReg" in param_dict:
            y, encoder = encode_data(y)
            #d_class_weights = dict(enumerate(compute_class_weight('balanced', np.unique(y), y)))
            logging.info(f'Training LogReg model with epochs={param_dict["LogReg"][0]["ep"]} epochs')
            root = BarebonesLogisticRegression(max_iter=param_dict["LogReg"][0]["ep"])#, class_weight=d_class_weights)
            root.fit(X, y)
            preds = root.predict(X)
            preds = encoder.inverse_transform(preds)
            self.encoders.append([encoder])
        elif "NN" in param_dict:
            y, encoder = encode_data(y)
            #d_class_weights = dict(enumerate(compute_class_weight('balanced', np.unique(y), y)))
            logging.info(f'Training NN with model, optimizer: {str(param_dict["NN"][0]["opt"])} and epochs: {param_dict["NN"][0]["ep"]}')
            print(X.shape, y.shape)
            if not pretrained_root:
                root = param_dict["NN"][0]["model"]
                root.compile(loss='categorical_crossentropy', metrics=['categorical_accuracy'], optimizer=param_dict["NN"][0]["opt"])
                y = to_categorical(y)
                #K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_‌​parallelism_threads=10,inter_op_parallelism_threads=10)))
                root.fit(X, y, epochs=param_dict["NN"][0]["ep"])#, class_weight=d_class_weights)
            else:
                root = param_dict["NN"][0]["model"]
            #root = load_model("../tmp/class-weights-100ep-adam0001.h5")
            preds = root.predict(X)
            preds = [np.argmax(p) for p in preds]
            preds = encoder.inverse_transform(preds)
            if na_label:
                preds = np.array(preds, dtype=np.float64)
                np.put(preds, np.where(preds == na_label), np.nan)
            self.encoders.append([encoder])
        elif "NNMult" in param_dict:
            #print(y)
            #d_class_weights = dict(enumerate(compute_class_weight('balanced', np.unique(np.argmax(y, axis=1)),np.argmax(y, axis=1))))
            #print(np.unique(np.argmax(y, axis=1)))
            #d_class_weights = dict(enumerate(compute_class_weight('balanced', np.unique(np.argmax(y, axis=1)),np.argmax(y, axis=1))))
            logging.info(f'Training NN with model: {param_dict["NNMult"][0]["model"]}, optimizer: {str(param_dict["NNMult"][0]["opt"])} and epochs: {param_dict["NNMult"][0]["ep"]}')
            root = param_dict["NNMult"][0]["model"](input_data_shape=X.shape[1],output_data_shape=y.shape[1])
            root.compile(loss='categorical_crossentropy', metrics=[categorical_accuracy], optimizer=param_dict["NNMult"][0]["opt"])
            #print(y.shape)
            self.encoders.append([encoder])
            if return_input:
                return X,y
            root.fit(X, y, epochs=param_dict["NNMult"][0]["ep"]) #, class_weight=d_class_weights)
            preds_full = [np.argmax(p) for p in root.predict(X)]
            #print(preds_full)
            preds = encoder.inverse_transform(preds_full) + 1
            #print(preds)
            #print(preds)
            if na_label:
                preds = np.array(preds, dtype=np.float64)
                np.put(preds, np.where(preds == na_label), np.nan)
        elif "GMM" in param_dict:
            logging.info(f'GMM | n_comp={param_dict["GMM"][0]["comp"]}')
            #root = GaussianMixture(n_components=param_dict["GMM"][0]["comp"], covariance_type='diag', max_iter=1)
            root = GaussianMixture(n_components=param_dict["GMM"][0]["comp"], init_params='kmeans', covariance_type='spherical', max_iter=1)
            root.fit(X)
            preds = root.predict(X)
        #print("preds"); print(preds); print(preds.shape)
        df_res_ls = []
        df_l1 = pd.DataFrame(np.array([preds, y_obj_id]).T, columns=[f"{self.labels[0]}_pred", "object_id"])
        df_res = df_.merge(df_l1, on="object_id")

        # B4
        if alg2:
            df_res_L1_corr = df_res[df_res[self.labels[0]] == df_res[self.labels[0]+"_pred"]]
            print(df_res_L1_corr.shape)
            df_res = df_res_L1_corr
            
        #df_res_ls.append(df_res[df_res["L2"].isna()])
        #df_res = df_res[~df_res["L2"].isna()]
        if return_root:
            return df_res
        self.stack.append(root)
        
        #group_cond = [f"{self.labels[i]}_pred" for i in range(i+1)]
        group_cond = [f"{self.labels[0]}_pred"]
        groups = df_res.groupby(group_cond)
        for i, label in enumerate(self.labels[1:]):
            print(f"df_res: {df_res.shape}")
            
            print(f"group_cond: {group_cond}")
            #groups = df_res[(~df_res[label].isna()) & (df_res[label] != na_label)].groupby(group_cond)

            stack_l1, df_l1, mapping_l1 = self.train_LMI_level(df_res, groups, param_dict, level=i+1, na_label=na_label)
            #df_res = df_res.merge(df_l1[["object_id", f"{label}_enc", f"{label}_pred"]], on=["object_id"], how="left")
            df_res = df_res.merge(df_l1[["object_id", f"{label}_pred"]], on=["object_id"], how="left")

            #print(df_res[df_res[label + "_pred"].isna()].shape)
            df_res_ls.append(df_res[df_res[label + "_pred"].isna()])
            
            if alg2:
                df_res = df_res[df_res[label] == df_res[label+"_pred"]]

            #df_res_ls.append(df_res)#df_res[df_res[label + "_pred"].isna()])
            df_res = df_res[~(df_res[label+"_pred"].isna())].copy()
            #print(df_res.shape)
            group_cond.append(label+"_pred") #[f"{self.labels[i]}_pred" for i in range(i+1)]
            groups = df_res.groupby(group_cond)

            #print(i, df_res.shape)
            self.mapping.append(mapping_l1)
            self.stack.append(stack_l1)

        df_res_ls.append(df_res[~(df_res[label + "_pred"].isna())])
        groupbys = []
        
        for i, df_res_l in enumerate(df_res_ls):
            cond = []
            for j in range(i+1):
                cond.append(f"L{j+1}_pred")
            groupbys.append(df_res_l.groupby(cond))
        
        #return groupbys
        
        for gb in groupbys:
            for name, g in gb:
                #self.objects_in_buckets[name] = g.shape[0]
                print(name)
                if type(name) is not int and type(name) is not float:
                    name_str = ""
                    for n in name:
                        name_str += f"{int(n)}."
                    name_str = name_str[:-1]
                    self.objects_in_buckets[name_str] = g.shape[0]
                else:
                    self.objects_in_buckets[f"{int(name)}"] = g.shape[0]

        return pd.concat(df_res_ls)

    train = train_LMI

    def post_train_misclassified(self, df, df_res, is_multi=False):
        preds_offset = 0; na_label = 150
        if len([c for c in df_res.columns if type(c) is str and c[-7:] == "_labels"]) != 0 or is_multi:
            labels = self.labels
            df_res.drop([f"{l}_labels" for l in labels], axis=1, errors='ignore', inplace=True)
            df.drop([f"{l}_labels" for l in labels], axis=1, errors='ignore', inplace=True)
            preds_offset = 1
        #print(preds_offset)
        df_diff = pd.concat([df,df_res.drop([f"{l}_pred" for l in self.labels], axis=1)]).drop_duplicates(keep=False)
        X = df_diff.drop(self.labels + ["object_id"], axis=1).values
        X_pred_miss = self.stack[0].predict(X)
        if len(X_pred_miss.shape) != 1:
            X_pred_miss = [np.argmax(p) for p in X_pred_miss]
        if self.encoders[0] != []:
            X_pred_miss = self.encoders[0][0].inverse_transform(X_pred_miss)
        #print(X_pred_miss)
        if preds_offset == 1:
            X_pred_miss = [X_pred_miss_+preds_offset if X_pred_miss_+preds_offset <= df[self.labels[0]].max() else df[self.labels[0]].max() for X_pred_miss_ in X_pred_miss]
        df_diff[f"{self.labels[0]}_pred"] = X_pred_miss
        for level in range(len(self.labels)-1):
            print(level)
            preds_l = []; object_ids = []
            group_cond = [f"{l}_pred" for l in self.labels[:level+1]]
            for n,g in df_diff.groupby(group_cond):
                #print(n,g.shape)
                if type(n) is int or type(n) is float:
                    idx_label = [n]
                else:
                    idx_label = [int(n_) for n_ in n]
                if idx_label in self.mapping[level]:
                    #print(f"Here | level={level} idx_label={idx_label}")
                    stack_index = self.mapping[level].index(idx_label)
                    preds = self.stack[level+1][stack_index].predict(g.drop(self.labels + [f"{l}_pred" for l in self.labels] + ["object_id"], errors='ignore', axis=1).values)
                    if len(preds.shape) != 1:
                        preds = [np.argmax(p) for p in preds]
                    #print(preds)
                    if preds_offset == 1:
                        preds = [preds_+preds_offset for preds_ in preds]
                        preds = np.array(preds, dtype=np.float64)
                        np.put(preds, np.where(preds == na_label), np.nan)
                        np.put(preds, np.where(preds == na_label+1), np.nan)
                    else:
                        #print(level, stack_index, n, preds)
                        if len(self.encoders) > level+1 and self.encoders[level+1] != []:
                            preds = self.encoders[level+1][stack_index].inverse_transform(preds)

                    preds_l.extend(preds)
                    object_ids.extend(g["object_id"].values)
            df_l = pd.DataFrame(np.array([preds_l, object_ids]).T, columns=[self.labels[level+1]+"_pred"] + ["object_id"])
            df_diff = df_diff.merge(df_l, on="object_id", how="left")
            #print(f"level: {level}, shape: {df_diff.shape}")
        
        return df_diff
    def add_missclassified_to_buckets(self, df_res, df_diff):
        not_in_buckets = []
        for name in df_diff[[f"{l}_pred" for l in self.labels]].values:
            
            name_str = ""
            for n in name:
                if str(n) != "nan":
                    name_str += f"{int(n)}."
            name_str = name_str[:-1]
            if name_str in self.objects_in_buckets:
                self.objects_in_buckets[name_str] += 1
            else:
                self.objects_in_buckets[name_str] = 0
                if name_str not in not_in_buckets:
                    not_in_buckets.append(name_str)

        for unknown_bucket in not_in_buckets:
            self.objects_in_buckets[unknown_bucket] += 1

        print(f"{len(not_in_buckets)} buckets that were not in original buckets")
        print(f"Objects in li.objects_in_buckets: {sum(self.objects_in_buckets.values())}")
        return pd.concat([df_res, df_diff])

    def get_objects_in_popped_bucket(self, popped, df_res, is_orig=False):
        
        labels = [label for i, label in enumerate(popped.split(".")) if i >= 2]
        if not is_orig:
            pred_labels = [f"{l}_pred" for l in self.labels]
        else:
            pred_labels = [l for l in self.labels]
            
        if len(labels) == 6 and len(self.labels) == 6:
            if not is_orig:
                leaf_df_res = df_res[~df_res[f"L6_pred"].isna()]
            else:
                leaf_df_res = df_res[~df_res[f"L6"].isna()]
        elif len(labels) == 8 and len(self.labels) == 8:
            if not is_orig:
                leaf_df_res = df_res[~df_res[f"L8_pred"].isna()]
            else:
                leaf_df_res = df_res[~df_res[f"L8"].isna()]
        elif len(labels) == 2 and len(self.labels) == 2:
            if not is_orig:
                leaf_df_res = df_res[~df_res[f"L2_pred"].isna()]
            else:
                leaf_df_res = df_res[~df_res[f"L2"].isna()]
        else:
            if not is_orig:
                leaf_df_res = df_res[df_res[f"L{len(labels)+1}_pred"].isna()]
            else:
                leaf_df_res = df_res[df_res[f"L{len(labels)+1}"].isna()]
        if len(labels) == 1:
            n_objects = leaf_df_res[(leaf_df_res[pred_labels[0]] == int(labels[0]))].shape[0]
        elif len(labels) == 2:
            n_objects = leaf_df_res[(leaf_df_res[pred_labels[0]] == int(labels[0])) & (leaf_df_res[pred_labels[1]] == int(labels[1]))].shape[0]
        elif len(labels) == 3:
            n_objects = leaf_df_res[(leaf_df_res[pred_labels[0]] == int(labels[0])) & (leaf_df_res[pred_labels[1]] == int(labels[1])) & (leaf_df_res[pred_labels[2]] == int(labels[2]))].shape[0]
        elif len(labels) == 4:
            n_objects = leaf_df_res[(leaf_df_res[pred_labels[0]] == int(labels[0])) & (leaf_df_res[pred_labels[1]] == int(labels[1])) & (leaf_df_res[pred_labels[2]] == int(labels[2])) & (leaf_df_res[pred_labels[3]] == int(labels[3]))].shape[0]
        elif len(labels) == 5:
            n_objects = leaf_df_res[(leaf_df_res[pred_labels[0]] == int(labels[0])) & (leaf_df_res[pred_labels[1]] == int(labels[1])) & (leaf_df_res[pred_labels[2]] == int(labels[2])) & (leaf_df_res[pred_labels[3]] == int(labels[3])) & (leaf_df_res[pred_labels[4]] == int(labels[4]))].shape[0]
        elif len(labels) == 6:
            n_objects = leaf_df_res[(leaf_df_res[pred_labels[0]] == int(labels[0])) & (leaf_df_res[pred_labels[1]] == int(labels[1])) & (leaf_df_res[pred_labels[2]] == int(labels[2])) & (leaf_df_res[pred_labels[3]] == int(labels[3])) & (leaf_df_res[pred_labels[4]] == int(labels[4]))  & (leaf_df_res[pred_labels[5]] == int(labels[5]))].shape[0]
        elif len(labels) == 7:
            n_objects = leaf_df_res[(leaf_df_res[pred_labels[0]] == int(labels[0])) & (leaf_df_res[pred_labels[1]] == int(labels[1])) & (leaf_df_res[pred_labels[2]] == int(labels[2])) & (leaf_df_res[pred_labels[3]] == int(labels[3])) & (leaf_df_res[pred_labels[4]] == int(labels[4]))  & (leaf_df_res[pred_labels[5]] == int(labels[5])) & (leaf_df_res[pred_labels[6]] == int(labels[6]))].shape[0]
        elif len(labels) == 8:
            n_objects = leaf_df_res[(leaf_df_res[pred_labels[0]] == int(labels[0])) & (leaf_df_res[pred_labels[1]] == int(labels[1])) & (leaf_df_res[pred_labels[2]] == int(labels[2])) & (leaf_df_res[pred_labels[3]] == int(labels[3])) & (leaf_df_res[pred_labels[4]] == int(labels[4]))  & (leaf_df_res[pred_labels[5]] == int(labels[5])) & (leaf_df_res[pred_labels[6]] == int(labels[6]))  & (leaf_df_res[pred_labels[7]] == int(labels[7]))].shape[0]

        #print(f"n_objects: {n_objects}")
        return n_objects

    def search(self, df_res, object_id, stop_cond_models=None, stop_cond_objects=None, custom_classifier=None, use_encoders=True, debug=False):
        s = time.time()
        row = df_res[(df_res['object_id'] == object_id)]
        n_steps = 0
        iterations = 0
        time_checkpoints = []; popped_nodes_checkpoints = []; steps_checkpoints = []
        x = row.drop((self.labels + [f"{l}_pred" for l in self.labels] + ["object_id"]), axis=1, errors='ignore').values
        #gts = row[self.labels].values[0]
        enc = None
        if len(self.encoders[0]) != 0:
            enc = self.encoders[0][0]
        #s = time.time()
        l1 = get_classification_probs_per_level(x, self.stack[0], encoder=enc, custom_classifier=custom_classifier)
        #print(time.time() - s)
        priority_q = [{'M.1': 1.0}]
        if debug:
            print(f"Step 0: M.1 added - PQ: {priority_q}\n")
        priority_q = self.add_level_to_queue(priority_q, l1)
        #return priority_q
        #return priority_q
        if debug:
            print(f"Step 1: L1 added - PQ: {priority_q}, ...\n")
        times_processed = 0
        times_bucket = 0
        current_stop_cond_idx = 0
        popped_nodes = []
        processed_ = []
        indexes = []
        #return
        #print("Here")
        i = 0
        while len(priority_q) != 0:
            #if iterations % 10000 == 0:
            #    print(iterations, n_steps)
            if stop_cond_models != None and len(stop_cond_models) == current_stop_cond_idx:
                return {'id': object_id, 'time_checkpoints': time_checkpoints, 'popped_nodes_checkpoints': popped_nodes_checkpoints, 'steps_checkpoints': steps_checkpoints}
            if stop_cond_objects != None and len(stop_cond_objects) == current_stop_cond_idx:
                #print(times_processed, iterations)
                return {'id': object_id, 'time_checkpoints': time_checkpoints, 'popped_nodes_checkpoints': popped_nodes_checkpoints, 'steps_checkpoints': steps_checkpoints}
            else:
                if debug:
                    print(f"Step {iterations + 2} - Model visit {iterations + 1}: ")
                #s = time.time()
                #priority_q, popped, dist 
                priority_q, popped = self.process_node(priority_q, x, custom_classifier=custom_classifier, debug=debug)
                #print(len(res))
                #priority_q = res[0]; popped = res[1]; dist = res[2]
                #print(res[0], res[1], res[2])
                if popped is None:
                    continue
                #processed_.append(dist)
                #print(priority_q)
                times_processed += time.time()-s
                #print(popped)
                if stop_cond_objects is not None:
                    #s = time.time()
                    index = tuple([int(p) for p in popped.split('.')[2:]])
                    if len(index) == 1:
                        index = str(index[0])
                    else:
                        index_str = ""
                        for n in index:
                            index_str += f"{int(n)}."
                        index_str = index_str[:-1]
                        index = index_str
                    if index in self.objects_in_buckets:
                        n_obj = self.objects_in_buckets[index]
                        indexes.append(index)
                        if debug: print(f"checked key {index}, found {n_obj} objects")
                    else:
                        if debug: print(f"checked key {index}, found 0 objects")
                        n_obj = 0
                    #times_bucket += time.time() - s
                    if n_obj != 0:
                        popped = "C" + popped[1:]
                    #times_processed += time.time()-s
                    if type(popped) is list:
                        popped_nodes.extend(popped)
                    else: popped_nodes.append(popped)
                    n_steps += n_obj
                    #print(f"searching in objs: {time.time()-s}")
                    #print(f"Current n. of objects found: {n_steps}")
                    if current_stop_cond_idx < len(stop_cond_objects) and stop_cond_objects[current_stop_cond_idx] <= n_steps:
                        time_checkpoint = time.time()
                        time_checkpoints.append(time_checkpoint-s)
                        popped_nodes_checkpoints.append(popped_nodes.copy())
                        steps_checkpoints.append(n_steps)
                        current_stop_cond_idx += 1
            iterations += 1
        #print(times_processed)
        return indexes, {'id': object_id, 'steps': n_steps, 'time_checkpoints': time_checkpoints, 'popped_nodes_checkpoints': popped_nodes_checkpoints, 'steps_checkpoints': steps_checkpoints}

    def process_node(self, priority_q, x, custom_classifier=None, debug=False):
        popped = priority_q.pop(0)
        for key, value in popped.items():
            processed_node = key
            node_value = value
        processed_node = list(popped.keys())[0]
        if debug:
            print(f"Popped {processed_node}")
        model_label = processed_node.split('.')
        #print(priority_q, processed_node, node_value)
        if model_label[-1] == "nan":
            return priority_q, None

        s = time.time()
        if len(model_label) == 3:
            #if len(self.stack[1]) > int(model_label[(-1)])-1:
            preds_index = [int(model_label[-1])]
            if preds_index in self.mapping[0]:
                stack_index = self.mapping[0].index(preds_index)
                model = self.stack[1][stack_index]
                if len(self.encoders) > 1 and len(self.encoders[1]) > stack_index: 
                    encoder = self.encoders[1][stack_index] 
                else: encoder = None
                #print(stack_index, encoder)
                probs = get_classification_probs_per_level(x, model, encoder=encoder, custom_classifier=custom_classifier, value='value_l2')
                priority_q = self.add_level_to_queue(priority_q, probs, (model_label[(-1)]), parent_value=node_value, pop=False)
            #print(gts, np.isnan(gts[2]))
            #if np.isnan(gts[2]):
            #    processed_node =  "C" + processed_node[1:]
        elif len(model_label) == 4:
            if len(self.labels) >= 3:# and not np.isnan(gts[2]):
                #if len(self.stack[2]) > int(model_label[(-1)])-1:
                preds_index = None
                preds_index = [int(model_label[-2]), int(model_label[-1])]
                if preds_index in self.mapping[1]:
                    stack_index = self.mapping[1].index(preds_index)
                    #print(stack_index)
                    model = self.stack[2][stack_index]
                    #if "DecisionTreeClassifier" in str(type(model)):
                    #    return (priority_q, None)
                    #print(stack_index)
                    if len(self.encoders) > 2 and len(self.encoders[2]) > stack_index: 
                        encoder = self.encoders[2][stack_index] 
                    else: encoder = None
                    #print(encoder)
                    #print(encoder)
                    probs = get_classification_probs_per_level(x, model, encoder=encoder, custom_classifier=custom_classifier, value='value_l3')
                    priority_q = self.add_level_to_queue(priority_q, probs, (model_label[(-2)]), (model_label[(-1)]), parent_value=node_value, pop=False)
                #print(gts, np.isnan(gts[3]))
                #if np.isnan(gts[3]):
                #    processed_node =  "C" + processed_node[1:]
            #else:
            #    processed_node =  "C" + processed_node[1:]

        elif len(model_label) == 5:
            if len(self.labels) >= 4:# and not np.isnan(gts[3]):
                #if len(self.stack[3]) > int(model_label[(-1)])-1:
                preds_index = None
                preds_index = [int(model_label[-3]), int(model_label[-2]), int(model_label[-1])]
                if preds_index in self.mapping[2]:
                    stack_index = self.mapping[2].index(preds_index)
                    model = self.stack[3][stack_index]
                    if len(self.encoders) > 3 and len(self.encoders[3]) > stack_index: 
                        encoder = self.encoders[3][stack_index] 
                    else: encoder = None
                    if encoder and len(encoder.classes_) == 1 and str(encoder.classes_[0]) == "nan":
                        return (priority_q, "C" + processed_node[1:])
                    probs = get_classification_probs_per_level(x, model, encoder=encoder, custom_classifier=custom_classifier, value='value_l4')
                    #print(probs)
                    priority_q = self.add_level_to_queue(priority_q, probs, (model_label[(-3)]), (model_label[(-2)]), (model_label[(-1)]), parent_value=node_value, pop=False)
                    #print(priority_q)
                #if np.isnan(gts[4]):
                #    processed_node =  "C" + processed_node[1:]
            #else:
            #    processed_node =  "C" + processed_node[1:]
    
        elif len(model_label) == 6:
            if len(self.labels) >= 5:# and not np.isnan(gts[4]):
                #if len(self.stack[4]) > int(model_label[(-1)])-1:
                preds_index = None
                preds_index = [int(model_label[-4]), int(model_label[-3]), int(model_label[-2]), int(model_label[-1])]
                if preds_index in self.mapping[3]:
                    stack_index = self.mapping[3].index(preds_index)
                    model = self.stack[4][stack_index]
                    if len(self.encoders) > 4 and len(self.encoders[4]) > stack_index: 
                        encoder = self.encoders[4][stack_index] 
                    else: encoder = None
                    if encoder and len(encoder.classes_) == 1 and str(encoder.classes_[0]) == "nan":
                        return (priority_q, "C" + processed_node[1:])
                    probs = get_classification_probs_per_level(x, model, encoder=encoder, custom_classifier=custom_classifier, value='value_l5')
                    priority_q = self.add_level_to_queue(priority_q, probs, (model_label[(-4)]), (model_label[(-3)]), (model_label[(-2)]), (model_label[(-1)]), parent_value=node_value, pop=False)
                #if np.isnan(gts[5]):
                #    processed_node =  "C" + processed_node[1:]
            #else:
            #    processed_node =  "C" + processed_node[1:]

        elif len(model_label) == 7:
            if len(self.labels) >= 6:# and not np.isnan(gts[5]):
                #if len(self.stack[5]) > int(model_label[(-1)])-1:
                preds_index = None
                preds_index = [int(model_label[-5]), int(model_label[-4]), int(model_label[-3]), int(model_label[-2]), int(model_label[-1])]
                if preds_index in self.mapping[4]:
                    stack_index = self.mapping[4].index(preds_index)
                    model = self.stack[5][stack_index]
                    if len(self.encoders) > 5 and len(self.encoders[5]) > stack_index: 
                        encoder = self.encoders[5][stack_index] 
                    else: encoder = None
                    if encoder and len(encoder.classes_) == 1 and str(encoder.classes_[0]) == "nan":
                        return (priority_q, "C" + processed_node[1:])
                    probs = get_classification_probs_per_level(x, model, encoder=encoder, custom_classifier=custom_classifier, value='value_l6')
                    priority_q = self.add_level_to_queue(priority_q, probs, (model_label[(-5)]), (model_label[(-4)]), (model_label[(-3)]), (model_label[(-2)]), (model_label[(-1)]), parent_value=node_value, pop=False)

        elif len(model_label) == 8:
            if len(self.labels) >= 7:# and not np.isnan(gts[5]):
               # if len(self.stack[6]) > int(model_label[(-1)])-1:
                preds_index = None
                preds_index = [int(model_label[-6]), int(model_label[-5]), int(model_label[-4]), int(model_label[-3]), int(model_label[-2]), int(model_label[-1])]
                if preds_index in self.mapping[5]:
                    stack_index = self.mapping[5].index(preds_index)
                    model = self.stack[6][stack_index]
                    if len(self.encoders) > 6 and len(self.encoders[6]) > stack_index: 
                        encoder = self.encoders[6][stack_index] 
                    else: encoder = None
                    if encoder and len(encoder.classes_) == 1 and str(encoder.classes_[0]) == "nan":
                        return (priority_q, "C" + processed_node[1:])
                    probs = get_classification_probs_per_level(x, model, encoder=encoder, custom_classifier=custom_classifier, value='value_l7')
                    priority_q = self.add_level_to_queue(priority_q, probs, (model_label[(-6)]), (model_label[(-5)]), (model_label[(-4)]), (model_label[(-3)]), (model_label[(-2)]), (model_label[(-1)]), parent_value=node_value, pop=False)

        elif len(model_label) == 9:
            if len(self.labels) >= 8:# and not np.isnan(gts[5]):
                #if len(self.stack[7]) > int(model_label[(-1)])-1:
                preds_index = None
                preds_index = [int(model_label[-7]), int(model_label[-6]), int(model_label[-5]), int(model_label[-4]), int(model_label[-3]), int(model_label[-2]), int(model_label[-1])]
                if preds_index in self.mapping[6]:
                    stack_index = self.mapping[6].index(preds_index)
                    model = self.stack[7][stack_index]
                    if len(self.encoders) > 7 and len(self.encoders[7]) > stack_index: 
                        encoder = self.encoders[7][stack_index] 
                    else: encoder = None
                    if encoder and len(encoder.classes_) == 1 and str(encoder.classes_[0]) == "nan":
                        return (priority_q, "C" + processed_node[1:])
                    probs = get_classification_probs_per_level(x, model, encoder=encoder, custom_classifier=custom_classifier, value='value_l8')
                    priority_q = self.add_level_to_queue(priority_q, probs, (model_label[(-7)]), (model_label[(-6)]), (model_label[(-5)]), (model_label[(-4)]), (model_label[(-3)]), (model_label[(-2)]), (model_label[(-1)]),parent_value=node_value, pop=False)


                #if np.isnan(gts[6]):
                #    processed_node =  "C" + processed_node[1:]
        else:
            processed_node =  "C" + processed_node[1:]
        #logging.info(f"{processed_node} - {list(popped.values())[0]}")
        #print(time.time() -s )
        #elif len(self.labels) >= 3 and len(model_label) == 5:
        #    processed_node =  "C" + processed_node[1:]
        #t = time.time() - s
        #print(f"{t},")
        #s = time.time()
        priority_q = sorted(priority_q, key=(lambda i: list(i.values())), reverse=True)
        #t = time.time() - s
        #print(f"sorting: {time.time()-s}")
        if debug:
            #logging.info(f"L[2-3] added - PQ: {priority_q}\n")
            print(f"L[2-3] added - PQ: {priority_q}\n")
        return priority_q, processed_node

    def search_mindex(self, df_orig, query_id, struct_df, regions, bucket_level, stop_cond_objects=None, is_profi=False, objects_labels=None, debug=False):
        visited_models = []
        query_row = df_orig[df_orig["object_id"] == query_id]
        pivot_ids = struct_df["object_id"].values
        pivot_descriptors = struct_df.drop(["object_id"], axis=1).values
        pow_list = [pow(0.75, i) for i in range(8)]
        labels = self.labels + ["object_id"]
        distances = []
        start = time.time()
        pivot_distances_fixed = get_mindex_distance(pivot_ids, pivot_descriptors, query_row, labels, is_profi=is_profi, objects_labels=objects_labels)
        pivot_distances = pivot_distances_fixed.copy()
        pivot_distances_normalized = estimate_distance_of_best_deepest_path(pivot_distances, pow_list, bucket_level)
        priority_queue = pivot_distances_normalized.copy()
        #distances.extend(pivot_distances_fixed)
        #return distances
        total = 0; last_n_steps = 0
        time_checkpoints = []; popped_nodes_checkpoints = []; steps_checkpoints = []; steps = []; indexes = []
        other_nodes_all = []; find_bucket=True
        current_stop_cond_idx = 0; n_steps = 0
        while len(visited_models) > 0 or n_steps == 0:
            if stop_cond_objects != None and len(stop_cond_objects) == current_stop_cond_idx:
                return distances, {'id': query_id, 'time_checkpoints': time_checkpoints, 'popped_nodes_checkpoints': popped_nodes_checkpoints, 'steps_checkpoints': steps_checkpoints}
            
            #print(priority_queue)
            popped = priority_queue.pop(0)
            if not popped[0] in visited_models:
                if debug: print(f"Popped: {popped}")
                visited_models.append(popped[0])
                distances.extend([[popped[0], popped[1]]])
                #try:
                #    assert ['21', 239.1783834832913] in priority_queue
                #except AssertionError:
                #   print(priority_queue)
                #print(region_id)
                #if (not "." in popped[0]) or ("." in popped[0] and len(popped[0].split(".")) < bucket_level):
                #print(len(region_id.split(".")))
                #if is_profi:
                if popped[0] not in self.objects_in_buckets and len(popped[0].split(".")) <= bucket_level-1:
                    #s = time.time()
                    priority_queue, pivot_distances = get_wspd(priority_queue, pivot_distances, regions, pow_list, popped, self.objects_in_buckets, max_levels=bucket_level, is_profi=is_profi, find_bucket=find_bucket)
                    #distances.extend(pivot_distances)
                    #total += t
                    #other_nodes_all.extend(other_nodes)
                #else:
                #s = time.time()
                #    priority_queue, t = get_wspd_cophir(priority_queue, pivot_distances, regions, pow_list, popped)
                #e = time.time()
                #total += t
                if stop_cond_objects != None:
                    """
                    index = tuple([int(p) for p in popped.split('.')[2:]])
                    if len(index) == 1:
                        index = str(index[0])
                    else:
                        index_str = ""
                        for n in index:
                            index_str += f"{int(n)}."
                        index_str = index_str[:-1]
                        index = index_str
                    """
                    if popped[0] in self.objects_in_buckets:
                        #print(f"{popped[0]} in objects_in_buckets")
                        step = self.objects_in_buckets[popped[0]]
                        n_steps += step
                        #print(n_steps)
                    else:
                        #print(f"{popped[0]} NOT in objects_in_buckets")
                        step = 0
                    steps.append(step)
                    """
                    indexes.append(index)
                    n_steps += step
                    if n_steps > last_n_steps:
                        last_n_steps = n_steps
                        print(f"n_steps: {n_steps}")
                    """
                """
                if len(priority_queue) == 0:
                    #return other_nodes_all
                    def MyFn(s):
                        return len(s[0].split("."))
                    other_nodes_all = sorted(other_nodes_all, key=MyFn, reverse=True)
                    #other_nodes_all = sorted(other_nodes_all, key=lambda x: x[1])
                    priority_queue = other_nodes_all#[:len(other_nodes_all)//2]
                    print(f"Popped everything from main PQ. Using backup PQ of length: {len(priority_queue)} | found objs: {n_steps}")
                    find_bucket = False
                """
                #if n_steps > 100000:
                #    #print(len(priority_queue), n_steps)
                #    return priority_queue
                if stop_cond_objects != None and current_stop_cond_idx < len(stop_cond_objects) and stop_cond_objects[current_stop_cond_idx] <= n_steps:
                    time_checkpoint = time.time()
                    time_checkpoints.append(time_checkpoint-start)
                    popped_nodes_ = []
                    for m,s in zip(visited_models, steps):
                        if s != 0:
                            popped_nodes_.append(f"C.1.{str(m)}")
                        else:
                            popped_nodes_.append(f"M.1.{str(m)}")
                    popped_nodes_checkpoints.append(popped_nodes_)
                    steps_checkpoints.append(n_steps)
                    current_stop_cond_idx += 1
        return visited_models

    #self.add_level_to_queue(priority_q, probs, (model_label[(-7)]), (model_label[(-6)]), (model_label[(-5)]), (model_label[(-4)]), (model_label[(-3)]), (model_label[(-2)]), (model_label[(-1)]), pop=False)
    def add_level_to_queue(self, priority_q, probs, from_l1_level=None, from_l2_level=None, from_l3_level=None, from_l4_level=None, from_l5_level=None, from_l6_level=None, from_l7_level=None,  parent_value=None, pop=True):
        #s= time.time()
        if len(priority_q) != 0 and pop:
            priority_q.pop()
        for i in probs:
            if from_l1_level != None and from_l2_level != None and from_l3_level != None and from_l4_level != None and from_l5_level != None and from_l6_level != None and from_l7_level != None:
                #if parent_value: i['votes_perc'] = (parent_value+ i['votes_perc'])/2#parent_value+ i['votes_perc']*pow(0.75, 7)
                priority_q.append({f"M.1.{from_l1_level}.{from_l2_level}.{from_l3_level}.{from_l4_level}.{from_l5_level}.{from_l6_level}.{from_l7_level}." + str(i['value_l8']): i['votes_perc']})
            elif from_l1_level != None and from_l2_level != None and from_l3_level != None and from_l4_level != None and from_l5_level != None and from_l6_level != None:
                #if parent_value: i['votes_perc'] =  (parent_value+ i['votes_perc'])/2#parent_value+ i['votes_perc']*pow(0.75, 6)
                priority_q.append({f"M.1.{from_l1_level}.{from_l2_level}.{from_l3_level}.{from_l4_level}.{from_l5_level}.{from_l6_level}." + str(i['value_l7']): i['votes_perc']})
            elif from_l1_level != None and from_l2_level != None and from_l3_level != None and from_l4_level != None and from_l5_level != None:
                #if parent_value: i['votes_perc'] =  (parent_value+ i['votes_perc'])/2#parent_value+ i['votes_perc']*pow(0.75, 5)
                priority_q.append({f"M.1.{from_l1_level}.{from_l2_level}.{from_l3_level}.{from_l4_level}.{from_l5_level}." + str(i['value_l6']): i['votes_perc']})
            elif from_l1_level != None and from_l2_level != None and from_l3_level != None and from_l4_level != None:
                #if parent_value: i['votes_perc'] = (parent_value+ i['votes_perc'])/2 #parent_value+ i['votes_perc']*pow(0.75, 4)
                priority_q.append({f"M.1.{from_l1_level}.{from_l2_level}.{from_l3_level}.{from_l4_level}." + str(i['value_l5']): i['votes_perc']})
            elif from_l1_level != None and from_l2_level != None and from_l3_level != None:
                #if parent_value: i['votes_perc'] =  (parent_value+ i['votes_perc'])/2#parent_value+ i['votes_perc']*pow(0.75, 3)
                priority_q.append({f"M.1.{from_l1_level}.{from_l2_level}.{from_l3_level}." + str(i['value_l4']): i['votes_perc']})
            elif from_l1_level != None and from_l2_level != None:
                #if parent_value: i['votes_perc'] =  (parent_value+ i['votes_perc'])/2#parent_value+ i['votes_perc']*pow(0.75, 2)
                priority_q.append({f"M.1.{from_l1_level}.{from_l2_level}." + str(i['value_l3']): i['votes_perc']})
            elif from_l1_level != None:
                #if parent_value: i['votes_perc'] =  (parent_value+ i['votes_perc'])/2#parent_value+ i['votes_perc']*pow(0.75, 1)
                priority_q.append({f"M.1.{from_l1_level}." + str(i['value_l2']): i['votes_perc']})
            else:
                priority_q.append({'M.1.' + str(i['value_l1']): i['votes_perc']})
        #print(time.time() - s)
        return priority_q

    def train_level(self, df, groupby_df, estimators, max_depth, y_label="L2", custom_classifier=None, is_multilabel=False, is_forest=True):
        stack_l1 = []; preds_l1 = []; object_ids = []; mapping = []
        labels = self.get_descriptive_col_names_pred()
        L2_max = int(df[y_label].max())
        merged = []
        for name, group in groupby_df:
            if not is_multilabel:
                X = group.drop(labels + [f"{l}_pred" for l in labels] + ["object_id"], axis=1, errors="ignore").values
                y = group[[y_label]].values
            else:
                labels_l = [f"L{l}_labels" for l in range(1, len(labels)+1)]
                X = group.drop(labels + [f"{l}_pred" for l in labels] + labels_l + ["object_id"], axis=1, errors='ignore').values
                y = one_hot_frequency_encode(group[y_label].values, L2_max)
            assert X.shape[1] == 282

            object_ids.extend(group["object_id"].values)
            if is_forest:
                logging.info(f"({name}) : L{str(int(y_label[-1])-1)} is_forest={is_forest} | {X.shape} samples")
                clf = self.get_forest(max_depth=max_depth, n_estimators=estimators)
                clf.fit(X, y)
                preds = clf.predict(X)
            elif is_multilabel:
                logging.info(f"({name}) : L{str(int(y_label[-1])-1)} is_multilabel={is_multilabel} | {X.shape} samples")
                #frequency_labels = one_hot_frequency_encode(group["L2_labels"], n_cats=L2_max)
                clf = get_simple_baseline_model(output_data_shape=L2_max)
                clf = compile_baseline(clf)
                #clf.compile(loss='categorical_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])
                #y_1 = to_categorical(y)
                clf.fit(X, y, epochs=1, batch_size=100, verbose=False)
                preds = [np.argmax(p)+1 for p in clf.predict(X)]
            elif custom_classifier is not None:
                logging.info(f"{y_label} | {name} : Training {custom_classifier.__class__.__name__} | {X.shape} samples")
                clf = clone(custom_classifier)
                if np.unique(y, return_counts=True)[1].shape[0] == 1:
                    logging.info(f"{name}, not fitting, merging {name}.{int(y[0][0])}")
                    merged.append((name, int(y[0][0])))
                    clf = DecisionTreeClassifier()
                    clf.fit(X, y)
                    preds = clf.predict(X)
                    #preds = y#np.array(y[0])
                    #print(f"Passing {print(preds.shape)}")
                else:
                    clf.fit(X, y)
                    preds = clf.predict(X)
                    #print(preds.shape)
            else:
                clf = construct_fully_connected_model_1M(output_data_shape=y.max())
                clf.fit(X, y, epochs=1)
                preds = clf.predict(X)
                preds = [np.argmax(p) for p in preds]
    
            stack_l1.append(clf)
            preds_l1.extend(preds)
            if type(name) is float or type(name) is int:
                mapping.append([int(name)])
            else:
                mapping.append([int(n) for n in name])
        df_l2 = pd.DataFrame(np.array([preds_l1, object_ids]).T, columns=[y_label+"_pred"] + ["object_id"])
        df_l1 = df.merge(df_l2, on="object_id")
        return stack_l1, df_l1, mapping, merged

    def get_data_labels(self, df, level=1):
        from tensorflow.keras.utils import to_categorical
        X = df.drop(["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "object_id"], axis=1).values
        y = df[["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"]].values
        X = X.reshape(1000000,282,1)
        #y = to_categorical(y[:, level-1])
        return X, to_categorical(y[:, level-1]), y[:, level-1]

    def train_mindex_lvl_8(self, df, estimators=50, max_depth=30, custom_classifier=None, is_multilabel=False, root_nn=None):
        #X, y_1, y_2, y_obj_id = self.get_training_data(df)
        df = df.sample(frac=1)
        is_forest = True
        merged_all = []
        labels = self.get_descriptive_col_names() # not counting object_id
        if is_multilabel:
            labels_ = [f"L{i}_labels" for i in range(1, len(labels)+1)]
            X = df.drop(labels + ["object_id"] + labels_, axis=1).values
            #print(labels[0])
            y = one_hot_frequency_encode(df[[labels[0]]].values, df["L1"].max())
        else:
            X = df.drop(labels + ["object_id"], axis=1).values
            y = df[[labels[0]]].values

        #logging.info(f"Training root model: {X.shape}")
        y_obj_id = df["object_id"].values
        if type(estimators) is list:
            est = estimators[0]
        else:
            est = estimators

        if custom_classifier is not None:
            logging.info(f"Training {custom_classifier.__class__.__name__}")
            root = clone(custom_classifier)
            root.fit(X, y)
            #logging.info(f"Predicting with {custom_classifier.__class__.__name__}")
            preds = root.predict(X)
            is_forest = False
        elif root_nn is None:
            logging.info(f"Training RF, using max_depth={max_depth} est={est}")
            root = self.get_forest(max_depth=max_depth, n_estimators=est)
            root.fit(X, y)
            preds = root.predict(X)
        elif is_multilabel is not None:
            logging.info(f"Predicting multilabel")
            root = root_nn
            preds = root.predict(X)
            preds = [np.argmax(p)+1 for p in preds]
            is_forest = False
        else:
            root = root_nn
            preds = root.predict(X.values.reshape(1000000,282,1))
            preds = [np.argmax(p) for p in preds]
            is_forest = False
        df_l1 = pd.DataFrame(np.array([preds, y_obj_id]).T, columns=["L1_pred", "object_id"])
        df_root = df.merge(df_l1, on="object_id")
        l1_df = df_root.dropna(subset=["L2"])
        #estimators = 20
        if type(estimators) is list and len(estimators) > 1:
            est = estimators[1]

        #logging.info(f"Training L1 model: {l1_df.shape}")
        stack_l1, df_l1, mapping_l1, merged = self.train_level(l1_df, l1_df.groupby(["L1_pred"]), est, max_depth, custom_classifier=custom_classifier, is_multilabel=is_multilabel, is_forest=is_forest, y_label="L2")
        merged_all.extend(merged)
        if type(estimators) is list and len(estimators) > 2:
            est = estimators[2]

        l2_df = df_l1.dropna(subset=["L3"])
        #logging.info(f"Training L2 model: {l2_df.shape}")
        stack_l2, df_l2, mapping_l2, merged = self.train_level(l2_df, l2_df.groupby(["L1_pred", "L2_pred"]), est, max_depth,custom_classifier=custom_classifier,is_multilabel=is_multilabel, is_forest=is_forest, y_label="L3")
        merged_all.extend(merged)
        if type(estimators) is list and len(estimators) > 3:
            est = estimators[3]

        l3_df = df_l2.dropna(subset=["L4"])
        #logging.info(f"Training L3 model: {l3_df.shape}")
        stack_l3, df_l3, mapping_l3, merged = self.train_level(l3_df, l3_df.groupby(["L1_pred", "L2_pred", "L3_pred"]), est, max_depth,custom_classifier=custom_classifier,is_multilabel=is_multilabel, is_forest=is_forest, y_label="L4")
        merged_all.extend(merged)
        if type(estimators) is list and len(estimators) > 4:
            est = estimators[4]

        l4_df = df_l3.dropna(subset=["L5"])
        #logging.info(f"Training L4 model: {l4_df.shape}")
        stack_l4, df_l4, mapping_l4, merged = self.train_level(l4_df, l4_df.groupby(["L1_pred", "L2_pred", "L3_pred","L4_pred"]), est, max_depth,custom_classifier=custom_classifier, is_multilabel=is_multilabel, is_forest=is_forest, y_label="L5")
        merged_all.extend(merged)
        if type(estimators) is list and len(estimators) > 5:
            est = estimators[5]

        l5_df = df_l4.dropna(subset=["L6"])
        #logging.info(f"Training L5 model: {l5_df.shape}")
        stack_l5, df_l5, mapping_l5, merged = self.train_level(l5_df, l5_df.groupby(["L1_pred", "L2_pred", "L3_pred","L4_pred", "L5_pred"]), est, max_depth,custom_classifier=custom_classifier, is_multilabel=is_multilabel, is_forest=is_forest, y_label="L6")
        merged_all.extend(merged)
        if type(estimators) is list and len(estimators) > 6:
            est = estimators[6]

        l6_df = df_l5.dropna(subset=["L7"])
        #logging.info(f"Training L6 model: {l6_df.shape}")
        stack_l6, df_l6, mapping_l6, merged = self.train_level(l6_df, l6_df.groupby(["L1_pred", "L2_pred", "L3_pred","L4_pred", "L5_pred", "L6_pred"]), est, max_depth, custom_classifier=custom_classifier, is_multilabel=is_multilabel, is_forest=is_forest, y_label="L7")
        merged_all.extend(merged)
        if type(estimators) is list and len(estimators) > 7:
            est = estimators[7]

        l7_df = df_l6.dropna(subset=["L8"])
        #logging.info(f"Training L7 model: {l7_df.shape}")
        stack_l7, df_l7, mapping_l7, merged = self.train_level(l7_df, l7_df.groupby(["L1_pred", "L2_pred", "L3_pred","L4_pred", "L5_pred", "L6_pred", "L7_pred"]), est, max_depth,custom_classifier=custom_classifier, is_multilabel=is_multilabel, is_forest=is_forest, y_label="L8")
        merged_all.extend(merged)
        stack = [root, stack_l1, stack_l2, stack_l3, stack_l4, stack_l5, stack_l6, stack_l7]
        c_df = df.copy()
        c_df = c_df.merge(df_root[["object_id", "L1_pred"]], on=["object_id"], how="left")
        c_df = c_df.merge(df_l1[["object_id", "L2_pred"]], on=["object_id"], how="left")
        c_df = c_df.merge(df_l2[["object_id", "L3_pred"]], on=["object_id"], how="left")
        c_df = c_df.merge(df_l3[["object_id", "L4_pred"]], on=["object_id"], how="left")
        c_df = c_df.merge(df_l4[["object_id", "L5_pred"]], on=["object_id"], how="left")
        c_df = c_df.merge(df_l5[["object_id", "L6_pred"]], on=["object_id"], how="left")
        c_df = c_df.merge(df_l6[["object_id", "L7_pred"]], on=["object_id"], how="left")
        c_df = c_df.merge(df_l7[["object_id", "L8_pred"]], on=["object_id"], how="left")
        df_res = c_df
        return stack, df_res, [mapping_l1, mapping_l2, mapping_l3, mapping_l4, mapping_l5, mapping_l6, mapping_l7], self.get_existing_models_buckets(df_res), merged_all

    def train_mindex_lvl_6(self, df, estimators=50, max_depth=30, custom_classifier=None, is_multilabel=False, root_nn=None):
        #X, y_1, y_2, y_obj_id = self.get_training_data(df)
        df = df.sample(frac=1)
        is_forest = True
        labels = self.get_descriptive_col_names() # not counting object_id
        if is_multilabel:
            labels_ = [f"L{i}_labels" for i in range(1, len(labels)+1)]
            X = df.drop(labels + ["object_id"] + labels_, axis=1).values
            #print(labels[0])
            y = one_hot_frequency_encode(df[[labels[0]]].values, df["L1"].max())
        else:
            X = df.drop(labels + ["object_id"], axis=1).values
            y = df[[labels[0]]].values
        assert X.shape[1] == 282
        #logging.info(f"Training root model: {X.shape}")
        y_obj_id = df["object_id"].values
        if type(estimators) is list:
            est = estimators[0]
        else:
            est = estimators

        if custom_classifier is not None:
            logging.info(f"Training {custom_classifier.__class__.__name__}")
            root = clone(custom_classifier)
            root.fit(X, y)
            #logging.info(f"Predicting with {custom_classifier.__class__.__name__}")
            preds = root.predict(X)
            is_forest = False
        elif root_nn is None:
            logging.info(f"Training RF, using max_depth={max_depth} est={est}")
            root = self.get_forest(max_depth=max_depth, n_estimators=est)
            root.fit(X, y)
            preds = root.predict(X)
        elif is_multilabel is not None:
            logging.info(f"Predicting multilabel")
            root = root_nn
            preds = root.predict(X)
            preds = [np.argmax(p)+1 for p in preds]
            is_forest = False
        else:
            root = root_nn
            preds = root.predict(X.values.reshape(1000000,282,1))
            preds = [np.argmax(p) for p in preds]
            is_forest = False
        df_l1 = pd.DataFrame(np.array([preds, y_obj_id]).T, columns=["L1_pred", "object_id"])
        df_root = df.merge(df_l1, on="object_id")
        l1_df = df_root.dropna(subset=["L2"])
        #estimators = 20
        #return root, l1_df
        if type(estimators) is list and len(estimators) > 1:
            est = estimators[1]

        stack_l1, df_l1, mapping_l1 = self.train_level(l1_df, l1_df.groupby(["L1_pred"]), est, max_depth, custom_classifier=custom_classifier, is_multilabel=is_multilabel, is_forest=is_forest, y_label="L2")

        if type(estimators) is list and len(estimators) > 2:
            est = estimators[2]

        l2_df = df_l1.dropna(subset=["L3"])
        stack_l2, df_l2, mapping_l2 = self.train_level(l2_df, l2_df.groupby(["L1_pred", "L2_pred"]), est, max_depth,custom_classifier=custom_classifier,is_multilabel=is_multilabel, is_forest=is_forest, y_label="L3")

        if type(estimators) is list and len(estimators) > 3:
            est = estimators[3]

        l3_df = df_l2.dropna(subset=["L4"])
        stack_l3, df_l3, mapping_l3 = self.train_level(l3_df, l3_df.groupby(["L1_pred", "L2_pred", "L3_pred"]), est, max_depth,custom_classifier=custom_classifier,is_multilabel=is_multilabel, is_forest=is_forest, y_label="L4")
        #return stack_l3, df_l3, mapping_l3
        if type(estimators) is list and len(estimators) > 4:
            est = estimators[4]

        l4_df = df_l3.dropna(subset=["L5"])
        stack_l4, df_l4, mapping_l4 = self.train_level(l4_df, l4_df.groupby(["L1_pred", "L2_pred", "L3_pred", "L4_pred"]), est, max_depth,custom_classifier=custom_classifier, is_multilabel=is_multilabel, is_forest=is_forest, y_label="L5")

        if type(estimators) is list and len(estimators) > 5:
            est = estimators[5]

        l5_df = df_l4.dropna(subset=["L6"])
        stack_l5, df_l5, mapping_l5 = self.train_level(l5_df, l5_df.groupby(["L1_pred", "L2_pred", "L3_pred", "L4_pred", "L5_pred"]), est, max_depth,custom_classifier=custom_classifier, is_multilabel=is_multilabel, is_forest=is_forest, y_label="L6")

        if type(estimators) is list and len(estimators) > 6:
            est = estimators[6]

        stack = [root, stack_l1, stack_l2, stack_l3, stack_l4, stack_l5]
        c_df = df.copy()
        c_df = c_df.merge(df_root[["object_id", "L1_pred"]], on=["object_id"], how="left")
        c_df = c_df.merge(df_l1[["object_id", "L2_pred"]], on=["object_id"], how="left")
        c_df = c_df.merge(df_l2[["object_id", "L3_pred"]], on=["object_id"], how="left")
        c_df = c_df.merge(df_l3[["object_id", "L4_pred"]], on=["object_id"], how="left")
        c_df = c_df.merge(df_l4[["object_id", "L5_pred"]], on=["object_id"], how="left")
        c_df = c_df.merge(df_l5[["object_id", "L6_pred"]], on=["object_id"], how="left")
        df_res = c_df
        return stack, df_res, [mapping_l1, mapping_l2, mapping_l3, mapping_l4, mapping_l5], self.get_existing_models_buckets_6(df_res)

    def unison_shuffled_copies(self, a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    def train_mindex_lvl_2(self, index_df, estimators=50, max_depth=30, custom_classifier=None, is_multilabel=False, labels=["L1", "L2"], root_nn=None):
        #X, y_1, y_2, y_obj_id = self.get_training_data(df)
        df = index_df.sample(frac=1)
        #objects_shuff, index_shuff = self.unison_shuffled_copies(objects, index.values)
        is_forest = True
        #labels = self.get_descriptive_col_names() # not counting object_id
        if is_multilabel:
            labels_ = [f"L{i}_labels" for i in range(1, len(labels)+1)]
            X = df.drop(labels + ["object_id"] + labels_, axis=1).values
            #print(labels[0])
            y = one_hot_frequency_encode(df[[labels[0]]].values, df["L1"].max())
        else:
            X = df.drop(labels + ["object_id"], axis=1).values
            y = df[[labels[0]]].values

        #logging.info(f"Training root model: {X.shape}")
        y_obj_id = df["object_id"].values
        if type(estimators) is list:
            est = estimators[0]
        else:
            est = estimators

        if custom_classifier is not None:
            logging.info(f"Training {custom_classifier.__class__.__name__}")
            root = clone(custom_classifier)
            root.fit(X, y)
            #logging.info(f"Predicting with {custom_classifier.__class__.__name__}")
            preds = root.predict(X)
            is_forest = False
        elif root_nn is None:
            logging.info(f"Training RF, using max_depth={max_depth} est={est}")
            root = self.get_forest(max_depth=max_depth, n_estimators=est)
            root.fit(X, y)
            preds = root.predict(X)
        elif is_multilabel is not None:
            logging.info(f"Predicting multilabel")
            root = root_nn
            preds = root.predict(X)
            preds = [np.argmax(p)+1 for p in preds]
            is_forest = False
        else:
            root = root_nn
            preds = root.predict(X.values.reshape(1000000,282,1))
            preds = [np.argmax(p) for p in preds]
            is_forest = False
        df_l1 = pd.DataFrame(np.array([preds, y_obj_id]).T, columns=["L1_pred", "object_id"])
        df_root = df.merge(df_l1, on="object_id")
        l1_df = df_root.dropna(subset=["L2"])
        #estimators = 20
        if type(estimators) is list and len(estimators) > 1:
            est = estimators[1]
        return df_root
        stack_l1, df_l1, mapping_l1 = self.train_level(l1_df, l1_df.groupby(["L1_pred"]), est, max_depth, custom_classifier=custom_classifier, is_multilabel=is_multilabel, is_forest=is_forest, y_label="L2")

        stack = [root, stack_l1]
        c_df = df.copy()
        c_df = c_df.merge(df_root[["object_id", "L1_pred"]], on=["object_id"], how="left")
        c_df = c_df.merge(df_l1[["object_id", "L2_pred"]], on=["object_id"], how="left")
        df_res = c_df
        return stack, df_res, [mapping_l1, mapping_l2], self.get_existing_models_buckets_2(df_res)


    def get_mindex_pivots_df(self, data_file="../struct/cophir-pivots-256-random.data", crop_n=128, should_be_int=True):
        # 128 for 127 being max value in L8 M-index
        labels, numerical, descr_lengths = parse_objects(data_file, is_filter=False)
        pivots_df = pd.DataFrame(numerical)
        pivots_df['object_id'] = labels
        if crop_n:
            pivots_df = pivots_df.head(crop_n)
        return pivots_df
    
    def make_bucket_objects_dict(self, df_orig):
        """
        if type(name) is not int:
            name_str = ""
            for n in name:
                name_str += f"{int(n)}."
            name_str = name_str[:-1]
            self.objects_in_buckets[name_str] = g.shape[0]
        else:
            self.objects_in_buckets[f"{name}"] = g.shape[0]
        """
        regions = []
        for i, label in enumerate(self.labels):
            #regions.append([])
            group_cond = [f"{self.labels[j]}" for j in range(0, i+1)]
            if len(self.labels) > i+1:
                na_groups = df_orig[df_orig[self.labels[i+1]].isna()].groupby(group_cond)
            else:
                na_groups = df_orig.groupby(group_cond)
            for name, g in na_groups:
                #print(name)
                if type(name) != int:
                    name_str = ""
                    for n in name:
                        name_str += f"{int(n)}."
                    name_str = name_str[:-1]
                    self.objects_in_buckets[name_str] = g.shape[0]
                    for substr in name_str:
                        for i in range(1, len(name_str.split("."))+1):
                            potential_region = ".".join(name_str.split(".")[:i])
                            #if potential_region not in regions:
                            regions.append(potential_region)
                else:
                    self.objects_in_buckets[f"{name}"] = g.shape[0]
                    regions.append(f"{name}")
                    #regions[-1].append(f"{name}")
        
        assert sum(list(self.objects_in_buckets.values())) == df_orig.shape[0]
        regions = list(dict.fromkeys(regions))
        regions = sorted(regions, key=len)
        regions = {i : True for i in regions}
        return regions

    def get_mindex_existing_regions_dict_profi_6(self, df_orig):
        """
        df_L2 = df_orig[df_orig["L2"].isna()][["L1", "L2"]]
        df_L3 = df_orig[~(df_orig["L2"].isna()) & (df_orig["L3"].isna())][["L1", "L2", "L3"]]
        df_L4 = df_orig[~(df_orig["L2"].isna()) & ~(df_orig["L3"].isna()) & (df_orig["L4"].isna())][["L1", "L2", "L3", "L4"]]
        df_L5 = df_orig[~(df_orig["L2"].isna()) & ~(df_orig["L3"].isna()) & ~(df_orig["L4"].isna()) & (df_orig["L5"].isna())][["L1", "L2", "L3", "L4", "L5"]]
        df_L6 = df_orig[~(df_orig["L2"].isna()) & ~(df_orig["L3"].isna()) & ~(df_orig["L4"].isna()) & ~(df_orig["L5"].isna()) & (df_orig["L6"].isna())][["L1", "L2", "L3", "L4", "L5", "L6"]]
        df_L6_na = df_orig[~(df_orig["L6"].isna())]

        existing_regions = np.array(list(df_orig.groupby(["L1", "L2", "L3", "L4", "L5", "L6"]).groups.keys()))
        existing_regions_unique = [ np.unique(existing_regions[:, :2][~np.isnan(existing_regions[:, :2]).any(axis=1)], axis=0), \
                                    np.unique(existing_regions[:, :3][~np.isnan(existing_regions[:, :3]).any(axis=1)], axis=0), \
                                    np.unique(existing_regions[:, :4][~np.isnan(existing_regions[:, :4]).any(axis=1)], axis=0), \
                                    np.unique(existing_regions[:, :5][~np.isnan(existing_regions[:, :5]).any(axis=1)], axis=0), \
                                    np.unique(existing_regions[:, :6][~np.isnan(existing_regions[:, :6]).any(axis=1)], axis=0)
                                    ]

                if type(name) is not int:
                    name_str = ""
                    for n in name:
                        name_str += f"{int(n)}."
                    name_str = name_str[:-1]
                    self.objects_in_buckets[name_str] = g.shape[0]
                else:
                    self.objects_in_buckets[f"{name}"] = g.shape[0]


        """
        regions = []
        for i, label in enumerate(self.labels):
            regions.append([])
            group_cond = [f"{self.labels[j]}" for j in range(0, i+1)]
            if len(self.labels) > i+1:
                na_groups = df_orig[df_orig[self.labels[i+1]].isna()].groupby(group_cond)
            else:
                na_groups = df_orig.groupby(group_cond)
            for name, g in na_groups:
                #print(name)
                if type(name) != int:
                    name = tuple([int(n) for n in name])
                    name_str = ""
                    for n in name:
                        name_str += f"{int(n)}."
                    name_str = name_str[:-1]
                else:
                    name_str = str(name)
                regions[-1].append(name_str)
                self.objects_in_buckets[name] = g.shape[0]       
        """
        existing_regions_dict = {}
        for regions in existing_regions_unique:
            for r in regions:
                s = ""
                for e in r:
                    s += f"{int(e)}."
                s = s[:-1]
                existing_regions_dict[s] = True
        existing_regions_dict
        """
        return existing_regions_unique, existing_regions_dict

    def get_mindex_existing_regions_dict_profi_2(self, df_orig):
        existing_regions = np.array(list(df_orig.groupby(["L1", "L2"]).groups.keys()))
        existing_regions_unique = [ np.unique(existing_regions[:, :2][~np.isnan(existing_regions[:, :2]).any(axis=1)], axis=0)]

        for i, label in enumerate(self.labels):
            group_cond = [f"{self.labels[j]}" for j in range(0, i+1)]
            if len(self.labels) > i+1:
                na_groups = df_orig[df_orig[self.labels[i+1]].isna()].groupby(group_cond)
            else:
                na_groups = df_orig.groupby(group_cond)
            for name, g in na_groups:
                print(name)
                if type(name) != int:
                    name = tuple([int(n) for n in name])
                self.objects_in_buckets[name] = g.shape[0]       
       
        existing_regions_dict = {}
        for regions in existing_regions_unique:
            for r in regions:
                s = ""
                for e in r:
                    s += f"{int(e)}."
                s = s[:-1]
                existing_regions_dict[s] = True
        existing_regions_dict
        return existing_regions_unique, existing_regions_dict

    def get_mindex_existing_regions_dict(self, df_orig):
        existing_regions = np.array(list(df_orig.groupby(["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"]).groups.keys()))
        existing_regions_unique = [ np.unique(existing_regions[:, :2][~np.isnan(existing_regions[:, :2]).any(axis=1)], axis=0), \
                                    np.unique(existing_regions[:, :3][~np.isnan(existing_regions[:, :3]).any(axis=1)], axis=0), \
                                    np.unique(existing_regions[:, :4][~np.isnan(existing_regions[:, :4]).any(axis=1)], axis=0), \
                                    np.unique(existing_regions[:, :5][~np.isnan(existing_regions[:, :5]).any(axis=1)], axis=0), \
                                    np.unique(existing_regions[:, :6][~np.isnan(existing_regions[:, :6]).any(axis=1)], axis=0), \
                                    np.unique(existing_regions[:, :7][~np.isnan(existing_regions[:, :7]).any(axis=1)], axis=0), \
                                    np.unique(existing_regions[:, :8][~np.isnan(existing_regions[:, :8]).any(axis=1)], axis=0)]

        for i, label in enumerate(self.labels):
            group_cond = [f"{self.labels[j]}" for j in range(0, i)]
            #print(label)
            #if len(self.labels) > i+1:
            na_groups = df_orig[df_orig[label].isna()].groupby(group_cond)
            #else:
            #    na_groups = df_orig.groupby(group_cond)
            for name, g in na_groups:
                if type(name) != int:
                    name = tuple([int(n) for n in name])
                self.objects_in_buckets[name] = g.shape[0]       
       
        existing_regions_dict = {}
        for regions in existing_regions_unique:
            for r in regions:
                s = ""
                for e in r:
                    s += f"{int(e)}."
                s = s[:-1]
                existing_regions_dict[s] = True
        existing_regions_dict
        return existing_regions_unique, existing_regions_dict

    def get_mindex_existing_regions_dict_6(self, df_orig, labels=["L1", "L2", "L3", "L4", "L5", "L6"]):
        labels_gb = df_orig.groupby(labels)
        existing_regions = np.array(list(labels_gb.groups.keys()))
        existing_regions_unique = [ np.unique(existing_regions[:, :2][~np.isnan(existing_regions[:, :2]).any(axis=1)], axis=0), \
                                    np.unique(existing_regions[:, :3][~np.isnan(existing_regions[:, :3]).any(axis=1)], axis=0), \
                                    np.unique(existing_regions[:, :4][~np.isnan(existing_regions[:, :4]).any(axis=1)], axis=0), \
                                    np.unique(existing_regions[:, :5][~np.isnan(existing_regions[:, :5]).any(axis=1)], axis=0), \
                                    np.unique(existing_regions[:, :6][~np.isnan(existing_regions[:, :6]).any(axis=1)], axis=0)]
        
        for i, label in enumerate(self.labels):
            group_cond = [f"{self.labels[j]}" for j in range(0, i)]
            #if len(self.labels) > i+1:
            na_groups = df_orig[df_orig[label].isna()].groupby(group_cond)
            #else:
            #    na_groups = df_orig.groupby(group_cond)
            for name, g in na_groups:
                if type(name) != int:
                    name = tuple([int(n) for n in name])
                self.objects_in_buckets[name] = g.shape[0]

        existing_regions_dict = {}
        for regions in existing_regions_unique:
            for r in regions:
                s = ""
                for e in r:
                    s += f"{int(e)}."
                s = s[:-1]
                existing_regions_dict[s] = True
        return existing_regions_unique, existing_regions_dict


    def load_indexes(self, filenames=['level-1.txt', 'level-2.txt', 'level-3.txt', 'level-4.txt', 'level-5.txt', 'level-6.txt']):
        df_i = pd.DataFrame([])
        col_names_all = self.get_descriptive_col_names()
        filenames.reverse()
        for c, filename in enumerate(filenames):
            if c != 0: col_names = col_names_all[:-c]
            else: col_names = col_names_all
            print(self.dir+filename)
            df_ = pd.read_csv(self.dir+filename, names = col_names + ["object_id"], sep=r'[.+\s+]', engine='python', header=None)   
            df_i = pd.concat([df_i, df_]).drop_duplicates(["object_id"])
        return df_i.apply(pd.to_numeric, errors='ignore')

    def scale_per_descriptor(self, df, descriptor_value_counts):
        col_pos = 0
        normalized = []
        numerical = df.drop(self.get_descriptive_col_names() + ["object_id"], axis=1).values
        for descriptor_value in descriptor_value_counts:
            current = numerical[:, col_pos:col_pos+descriptor_value]
            normalized.append(preprocessing.scale(current))
            col_pos += descriptor_value
        df = df.drop(df.columns.difference(self.get_descriptive_col_names()+["object_id"]), 1)
        df = pd.concat([df, pd.DataFrame(np.hstack((normalized)))], axis=1)
        return df

    def get_dfs(self, filenames=['level-1.txt', 'level-2.txt', 'level-3.txt', 'level-4.txt', 'level-5.txt', 'level-6.txt']):
        index_df = self.load_indexes(filenames)
        labels, numerical, descr_lengths = parse_objects(f"{self.dir}/objects.txt")
        #return labels, pd.DataFrame(numerical), index_df
        df_orig = merge_dfs(numerical, labels, index_df)
        df = self.scale_per_descriptor(df_orig, descr_lengths)
        self.L1_range = (df["L1"].min(), df["L1"].max())
        self.L2_range = (df["L2"].min(), df["L2"].max())
        self.L3_range = (df["L3"].min(), df["L3"].max())
        self.L4_range = (df["L4"].min(), df["L4"].max())
        self.L5_range = (df["L5"].min(), df["L5"].max())
        self.L6_range = (df["L6"].min(), df["L6"].max())
        return df, df_orig

    
    def get_forest(self, max_depth=30, n_estimators=200, min_samples_split=10, min_samples_leaf=4, n_jobs=5):
        # temp, TODO: FIX
        return RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=21, n_jobs=n_jobs)
        #return RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, n_jobs = n_jobs, random_state=21)

    def get_training_data(self, df, random_state=21):
        X = df.drop(self.get_descriptive_col_names() + ["object_id"], axis=1)
        y = df[self.get_descriptive_col_names() + ["object_id"]].values
        X, y = shuffle(X,y, random_state=random_state)
        y_1 = y[:, 0]; y_2 = y[:, 1]; y_obj_id = y[:, -1]
        print(X.shape, y_1.shape, y_2.shape)
        return X, y_1, y_2, y_obj_id

    def get_predictions_root(self, clf, X, y_1, y_2, y_obj_id):
        est_predictions  = estimator_predict(clf, X, y_1)
        predictions_1 = majority_voting(clf, est_predictions, self.L1_range[1])
        df_1 = X.copy()
        df_1["L1"] = y_1
        df_1["L2"] = y_2
        df_1["L1_pred"] = predictions_1
        df_1["object_id"] = y_obj_id
        print(f"Accuracy on root: {1 - df_1[df_1['L1'] != df_1['L1_pred']].shape[0] / df_1.shape[0]}")
        return df_1

    def get_ranges(self):
        return self.L1_range, self.L2_range, self.L3_range, self.L4_range, self.L5_range, self.L6_range

    def get_splits_L1(self, df_root):
        split_dfs = [df_root.dropna(subset=["L2"])[df_root.dropna(subset=["L2"])["L1_pred"] == i] for i in range(1, int(self.L1_range[1])+1, 1)]
        split_data = []
        for df_ in split_dfs:
            X = df_.drop(self.get_descriptive_col_names_pred()+['object_id'], axis=1, errors='ignore')
            assert X.shape[1] == 282
            y = df_[["L1", "L2", "object_id"]].values
            split_data.append({'X': X, 'y_2': y[:,1], 'y_1': y[:,0], 'object_id': y[:,-1]})
        return split_dfs, split_data

    def collect_predictions_for_L2(self, df_root, models, split_data, level_label="L2_pred", level_target="y_2", level_target_label="L2"):
        t = np.array([]); p = np.array([]); y_2 = []
        all_models_predictions = []
        for model, data_for_model in zip(models, split_data):
            if data_for_model['X'].shape[0] != 0:
                est_predictions = estimator_predict(model, data_for_model['X'], data_for_model[level_target])
                predictions_ = majority_voting(model, est_predictions)
                t = np.hstack((t, data_for_model['object_id']))
                p = np.hstack((p, predictions_))
                y_2.extend(data_for_model[level_target])
        df_2 = pd.DataFrame(data={"object_id": t, level_label: p}, dtype=np.int64)
        df_2[level_target_label] = y_2
        return df_2

    """
    def train_level(self, split_data, level="L1", n_estimators=200, max_depth=20):
        label_range = None
        if level == "L1": label_range = int(self.L1_range[1])
        clfs_2 = [self.get_forest(max_depth=max_depth, n_estimators=n_estimators) for x in range(1, label_range+1, 1)]
        for i in range(label_range):
            if split_data[i]['X'].shape[0] != 0:
                print(i, split_data[i]['X'].shape)
                clfs_2[i].fit(split_data[i]['X'], split_data[i]['y_2'])
        return clfs_2
    """
    def get_splits_L2(self, df_prev):
        #split_dfs = [df_l1[(df_l1[\"L1_pred\"] == i) & (df_l1[\"L2_pred\"] == j)] for i in range(1, 10+1, 1) for j in range(1, 50+1, 1)]
        split_dfs = df_prev.groupby(["L1_pred", "L2_pred"])
        groups = []
        split_data = []
        for name, df_ in split_dfs:
            X = df_.drop(self.get_descriptive_col_names_pred()+['object_id'], axis=1, errors='ignore')
            assert X.shape[1] == 282
            y = df_[["L1", "L2", "L3", "object_id"]].values
            split_data.append({'X': X, 'y_3': y[:,2], 'y_2': y[:,1], 'y_1': y[:,0], 'object_id': y[:,-1]})
            groups.append(name)
        return split_dfs, split_data, groups
    
    def train_level_L2(self, split_data, groups, max_depth=30, n_estimators=200):
        groups = np.array(groups)
        max_L1 = groups[:, 0].max(); max_L2 = groups[:, 1].max()
        clfs_2 = [[[] for j in range(max_L2+1)] for i in range(max_L1+1)]
        for i, g in enumerate(groups):
            clfs_2[g[0]][g[1]] = self.get_forest(max_depth=max_depth, n_estimators=n_estimators)
            clfs_2[g[0]][g[1]].fit(split_data[i]['X'], split_data[i]['y_3'])
        return clfs_2

    def collect_predictions_for_L3(self, models, split_data, groups, level_label="L3_pred", level_target="y_3", level_target_label="L3"):
        t = np.array([]); p = np.array([]); y_2 = []
        all_models_predictions = []
        for i, g in enumerate(groups):
            est_predictions = estimator_predict(models[g[0]][g[1]], split_data[i]['X'], split_data[i][level_target])
            predictions_ = majority_voting(models[g[0]][g[1]], est_predictions)
            t = np.hstack((t, split_data[i]['object_id']))
            p = np.hstack((p, predictions_))
            y_2.extend(split_data[i][level_target])
        df_2 = pd.DataFrame(data={"object_id": t, level_label: p}, dtype=np.int64)
        df_2[level_target_label] = y_2
        df_2 = df_2.sort_values(by=["object_id"])
        return df_2

    def get_splits_L3(self, df_prev):
        #split_dfs = [df_l1[(df_l1[\"L1_pred\"] == i) & (df_l1[\"L2_pred\"] == j)] for i in range(1, 10+1, 1) for j in range(1, 50+1, 1)]
        split_dfs = df_prev.groupby(["L1_pred", "L2_pred", "L3_pred"])
        groups = []
        split_data = []
        for name, df_ in split_dfs:
            X = df_.drop(self.get_descriptive_col_names_pred()+['object_id'], axis=1, errors='ignore')
            assert X.shape[1] == 282
            y = df_[["L1", "L2", "L3", "L4", "object_id"]].values
            split_data.append({'X': X, 'y_4': y[:,3], 'y_3': y[:,2], 'y_2': y[:,1], 'y_1': y[:,0], 'object_id': y[:,-1]})
            groups.append(name)
        return split_dfs, split_data, groups

    def train_level_L3(self, split_data, groups, n_estimators=200, max_depth=20):
        groups = np.array(groups)
        max_L1 = groups[:, 0].max(); max_L2 = groups[:, 1].max(); max_L3 = groups[:, 2].max()
        clfs_2 = [[[[] for k in range(max_L3+1)] for j in range(max_L2+1)] for i in range(max_L1+1)]
        for i, g in enumerate(groups):
            clfs_2[g[0]][g[1]][g[2]] = self.get_forest(n_estimators=n_estimators, max_depth=max_depth)
            clfs_2[g[0]][g[1]][g[2]].fit(split_data[i]['X'], split_data[i]['y_4'])
        return clfs_2

    def collect_predictions_for_L4(self, models, split_data, groups, level_label="L4_pred", level_target="y_4", level_target_label="L4"):
        t = np.array([]); p = np.array([]); y_2 = []
        all_models_predictions = []
        for i, g in enumerate(groups):
            est_predictions = estimator_predict(models[g[0]][g[1]][g[2]], split_data[i]['X'], split_data[i][level_target])
            predictions_ = majority_voting(models[g[0]][g[1]][g[2]], est_predictions)
            t = np.hstack((t, split_data[i]['object_id']))
            p = np.hstack((p, predictions_))
            y_2.extend(split_data[i][level_target])
        df_2 = pd.DataFrame(data={"object_id": t, level_label: p}, dtype=np.int64)
        df_2[level_target_label] = y_2
        df_2 = df_2.sort_values(by=["object_id"])
        return df_2

    def get_splits_L4(self, df_prev):
        split_dfs = df_prev.groupby(["L1_pred", "L2_pred", "L3_pred", "L4_pred"])
        groups = []
        split_data = []
        for name, df_ in split_dfs:
            X = df_.drop(self.get_descriptive_col_names_pred()+['object_id'], axis=1, errors='ignore')
            assert X.shape[1] == 282
            y = df_[["L1", "L2", "L3", "L4", "L5", "object_id"]].values
            split_data.append({'X': X, 'y_5': y[:,4], 'y_4': y[:,3], 'y_3': y[:,2], 'y_2': y[:,1], 'y_1': y[:,0], 'object_id': y[:,-1]})
            groups.append(name)
        return split_dfs, split_data, groups

    def train_level_L4(self, split_data, groups, n_estimators=200, max_depth=30):
        groups = np.array(groups)
        max_L1 = groups[:, 0].max(); max_L2 = groups[:, 1].max(); max_L3 = groups[:, 2].max();  max_L4 = groups[:, 3].max()
        clfs_2 = [[[[[] for l in range(max_L4+1)] for k in range(max_L3+1)] for j in range(max_L2+1)] for i in range(max_L1+1)]
        for i, g in enumerate(groups):
            clfs_2[g[0]][g[1]][g[2]][g[3]] = self.get_forest(n_estimators=n_estimators, max_depth=max_depth)
            clfs_2[g[0]][g[1]][g[2]][g[3]].fit(split_data[i]['X'], split_data[i]['y_5'])
        return clfs_2

    def collect_predictions_for_L5(self, models, split_data, groups, level_label="L5_pred", level_target="y_5", level_target_label="L5"):
        t = np.array([]); p = np.array([]); y_2 = []
        all_models_predictions = []
        for i, g in enumerate(groups):
            est_predictions = estimator_predict(models[g[0]][g[1]][g[2]][g[3]], split_data[i]['X'], split_data[i][level_target])
            predictions_ = majority_voting(models[g[0]][g[1]][g[2]][g[3]], est_predictions)
            t = np.hstack((t, split_data[i]['object_id']))
            p = np.hstack((p, predictions_))
            y_2.extend(split_data[i][level_target])
        df_2 = pd.DataFrame(data={"object_id": t, level_label: p}, dtype=np.int64)
        df_2[level_target_label] = y_2
        df_2 = df_2.sort_values(by=["object_id"])
        return df_2

    def get_splits_L5(self, df_prev):
        split_dfs = df_prev.groupby(["L1_pred", "L2_pred", "L3_pred", "L4_pred", "L5_pred"])
        groups = []
        split_data = []
        for name, df_ in split_dfs:
            X = df_.drop(self.get_descriptive_col_names_pred()+['object_id'], axis=1, errors='ignore')
            assert X.shape[1] == 282
            y = df_[["L1", "L2", "L3", "L4", "L5", "L6", "object_id"]].values
            split_data.append({'X': X, 'y_6': y[:,5], 'y_5': y[:,4], 'y_4': y[:,3], 'y_3': y[:,2], 'y_2': y[:,1], 'y_1': y[:,0], 'object_id': y[:,-1]})
            groups.append(name)
        return split_dfs, split_data, groups

    def get_splits_L6(self, df_prev):
        split_dfs = df_prev.groupby(["L1_pred", "L2_pred", "L3_pred", "L4_pred", "L5_pred", "L6_pred"])
        groups = []
        split_data = []
        for name, df_ in split_dfs:
            X = df_.drop(self.get_descriptive_col_names_pred()+['object_id'], axis=1, errors='ignore')
            assert X.shape[1] == 282
            y = df_[["L1", "L2", "L3", "L4", "L5", "L6", "L7",  "object_id"]].values
            split_data.append({'X': X, 'y_7': y[:,6],  'y_6': y[:,5], 'y_5': y[:,4], 'y_4': y[:,3], 'y_3': y[:,2], 'y_2': y[:,1], 'y_1': y[:,0], 'object_id': y[:,-1]})
            groups.append(name)
        return split_dfs, split_data, groups

    def get_splits_L7(self, df_prev):
        split_dfs = df_prev.groupby(["L1_pred", "L2_pred", "L3_pred", "L4_pred", "L5_pred", "L6_pred", "L7_pred"])
        groups = []
        split_data = []
        for name, df_ in split_dfs:
            X = df_.drop(self.get_descriptive_col_names_pred()+['object_id'], axis=1, errors='ignore')
            assert X.shape[1] == 282
            y = df_[["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "object_id"]].values
            split_data.append({'X': X, 'y_8': y[:,7], 'y_7': y[:,6],  'y_6': y[:,5], 'y_5': y[:,4], 'y_4': y[:,3], 'y_3': y[:,2], 'y_2': y[:,1], 'y_1': y[:,0], 'object_id': y[:,-1]})
            groups.append(name)
        return split_dfs, split_data, groups

    def train_level_L5(self, split_data, groups, n_estimators=200, max_depth=30):
        groups = np.array(groups)
        max_L1 = groups[:, 0].max(); max_L2 = groups[:, 1].max(); max_L3 = groups[:, 2].max();  max_L4 = groups[:, 3].max(); max_L5 = groups[:, 4].max()
        clfs_2 = [[[[[[] for k in range(max_L5+1)] for l in range(max_L4+1)] for k in range(max_L3+1)] for j in range(max_L2+1)] for i in range(max_L1+1)]
        for i, g in enumerate(groups):
            clfs_2[g[0]][g[1]][g[2]][g[3]][g[4]] = self.get_forest(n_estimators=n_estimators, max_depth=max_depth)
            clfs_2[g[0]][g[1]][g[2]][g[3]][g[4]].fit(split_data[i]['X'], split_data[i]['y_6'])
        return clfs_2

    def train_level_L6(self, split_data, groups, n_estimators=200, max_depth=30):
        groups = np.array(groups)
        max_L1 = groups[:, 0].max(); max_L2 = groups[:, 1].max(); max_L3 = groups[:, 2].max();  max_L4 = groups[:, 3].max(); max_L5 = groups[:, 4].max(); max_L6=groups[:, 5].max()
        clfs_2 = [[[[[[[] for j in range(max_L6+1)] for k in range(max_L5+1)] for l in range(max_L4+1)] for k in range(max_L3+1)] for j in range(max_L2+1)] for i in range(max_L1+1)]
        for i, g in enumerate(groups):
            clfs_2[g[0]][g[1]][g[2]][g[3]][g[4]][g[5]] = self.get_forest(n_estimators=n_estimators, max_depth=max_depth)
            clfs_2[g[0]][g[1]][g[2]][g[3]][g[4]][g[5]].fit(split_data[i]['X'], split_data[i]['y_7'])
        return clfs_2

    def train_level_L7(self, split_data, groups, n_estimators=200, max_depth=30):
        groups = np.array(groups)
        max_L1 = groups[:, 0].max(); max_L2 = groups[:, 1].max(); max_L3 = groups[:, 2].max();  max_L4 = groups[:, 3].max(); max_L5 = groups[:, 4].max(); max_L6=groups[:, 5].max(); max_L7=groups[:, 6].max()
        clfs_2 = [[[[[[[[] for i in range(max_L7+1)] for j in range(max_L6+1)] for k in range(max_L5+1)] for l in range(max_L4+1)] for k in range(max_L3+1)] for j in range(max_L2+1)] for i in range(max_L1+1)]
        for i, g in enumerate(groups):
            clfs_2[g[0]][g[1]][g[2]][g[3]][g[4]][g[5]][g[6]] = self.get_forest(n_estimators=n_estimators, max_depth=max_depth)
            clfs_2[g[0]][g[1]][g[2]][g[3]][g[4]][g[5]][g[6]].fit(split_data[i]['X'], split_data[i]['y_8'])
        return clfs_2

    def collect_predictions_for_L6(self, models, split_data, groups, level_label="L6_pred", level_target="y_6", level_target_label="L6"):
        t = np.array([]); p = np.array([]); y_2 = []
        all_models_predictions = []
        for i, g in enumerate(groups):
            est_predictions = estimator_predict(models[g[0]][g[1]][g[2]][g[3]][g[4]], split_data[i]['X'], split_data[i][level_target])
            predictions_ = majority_voting(models[g[0]][g[1]][g[2]][g[3]][g[4]], est_predictions)
            t = np.hstack((t, split_data[i]['object_id']))
            p = np.hstack((p, predictions_))
            y_2.extend(split_data[i][level_target])
        df_2 = pd.DataFrame(data={"object_id": t, level_label: p}, dtype=np.int64)
        df_2[level_target_label] = y_2
        df_2 = df_2.sort_values(by=["object_id"])
        return df_2

    def collect_predictions_for_L7(self, models, split_data, groups, level_label="L7_pred", level_target="y_7", level_target_label="L7"):
        t = np.array([]); p = np.array([]); y_2 = []
        all_models_predictions = []
        for i, g in enumerate(groups):
            est_predictions = estimator_predict(models[g[0]][g[1]][g[2]][g[3]][g[4]][g[5]], split_data[i]['X'], split_data[i][level_target])
            predictions_ = majority_voting(models[g[0]][g[1]][g[2]][g[3]][g[4]][g[5]], est_predictions)
            t = np.hstack((t, split_data[i]['object_id']))
            p = np.hstack((p, predictions_))
            y_2.extend(split_data[i][level_target])
        df_2 = pd.DataFrame(data={"object_id": t, level_label: p}, dtype=np.int64)
        df_2[level_target_label] = y_2
        df_2 = df_2.sort_values(by=["object_id"])
        return df_2

    def collect_predictions_for_L8(self, models, split_data, groups, level_label="L8_pred", level_target="y_8", level_target_label="L8"):
        t = np.array([]); p = np.array([]); y_2 = []
        all_models_predictions = []
        for i, g in enumerate(groups):
            est_predictions = estimator_predict(models[g[0]][g[1]][g[2]][g[3]][g[4]][g[5]][g[6]], split_data[i]['X'], split_data[i][level_target])
            predictions_ = majority_voting(models[g[0]][g[1]][g[2]][g[3]][g[4]][g[5]][g[6]], est_predictions)
            t = np.hstack((t, split_data[i]['object_id']))
            p = np.hstack((p, predictions_))
            y_2.extend(split_data[i][level_target])
        df_2 = pd.DataFrame(data={"object_id": t, level_label: p}, dtype=np.int64)
        df_2[level_target_label] = y_2
        df_2 = df_2.sort_values(by=["object_id"])
        return df_2

    def approximate_search_2(self, model_stack, df_res, object_id, steps_limit_leaf=None, end_on_exact_hit=True, over_approx_dict=None, debug=False):
        row = df_res[(df_res['object_id'] == object_id)]
        n_steps_global = 0
        n_leaf_steps_global = 0
        x = row.drop((self.get_descriptive_col_names_pred() + ["object_id"]), axis=1, errors='ignore').values
        gts = row[self.get_descriptive_col_names_2()].values[0]
        l1 = get_classification_probs_per_level(x, model_stack[0])
        priority_q = [{'M.1': 1.0}]
        if debug:
            print(f"Step 0: M.1 added - PQ: {priority_q}\n")
        priority_q = self.add_level_to_queue(priority_q, l1)
        if debug:
            print(f"Step 1: L1 added - PQ: {priority_q[:5]}, ...\n")
        is_L1_hit = False; is_L2_hit = False; is_leaf_hit = False
        popped_nodes = []
        while (not end_on_exact_hit and len(priority_q) != 0) or (is_leaf_hit != True and len(priority_q) != 0):
            if steps_limit_leaf != None and steps_limit_leaf <= n_leaf_steps_global:
                return {'id':object_id, 'leaf_nodes_hit':n_leaf_steps_global, 'steps_to_hit':n_steps_global, 'is_hit':is_leaf_hit, 'popped_nodes':popped_nodes}
            else:
                if debug:
                    print(f"Step {n_steps_global + 2} - Model visit {n_steps_global + 1}: ")
                priority_q, popped, is_curr_L1_hit, is_curr_L2_hit, is_curr_leaf_hit = self.process_node_2(priority_q, x, gts, model_stack, over_approx_dict, debug=debug)
                #print(is_curr_L1_hit, is_curr_L2_hit, is_curr_L3_hit, is_curr_L4_hit, is_curr_L5_hit, is_curr_L6_hit, is_curr_leaf_hit)
                if type(popped) is list:
                    popped_nodes.extend(popped)
                else: popped_nodes.append(popped)
                popped_nodes = list(set(popped_nodes))
                popped_nodes = sorted(popped_nodes, key=len)
                #print(popped_nodes)
                n_leaf_steps_global = self.get_number_of_leaf_node_models(popped_nodes)
                if is_curr_L1_hit: is_L1_hit = True
                if is_curr_L2_hit: is_L2_hit = True
                if is_curr_leaf_hit: is_leaf_hit = True
            n_steps_global += 1

        return {'id':object_id, 'leaf_nodes_hit':n_leaf_steps_global, 'steps_to_hit':n_steps_global, 'is_hit': is_leaf_hit, 'popped_nodes':popped_nodes}

    def approximate_search(self, model_stack, df_res, object_id, steps_limit_leaf=None, existing_buckets= None, end_on_exact_hit=True, over_approx_dict=None, debug=False):
        row = df_res[(df_res['object_id'] == object_id)]
        n_steps_global = 0
        n_leaf_steps_global = 0
        x = row.drop((self.get_descriptive_col_names_pred() + ["object_id"]), axis=1, errors='ignore').values
        gts = row[self.get_descriptive_col_names()].values[0]
        l1 = get_classification_probs_per_level(x, model_stack[0])
        priority_q = [{'M.1': 1.0}]
        if debug:
            print(f"Step 0: M.1 added - PQ: {priority_q}\n")
        priority_q = self.add_level_to_queue(priority_q, l1)
        if debug:
            print(f"Step 1: L1 added - PQ: {priority_q[:5]}, ...\n")
        is_L1_hit = False; is_L2_hit = False; is_L3_hit = False; is_L4_hit = False; is_L5_hit = False; is_L6_hit = False; is_leaf_hit = False
        popped_nodes = []
        while (not end_on_exact_hit and len(priority_q) != 0) or (is_leaf_hit != True and len(priority_q) != 0):
            if steps_limit_leaf != None and steps_limit_leaf <= n_leaf_steps_global:
                return {'id':object_id, 'leaf_nodes_hit':n_leaf_steps_global, 'steps_to_hit':n_steps_global, 'is_hit':is_leaf_hit, 'popped_nodes':popped_nodes}
            else:
                if debug:
                    print(f"Step {n_steps_global + 2} - Model visit {n_steps_global + 1}: ")
                priority_q, popped, is_curr_L1_hit, is_curr_L2_hit, is_curr_L3_hit, is_curr_L4_hit, is_curr_L5_hit, is_curr_L6_hit, is_curr_leaf_hit = self.process_node(priority_q, x, gts, model_stack, over_approx_dict, existing_buckets=existing_buckets, debug=debug)
                #print(is_curr_L1_hit, is_curr_L2_hit, is_curr_L3_hit, is_curr_L4_hit, is_curr_L5_hit, is_curr_L6_hit, is_curr_leaf_hit)
                if type(popped) is list:
                    popped_nodes.extend(popped)
                else: popped_nodes.append(popped)
                popped_nodes = list(set(popped_nodes))
                popped_nodes = sorted(popped_nodes, key=len)
                #print(popped_nodes)
                n_leaf_steps_global = self.get_number_of_leaf_node_models(popped_nodes)
                if is_curr_L1_hit: is_L1_hit = True
                if is_curr_L2_hit: is_L2_hit = True
                if is_curr_L3_hit: is_L3_hit = True
                if is_curr_L4_hit: is_L4_hit = True
                if is_curr_L5_hit: is_L5_hit = True
                if is_curr_L6_hit: is_L6_hit = True
                if is_curr_leaf_hit: is_leaf_hit = True
            n_steps_global += 1

        return {'id':object_id, 'leaf_nodes_hit':n_leaf_steps_global, 'steps_to_hit':n_steps_global, 'is_hit': is_leaf_hit, 'popped_nodes':popped_nodes}

    def remove_merged_paths(self, merged, popped, debug=False):
        for parent in merged:
            base_node_label = "M.1."
            #print(parent, type(parent[0]))
            if type(parent[0]) is not int:
                for p in parent[0]:
                    base_node_label += f"{p}."
            else:
                for p in parent:
                    base_node_label += f"{p}."
            base_node_label = base_node_label[:-1]
            if base_node_label == popped:
                print(f"Popped {base_node_label}")
                return None
        return popped

    def approximate_search_8(self, model_stack, df_res, object_id, mapping, steps_limit_leaf=None, existing_buckets= None, custom_classifier=None, steps_limit=None, end_on_exact_hit=True, over_approx_dict=None, merged_paths=None, debug=False):
        
        row = df_res[(df_res['object_id'] == object_id)]
        n_steps_global = 0
        n_leaf_steps_global = 0
        x = row.drop((self.get_descriptive_col_names_pred() + ["object_id"]), axis=1, errors='ignore').values
        gts = row[self.get_descriptive_col_names()].values[0]
        gts_preds = row[[v + "_pred" for v in self.get_descriptive_col_names()]].values[0]
        gts_preds = [g for g in gts_preds if not math.isnan(g)]
        #print(gts_preds)
        s = time.time()
        l1 = get_classification_probs_per_level(x, model_stack[0], custom_classifier=custom_classifier)
        #print(time.time() - s)
        #return
        priority_q = [{'M.1': 1.0}]
        if debug:
            print(f"Step 0: M.1 added - PQ: {priority_q}\n")
        priority_q = self.add_level_to_queue(priority_q, l1)
        #print(l1)
        if debug:
            print(f"Step 1: L1 added - PQ: {priority_q}\n")
        is_L1_hit = False; is_L2_hit = False; is_L3_hit = False; is_L4_hit = False; is_L5_hit = False; is_L6_hit = False; is_L7_hit = False; is_L8_hit = False; is_leaf_hit = False
        popped_nodes = []
        bucket_hit = 0
        #print(len(priority_q))

        while (not end_on_exact_hit and len(priority_q) != 0) or (is_leaf_hit != True and len(priority_q) != 0):
            if (steps_limit != None and steps_limit <= n_steps_global) or (steps_limit_leaf != None and steps_limit_leaf <= n_leaf_steps_global):
                return {'id':object_id, 'leaf_nodes_hit':n_leaf_steps_global, 'steps_to_hit':n_steps_global, 'is_hit':bucket_hit >= len(gts_preds), 'popped_nodes':popped_nodes}
            else:
                if debug:
                    print(f"Step {n_steps_global + 2} - Model visit {n_steps_global + 1}: ")
                priority_q, popped, is_curr_L1_hit, is_curr_L2_hit, is_curr_L3_hit, is_curr_L4_hit, is_curr_L5_hit, is_curr_L6_hit, is_curr_L7_hit, is_curr_L8_hit = self.process_node_8(priority_q, x, gts, gts_preds, mapping, model_stack, existing_buckets=existing_buckets, custom_classifier=custom_classifier, debug=debug)
                #print(priority_q)
                #print(len(priority_q))
                if len(priority_q) > steps_limit:
                    priority_q = priority_q[:steps_limit]
                if merged_paths is not None:
                    popped = self.remove_merged_paths(merged_paths, popped)
                if (merged_paths and popped is not None) or (not merged_paths):
                    if type(popped) is list:
                        popped_nodes.extend(popped)
                        popped_nodes = list(set(popped_nodes))
                    else: popped_nodes.append(popped)
                    popped_nodes = list(set(popped_nodes))
                    popped_nodes = sorted(popped_nodes, key=len)
                    n_leaf_steps_global = self.get_number_of_leaf_node_models(popped_nodes)
                    #print( is_curr_L1_hit, is_curr_L2_hit, is_curr_L3_hit, is_curr_L4_hit, is_curr_L5_hit, is_curr_L6_hit, is_curr_L7_hit, is_curr_L8_hit)
                    if is_curr_L1_hit: 
                        is_L1_hit = True
                        bucket_hit += 1
                    if is_curr_L2_hit: 
                        is_L2_hit = True
                        bucket_hit += 1
                    if is_curr_L3_hit: 
                        is_L3_hit = True
                        bucket_hit += 1
                    if is_curr_L4_hit:
                        is_L4_hit = True
                        bucket_hit += 1
                    if is_curr_L5_hit:
                        is_L5_hit = True
                        bucket_hit += 1
                    if is_curr_L6_hit:
                        is_L6_hit = True
                        bucket_hit += 1
                    if is_curr_L7_hit:
                        is_L7_hit = True
                        bucket_hit += 1
                    if is_curr_L8_hit:
                        is_L8_hit = True
                        bucket_hit += 1
                    n_steps_global = len(list(set(popped_nodes)))
            #print(bucket_hit)
        return {'id':object_id, 'leaf_nodes_hit':n_leaf_steps_global, 'steps_to_hit':n_steps_global, 'is_hit': bucket_hit >= len(gts_preds), 'popped_nodes':popped_nodes}

    def approximate_search_6(self, model_stack, df_res, object_id, mapping, steps_limit_leaf=None, existing_buckets= None, custom_classifier=None, steps_limit=None, end_on_exact_hit=True, over_approx_dict=None, debug=False):
        
        row = df_res[(df_res['object_id'] == object_id)]
        n_steps_global = 0
        n_leaf_steps_global = 0
        x = row.drop((self.get_descriptive_col_names_pred() + ["object_id"]), axis=1, errors='ignore').values
        gts = row[self.get_descriptive_col_names()].values[0]
        gts_preds = row[[v + "_pred" for v in self.get_descriptive_col_names()]].values[0]
        gts_preds = [g for g in gts_preds if not math.isnan(g)]
        #print(gts_preds)
        s = time.time()
        l1 = get_classification_probs_per_level(x, model_stack[0], custom_classifier=custom_classifier)
        #print(time.time() - s)
        #return
        priority_q = [{'M.1': 1.0}]
        if debug:
            print(f"Step 0: M.1 added - PQ: {priority_q}\n")
        priority_q = self.add_level_to_queue(priority_q, l1)
        #print(l1)
        if debug:
            print(f"Step 1: L1 added - PQ: {priority_q}\n")
        is_L1_hit = False; is_L2_hit = False; is_L3_hit = False; is_L4_hit = False; is_L5_hit = False; is_L6_hit = False; is_leaf_hit = False
        popped_nodes = []
        bucket_hit = 0
        #print(len(priority_q))

        while (not end_on_exact_hit and len(priority_q) != 0) or (is_leaf_hit != True and len(priority_q) != 0):
            if (steps_limit != None and steps_limit <= n_steps_global) or (steps_limit_leaf != None and steps_limit_leaf <= n_leaf_steps_global):
                return {'id':object_id, 'leaf_nodes_hit':n_leaf_steps_global, 'steps_to_hit':n_steps_global, 'is_hit':bucket_hit >= len(gts_preds), 'popped_nodes':popped_nodes}
            else:
                if debug:
                    print(f"Step {n_steps_global + 2} - Model visit {n_steps_global + 1}: ")
                priority_q, popped, is_curr_L1_hit, is_curr_L2_hit, is_curr_L3_hit, is_curr_L4_hit, is_curr_L5_hit, is_curr_L6_hit = self.process_node_6(priority_q, x, gts, gts_preds, mapping, model_stack, existing_buckets=existing_buckets, custom_classifier=custom_classifier, debug=debug)
                #print(priority_q)
                #print(len(priority_q))
                if len(priority_q) > steps_limit:
                    priority_q = priority_q[:steps_limit] 
                if type(popped) is list:
                    popped_nodes.extend(popped)
                    popped_nodes = list(set(popped_nodes))
                else: popped_nodes.append(popped)
                popped_nodes = list(set(popped_nodes))
                popped_nodes = sorted(popped_nodes, key=len)
                n_leaf_steps_global = self.get_number_of_leaf_node_models(popped_nodes)
                #print( is_curr_L1_hit, is_curr_L2_hit, is_curr_L3_hit, is_curr_L4_hit, is_curr_L5_hit, is_curr_L6_hit, is_curr_L7_hit, is_curr_L8_hit)
                if is_curr_L1_hit: 
                    is_L1_hit = True
                    bucket_hit += 1
                if is_curr_L2_hit: 
                    is_L2_hit = True
                    bucket_hit += 1
                if is_curr_L3_hit: 
                    is_L3_hit = True
                    bucket_hit += 1
                if is_curr_L4_hit:
                    is_L4_hit = True
                    bucket_hit += 1
                if is_curr_L5_hit:
                    is_L5_hit = True
                    bucket_hit += 1
                if is_curr_L6_hit:
                    is_L6_hit = True
                    bucket_hit += 1
            n_steps_global = len(list(set(popped_nodes)))
            #print(bucket_hit)
        return {'id':object_id, 'leaf_nodes_hit':n_leaf_steps_global, 'steps_to_hit':n_steps_global, 'is_hit': bucket_hit >= len(gts_preds), 'popped_nodes':popped_nodes}


    def get_existing_models_buckets(self, df):
        L2_leaf_cats = list(df[df["L3_pred"].isna()].groupby(["L1_pred", "L2_pred"]).groups.keys())
        L3_leaf_cats = list(df[(df["L4_pred"].isna()) & (~df["L3"].isna())].groupby(["L1_pred", "L2_pred", "L3_pred"]).groups.keys())
        L4_leaf_cats = list(df[df["L5_pred"].isna() & (~df['L4_pred'].isna()) & (~df['L3_pred'].isna())].groupby(["L1_pred", "L2_pred", "L3_pred", "L4_pred"]).groups.keys())
        L5_leaf_cats = list(df[df["L6_pred"].isna() & (~df['L5_pred'].isna()) & (~df['L4_pred'].isna()) & (~df['L3_pred'].isna())].groupby(["L1_pred", "L2_pred", "L3_pred", "L4_pred", "L5_pred"]).groups.keys())
        L6_leaf_cats = list(df[df["L7_pred"].isna() & (~df['L6_pred'].isna()) & (~df['L5_pred'].isna()) & (~df['L4_pred'].isna()) & (~df['L3_pred'].isna())].groupby(["L1_pred", "L2_pred", "L3_pred", "L4_pred", "L5_pred", "L6_pred"]).groups.keys())
        L7_leaf_cats = list(df[df["L8_pred"].isna() & (~df['L7_pred'].isna()) & (~df['L6_pred'].isna()) & (~df['L5_pred'].isna()) & (~df['L4_pred'].isna()) & (~df['L3_pred'].isna())].groupby(["L1_pred", "L2_pred", "L3_pred", "L4_pred", "L5_pred", "L6_pred", "L7_pred"]).groups.keys())
        L8_leaf_cats = list(df[~df["L8_pred"].isna()].groupby(["L1_pred", "L2_pred", "L3_pred", "L4_pred", "L5_pred", "L6_pred", "L7_pred", "L8_pred"]).groups.keys())

        existing_buckets = L2_leaf_cats
        existing_buckets.extend(L3_leaf_cats)
        existing_buckets.extend(L4_leaf_cats)
        existing_buckets.extend(L5_leaf_cats)
        existing_buckets.extend(L6_leaf_cats)
        existing_buckets.extend(L7_leaf_cats)
        existing_buckets.extend(L8_leaf_cats)
        print(f"N. of non-empty buckets: {len(existing_buckets)}")
        L2_inner_cats = list(df[~df["L3_pred"].isna()].groupby(["L1_pred", "L2_pred"]).groups.keys())
        L3_inner_cats = list(df[(~df["L4_pred"].isna()) & (~df["L3_pred"].isna())].groupby(["L1_pred", "L2_pred", "L3_pred"]).groups.keys())
        L4_inner_cats = list(df[~df["L5_pred"].isna() & (~df['L4_pred'].isna()) & (~df['L3_pred'].isna())].groupby(["L1_pred", "L2_pred", "L3_pred", "L4_pred"]).groups.keys())
        L5_inner_cats = list(df[~df["L6_pred"].isna() & (~df['L5_pred'].isna()) & (~df['L4_pred'].isna()) & (~df['L3_pred'].isna())].groupby(["L1_pred", "L2_pred", "L3_pred", "L4_pred", "L5_pred"]).groups.keys())
        L6_inner_cats = list(df[~df["L7_pred"].isna() & (~df['L6_pred'].isna()) & (~df['L5_pred'].isna()) & (~df['L4_pred'].isna()) & (~df['L3_pred'].isna())].groupby(["L1_pred", "L2_pred", "L3_pred", "L4_pred", "L5_pred", "L6_pred"]).groups.keys())
        L7_inner_cats = list(df[~df["L8_pred"].isna() & (~df['L7_pred'].isna()) & (~df['L6_pred'].isna()) & (~df['L5_pred'].isna()) & (~df['L4_pred'].isna()) & (~df['L3_pred'].isna())].groupby(["L1_pred", "L2_pred", "L3_pred", "L4_pred", "L5_pred", "L6_pred", "L7_pred"]).groups.keys())
        
        existing_models = L2_inner_cats
        existing_models.extend(L3_inner_cats)
        existing_models.extend(L4_inner_cats)
        existing_models.extend(L5_inner_cats)
        existing_models.extend(L6_inner_cats)
        existing_models.extend(L7_inner_cats)
        print(f"N. of non-empty inner models: {len(existing_models)}")

        existing_bucket_models = existing_buckets
        existing_bucket_models.extend(existing_models)
        return existing_bucket_models

    def get_existing_models_buckets_6(self, df):
        L2_leaf_cats = list(df[df["L3_pred"].isna()].groupby(["L1_pred", "L2_pred"]).groups.keys())
        L3_leaf_cats = list(df[(df["L4_pred"].isna()) & (~df["L3"].isna())].groupby(["L1_pred", "L2_pred", "L3_pred"]).groups.keys())
        L4_leaf_cats = list(df[df["L5_pred"].isna() & (~df['L4_pred'].isna()) & (~df['L3_pred'].isna())].groupby(["L1_pred", "L2_pred", "L3_pred", "L4_pred"]).groups.keys())
        L5_leaf_cats = list(df[df["L6_pred"].isna() & (~df['L5_pred'].isna()) & (~df['L4_pred'].isna()) & (~df['L3_pred'].isna())].groupby(["L1_pred", "L2_pred", "L3_pred", "L4_pred", "L5_pred"]).groups.keys())
        L6_leaf_cats = list(df[~df["L6_pred"].isna()].groupby(["L1_pred", "L2_pred", "L3_pred", "L4_pred", "L5_pred", "L6_pred"]).groups.keys())

        existing_buckets = L2_leaf_cats
        existing_buckets.extend(L3_leaf_cats)
        existing_buckets.extend(L4_leaf_cats)
        existing_buckets.extend(L5_leaf_cats)
        existing_buckets.extend(L6_leaf_cats)
        print(f"N. of non-empty buckets: {len(existing_buckets)}")
        L2_inner_cats = list(df[~df["L3_pred"].isna()].groupby(["L1_pred", "L2_pred"]).groups.keys())
        L3_inner_cats = list(df[(~df["L4_pred"].isna()) & (~df["L3_pred"].isna())].groupby(["L1_pred", "L2_pred", "L3_pred"]).groups.keys())
        L4_inner_cats = list(df[~df["L5_pred"].isna() & (~df['L4_pred'].isna()) & (~df['L3_pred'].isna())].groupby(["L1_pred", "L2_pred", "L3_pred", "L4_pred"]).groups.keys())
        L5_inner_cats = list(df[~df["L6_pred"].isna() & (~df['L5_pred'].isna()) & (~df['L4_pred'].isna()) & (~df['L3_pred'].isna())].groupby(["L1_pred", "L2_pred", "L3_pred", "L4_pred", "L5_pred"]).groups.keys())
        
        existing_models = L2_inner_cats
        existing_models.extend(L3_inner_cats)
        existing_models.extend(L4_inner_cats)
        existing_models.extend(L5_inner_cats)
        print(f"N. of non-empty inner models: {len(existing_models)}")

        existing_bucket_models = existing_buckets
        existing_bucket_models.extend(existing_models)
        return existing_bucket_models

    def get_existing_models_buckets_2(self, df):
        L2_leaf_cats = list(df[df["L2_pred"].isna()].groupby(["L1_pred"]).groups.keys())

        existing_buckets = L2_leaf_cats
        print(f"N. of non-empty buckets: {len(existing_buckets)}")
        L2_inner_cats = list(df[~df["L2_pred"].isna()].groupby(["L1_pred"]).groups.keys())

        existing_models = L2_inner_cats
        print(f"N. of non-empty inner models: {len(existing_models)}")

        existing_bucket_models = existing_buckets
        existing_bucket_models.extend(existing_models)
        return existing_bucket_models

    def add_level_to_queue_8(self, priority_q, probs, from_l1_level=None, from_l2_level=None, from_l3_level=None, from_l4_level=None, from_l5_level=None, from_l6_level=None, from_l7_level=None, from_l8_level=None, pop=True, value="value_l7"):
        if len(priority_q) != 0:
            if pop:
                priority_q.pop()
        modelclass = "M"
        if value == "value_leaf":
            modelclass = "C"
        key = list(probs[0].keys())[0]
        for i in probs:
            #print(i[key], from_l2_level)
            #if i['votes_perc'] != 0:
            if from_l1_level and from_l2_level != None and from_l3_level != None and from_l4_level != None and from_l5_level != None and from_l6_level != None and from_l7_level != None and from_l8_level != None:
                if modelclass == "C": i['value_l9'] = i['value_leaf']
                priority_q.append({f"{modelclass}.1.{from_l1_level}.{from_l2_level}.{from_l3_level}.{from_l4_level}.{from_l5_level}.{from_l6_level}.{from_l7_level}.{from_l8_level}." + str(i['value_l9']): i['votes_perc']})
            elif from_l1_level and from_l2_level != None and from_l3_level != None and from_l4_level != None and from_l5_level != None and from_l6_level != None and from_l7_level != None:
                if modelclass == "C": i['value_l8'] = i['value_leaf']
                #if ((int(from_l1_level), int(from_l2_level), int(from_l3_level), int(from_l4_level), int(from_l5_level), int(from_l6_level), int(from_l7_level), int(i['value_l8'])) in existing_bucket_models):
                priority_q.append({f"{modelclass}.1.{from_l1_level}.{from_l2_level}.{from_l3_level}.{from_l4_level}.{from_l5_level}.{from_l6_level}.{from_l7_level}." + str(i['value_l8']): i['votes_perc']})
                #else:
                #    print(f"{(int(from_l1_level), int(from_l2_level), int(from_l3_level), int(from_l4_level), int(from_l5_level), int(from_l6_level), int(from_l7_level), int(i['value_l8']))} is not a leaf nor an inner model")

            elif from_l1_level and from_l2_level != None and from_l3_level != None and from_l4_level != None and from_l5_level != None and from_l6_level != None:
                if modelclass == "C": i['value_l7'] = i['value_leaf']
                #if ((int(from_l1_level), int(from_l2_level), int(from_l3_level), int(from_l4_level), int(from_l5_level), int(from_l6_level), int(i['value_l7'])) in existing_bucket_models):
                priority_q.append({f"{modelclass}.1.{from_l1_level}.{from_l2_level}.{from_l3_level}.{from_l4_level}.{from_l5_level}.{from_l6_level}." + str(i['value_l7']): i['votes_perc']})
                #else:
                #    print(f"{(int(from_l1_level), int(from_l2_level), int(from_l3_level), int(from_l4_level), int(from_l5_level), int(from_l6_level), int(i['value_l7']))} is not a leaf nor an inner model")

            elif from_l1_level and from_l2_level != None and from_l3_level != None and from_l4_level != None and from_l5_level != None:
                if modelclass == "C": i['value_l6'] = i['value_leaf']
                #if ((int(from_l1_level), int(from_l2_level), int(from_l3_level), int(from_l4_level), int(from_l5_level), int(i['value_l6'])) in existing_bucket_models):
                priority_q.append({f"{modelclass}.1.{from_l1_level}.{from_l2_level}.{from_l3_level}.{from_l4_level}.{from_l5_level}." + str(i['value_l6']): i['votes_perc']})
                #else:
                #    print(f"{(int(from_l1_level), int(from_l2_level), int(from_l3_level), int(from_l4_level), int(from_l5_level), int(i['value_l6']))} is not a leaf nor an inner model")

            elif from_l1_level and from_l2_level != None and from_l3_level != None and from_l4_level != None:
                if modelclass == "C": i['value_l5'] = i['value_leaf']
                #if ((int(from_l1_level), int(from_l2_level), int(from_l3_level),  int(from_l4_level), int(i['value_l5'])) in existing_bucket_models):
                priority_q.append({f"{modelclass}.1.{from_l1_level}.{from_l2_level}.{from_l3_level}.{from_l4_level}." + str(i['value_l5']): i['votes_perc']})
                #else:
                #    print(f"{(int(from_l1_level), int(from_l2_level), int(from_l3_level), int(from_l4_level), int(i['value_l5']))} is not a leaf nor an inner model")

            elif from_l1_level and from_l2_level != None and from_l3_level != None:
                if modelclass == "C": i['value_l4'] =i['value_leaf']
                #if ((int(from_l1_level), int(from_l2_level), int(from_l3_level), int(i['value_l4'])) in existing_bucket_models):
                priority_q.append({f"{modelclass}.1.{from_l1_level}.{from_l2_level}.{from_l3_level}." + str(i['value_l4']): i['votes_perc']})
                #else:
                #    print(f"{(int(from_l1_level), int(from_l2_level), int(from_l3_level), int(i['value_l4']))} is not a leaf nor an inner model")

            elif from_l1_level and from_l2_level != None:
                if modelclass == "C": i['value_l3'] = i['value_leaf']
                #if ((int(from_l1_level), int(from_l2_level), int(i['value_l3'])) in existing_bucket_models):
                priority_q.append({f"{modelclass}.1.{from_l1_level}.{from_l2_level}." + str(i['value_l3']): i['votes_perc']})
                #else:
                #    print(f"{(int(from_l1_level), int(from_l2_level), int(i['value_l3']))} is not a leaf nor an inner model")

            elif from_l1_level != None:
                if modelclass == "C": i['value_l2'] = i['value_leaf']
                #print("C " + i['value_l2'] + " "+ i['value_leaf'])
                #print((from_l1_level, i['value_l2']))
                #if ((int(from_l1_level), int(i['value_l2'])) in existing_bucket_models):
                priority_q.append({f"{modelclass}.1.{from_l1_level}." + str(i['value_l2']): i['votes_perc']})
                #else:
                #    print(f"{(int(from_l1_level), int(i['value_l2']))} is not a leaf nor an inner model")
            else:
                priority_q.append({'M.1.' + str(i['value_l1']): i['votes_perc']})

        return priority_q
    """
    def add_level_to_queue(self, priority_q, probs, from_l1_level=None, from_l2_level=None, from_l3_level=None, from_l4_level=None, from_l5_level=None, from_l6_level=None, pop=True, value="value_l7"):
        if len(priority_q) != 0:
            if pop:
                priority_q.pop()
        modelclass = "M"
        if value == "value_leaf":
            modelclass = "C"
        for i in probs:
            #if i['votes_perc'] != 0:
            if from_l1_level and from_l2_level != None and from_l3_level != None and from_l4_level != None and from_l5_level != None and from_l6_level != None:
                if modelclass == "C": 
                    if 'value_leaf' in i:
                        i['value_l7'] =  i['value_leaf']
                    else:
                        i['value_l7'] = from_l6_level
                priority_q.append({f"{modelclass}.1.{from_l1_level}.{from_l2_level}.{from_l3_level}.{from_l4_level}.{from_l5_level}.{from_l6_level}." + str(i['value_l7']): i['votes_perc']})
            elif from_l1_level and from_l2_level != None and from_l3_level != None and from_l4_level != None and from_l5_level != None:
                if modelclass == "C": 
                    if 'value_leaf' in i:
                        i['value_l6'] =  i['value_leaf']
                    else:
                        i['value_l6'] = from_l5_level
                priority_q.append({f"{modelclass}.1.{from_l1_level}.{from_l2_level}.{from_l3_level}.{from_l4_level}.{from_l5_level}." + str(i['value_l6']): i['votes_perc']})
            elif from_l1_level and from_l2_level != None and from_l3_level != None and from_l4_level != None:
                if modelclass == "C": 
                    if 'value_leaf' in i:
                        i['value_l5'] =  i['value_leaf']
                    else:
                        i['value_l5'] = from_l4_level
                priority_q.append({f"{modelclass}.1.{from_l1_level}.{from_l2_level}.{from_l3_level}.{from_l4_level}." + str(i['value_l5']): i['votes_perc']})
            elif from_l1_level and from_l2_level != None and from_l3_level != None:
                if modelclass == "C": 
                    if 'value_leaf' in i:
                        i['value_l4'] =  i['value_leaf']
                    else:
                        i['value_l4'] = from_l3_level
                priority_q.append({f"{modelclass}.1.{from_l1_level}.{from_l2_level}.{from_l3_level}." + str(i['value_l4']): i['votes_perc']})
            elif from_l1_level and from_l2_level != None:
                if modelclass == "C": 
                    if 'value_leaf' in i:
                        i['value_l3'] =  i['value_leaf']
                    else:
                        i['value_l3'] = from_l2_level
                priority_q.append({f"{modelclass}.1.{from_l1_level}.{from_l2_level}." + str(i['value_l3']): i['votes_perc']})
            elif from_l1_level != None:
                if modelclass == "C": 
                    if 'value_leaf' in i:
                        i['value_l2'] =  i['value_leaf']
                    else:
                        i['value_l2'] = from_l1_level
                priority_q.append({f"{modelclass}.1.{from_l1_level}." + str(i['value_l2']): i['votes_perc']})
            else:
                priority_q.append({'M.1.' + str(i['value_l1']): i['votes_perc']})

        return priority_q
    """
    def is_rf(self, model):
        return type(model) is RandomForestClassifier

    def process_node_2(self, priority_q, x, gts, model_stack, over_approx_dict=None, ignore_gts=False, debug=False):
        is_L1_hit = False; is_L2_hit = False; is_leaf_hit = False
        popped = priority_q.pop(0)
        node_to_process = list(popped.keys())[0]
        model_label = node_to_process.split('.')
        processed_node = list(popped.keys())[0]
        if over_approx_dict and node_to_process in over_approx_dict:
            #if debug:
            #    print(f"Popped {popped} (symlink to {node_to_process})")
            old_node_to_process = node_to_process
            old_popped = popped
            node_to_process = over_approx_dict[node_to_process]
            processed_node = [node_to_process]
            prev_models = "."     
            for node in node_to_process.split(".")[2:-1]:
                processed_node.append(f"M.1{prev_models}{node}")
                prev_models += node + "."
            if debug:
                print(f"Popped {old_popped} (symlink to {node_to_process})")
                print(f"New popped: {processed_node}")
            model_label = node_to_process.split('.')
            if len(model_label) == 4: is_L1_hit = True; is_L2_hit = True
        elif debug: print(f"Popped {popped}")
        n_non_na_gts = len(gts)-sum(math.isnan(x) for x in gts)
        model_exists = True
        if len(model_label) == 3:
            try:
                model = model_stack[1][(int(model_label[(-1)]) - 1)]
            except IndexError:
                model_exists = False
            
            if not model_exists or model == [] or n_non_na_gts == 1:
                try:
                    #leaf_model = model_stack[2][0][(int(model_label[(-1)]))]
                    #if self.is_rf(leaf_model):
                    #probs = get_classification_probs_per_level(x,leaf_model, value='value_leaf')
                    #priority_q = self.add_level_to_queue(priority_q, probs, (model_label[(-1)]), pop=False, value='value_leaf')
                    priority_q = self.add_level_to_queue_8(priority_q, [{"votes_perc":1.0, "value_leaf": (model_label[(-1)])}], (model_label[(-1)]), pop=False, value='value_leaf')

                except (IndexError, AttributeError) as e:
                    if debug: print(f"Didn't find leaf model for {model_label}")
            else: # and not str(gts[1]) == "nan":
                if model_exists:
                    if gts[0] == int(model_label[(-1)]):
                        is_L1_hit = True
                    try:
                        probs = get_classification_probs_per_level(x, model, value='value_l2')
                        priority_q = self.add_level_to_queue(priority_q, probs, (model_label[(-1)]), pop=False)
                    except AttributeError:
                        if debug: print(f"Model was not trained {model_label}")

        if len(model_label) == 4:
            model_exists = False
            
            if model_label[0] == "C":
                if gts[0] == int(model_label[(-2)]):
                    is_leaf_hit = True
            elif not model_exists or model == [] or n_non_na_gts == 2:
                try:
                    #leaf_model = model_stack[2][1][int(model_label[(-2)])][int(model_label[(-1)])]
                    #print(leaf_model)
                    #if self.is_rf(leaf_model):
                        #print(model_label)
                        #print(leaf_model)
                    #probs = get_classification_probs_per_level(x, leaf_model, value='value_leaf')
                        #priority_q = self.add_level_to_queue(priority_q, probs, (model_label[(-2)]), (model_label[(-1)]), pop=False, value='value_leaf')
                    priority_q = self.add_level_to_queue_8(priority_q, [{"votes_perc":1.0, "value_leaf": (model_label[(-1)])}], (model_label[(-2)]), (model_label[(-1)]), pop=False, value='value_leaf')

                except (IndexError, AttributeError) as e:
                    if debug: print(f"Didn't find leaf model for {model_label}")
                    #priority_q = self.add_level_to_queue(priority_q, [{"votes_perc":1.0, "value_l3": (model_label[(-1)])}], (model_label[(-2)]), (model_label[(-1)]), pop=False, value='value_leaf')
            else: # and not str(gts[2]) == "nan":
                if model_exists:
                    if gts[1] == int(model_label[(-1)]) and gts[0] == int(model_label[(-2)]):
                        is_L2_hit = True
                    probs = get_classification_probs_per_level(x, model, value='value_l3')
                    priority_q = self.add_level_to_queue(priority_q, probs, (model_label[(-2)]), (model_label[(-1)]), pop=False)
                    #priority_q = self.add_level_to_queue_8(priority_q, [{"votes_perc":1.0, "value_leaf": (model_label[(-1)])}], (model_label[(-3)]), (model_label[(-2)]), (model_label[(-1)]), pop=False,value='value_leaf')

        if len(model_label) == 5:
            if model_label[0] == "C":
                if gts[0] == int(model_label[(-3)]) and gts[1] == int(model_label[(-2)]):
                    is_leaf_hit = True

        priority_q = sorted(priority_q, key=(lambda i: list(i.values())), reverse=True)
        if debug:
            print(f"L[2-3] added - PQ: {priority_q}, ...\n")
        return (priority_q, processed_node, is_L1_hit, is_L2_hit, is_leaf_hit)

    """
    def process_node(self, priority_q, x, gts, model_stack, over_approx_dict=None, ignore_gts=False, debug=False):
        is_L1_hit = False; is_L2_hit = False; is_L3_hit = False; is_L4_hit = False; is_L5_hit = False; is_L6_hit = False; is_leaf_hit = False
        popped = priority_q.pop(0)
        node_to_process = list(popped.keys())[0]
        model_label = node_to_process.split('.')
        processed_node = list(popped.keys())[0]
        if over_approx_dict and node_to_process in over_approx_dict:
            #if debug:
            #    print(f"Popped {popped} (symlink to {node_to_process})")
            old_node_to_process = node_to_process
            old_popped = popped
            node_to_process = over_approx_dict[node_to_process]
            processed_node = [node_to_process]
            prev_models = "."     
            for node in node_to_process.split(".")[2:-1]:
                processed_node.append(f"M.1{prev_models}{node}")
                prev_models += node + "."
            if debug:
                print(f"Popped {old_popped} (symlink to {node_to_process})")
                print(f"New popped: {processed_node}")
            model_label = node_to_process.split('.')
            if len(model_label) == 4: is_L1_hit = True; is_L2_hit = True
            if len(model_label) == 5: is_L1_hit = True; is_L2_hit = True; is_L3_hit = True
        elif debug: print(f"Popped {popped}")
        n_non_na_gts = len(gts)-sum(math.isnan(x) for x in gts)
        model_exists = True
        
        if len(model_label) == 3:
            try:
                model = model_stack[1][(int(model_label[(-1)]) - 1)]
            except IndexError:
                model_exists = False
            
            if not model_exists or model == [] or n_non_na_gts == 1:
                try:
                    leaf_model = model_stack[6][0][(int(model_label[(-1)]))]
                    if self.is_rf(leaf_model):
                        probs = get_classification_probs_per_level(x,leaf_model, value='value_leaf')
                        priority_q = self.add_level_to_queue(priority_q, probs, (model_label[(-1)]), pop=False, value='value_leaf')
                except (IndexError, AttributeError) as e:
                    if debug: print(f"Didn't find leaf model for {model_label}")
            else: # and not str(gts[1]) == "nan":
                if model_exists:
                    if gts[0] == int(model_label[(-1)]):
                        is_L1_hit = True
                    try:
                        probs = get_classification_probs_per_level(x, model, value='value_l2')
                        priority_q = self.add_level_to_queue(priority_q, probs, (model_label[(-1)]), pop=False)
                    except AttributeError:
                        if debug: print(f"Model was not trained {model_label}")

        if len(model_label) == 4:
            try:
                model = model_stack[2][(int(model_label[(-2)]))][(int(model_label[(-1)]))]
            except IndexError:
                model_exists = False
            
            if model_label[0] == "C":
                if gts[0] == int(model_label[(-2)]):
                    is_leaf_hit = True
            elif not model_exists or model == [] or n_non_na_gts == 2:
                try:
                    leaf_model = model_stack[6][1][int(model_label[(-2)])][int(model_label[(-1)])]
                    if self.is_rf(leaf_model):
                        #print(model_label)
                        #print(leaf_model)
                        probs = get_classification_probs_per_level(x, leaf_model, value='value_leaf')
                        priority_q = self.add_level_to_queue(priority_q, probs, (model_label[(-2)]), (model_label[(-1)]), pop=False, value='value_leaf')
                except (IndexError, AttributeError) as e:
                    if debug: print(f"Didn't find leaf model for {model_label}")
                    priority_q = self.add_level_to_queue(priority_q, [{"votes_perc":1.0, "value_l3": (model_label[(-1)])}], (model_label[(-2)]), (model_label[(-1)]), pop=False, value='value_leaf')
            else: # and not str(gts[2]) == "nan":
                if model_exists:
                    if gts[1] == int(model_label[(-1)]) and gts[0] == int(model_label[(-2)]):
                        is_L2_hit = True
                    probs = get_classification_probs_per_level(x, model, value='value_l3')
                    priority_q = self.add_level_to_queue(priority_q, probs, (model_label[(-2)]), (model_label[(-1)]), pop=False)

        if len(model_label) == 5:
            try:
                model =  model_stack[3][(int(model_label[(-3)]))][(int(model_label[(-2)]))][(int(model_label[(-1)]))]
            except IndexError:
                model_exists = False

            if model_label[0] == "C":
                if gts[1] == int(model_label[(-2)]) and gts[0] == int(model_label[(-3)]):
                    is_leaf_hit = True
            elif not model_exists or model == [] or n_non_na_gts == 3:
                try:
                    leaf_model = (model_stack[6][2][(int(model_label[(-3)]))][(int(model_label[(-2)]))][(int(model_label[(-1)]))])
                    if self.is_rf(leaf_model):
                        probs = get_classification_probs_per_level(x, leaf_model, value='value_leaf')
                        priority_q = self.add_level_to_queue(priority_q, probs, (model_label[(-3)]), (model_label[(-2)]), (model_label[(-1)]), pop=False,value='value_leaf')
                except (IndexError, AttributeError) as e:
                    if debug: print(f"Didn't find leaf model for {model_label}")
                    priority_q = self.add_level_to_queue(priority_q, [{"votes_perc":1.0, "value_l4": (model_label[(-1)])}], (model_label[(-3)]), (model_label[(-2)]), (model_label[(-1)]), pop=False, value='value_leaf')
            else: # and not str(gts[5]) == "nan":
                if model_exists:# and not str(gts[3]) == "nan":
                    if gts[2] == int(model_label[(-1)]) and gts[1] == int(model_label[(-2)]) and  gts[0] == int(model_label[(-3)]):
                        is_L3_hit = True
                    probs = get_classification_probs_per_level(x, model, value='value_l4')
                    priority_q = self.add_level_to_queue(priority_q, probs, (model_label[(-3)]), (model_label[(-2)]), (model_label[(-1)]), pop=False)

        if len(model_label) == 6:
            try:
                model = model_stack[4][(int(model_label[(-4)]))][(int(model_label[(-3)]))][(int(model_label[(-2)]))][(int(model_label[(-1)]))]
            except IndexError:
                model_exists = False

            if model_label[0] == "C":
                if gts[2] == int(model_label[(-2)]) and gts[1] == int(model_label[(-3)]) and gts[0] == int(model_label[(-4)]):
                    is_leaf_hit = True
            elif not model_exists or model == [] or n_non_na_gts == 4:
                try:
                    leaf_model = model_stack[6][3][int(model_label[(-4)])][int(model_label[(-3)])][int(model_label[(-2)])][(int(model_label[(-1)]))]
                    if self.is_rf(leaf_model):
                        probs = get_classification_probs_per_level(x, leaf_model, value='value_leaf')
                        priority_q = self.add_level_to_queue(priority_q, probs, (model_label[(-4)]), (model_label[(-3)]), (model_label[(-2)]), (model_label[(-1)]), pop=False, value='value_leaf')
                except (IndexError, AttributeError) as e:
                    if debug: print(f"Didn't find leaf model for {model_label}")
                    priority_q = self.add_level_to_queue(priority_q, [{"votes_perc":1.0, "value_l5": (model_label[(-1)])}], (model_label[(-4)]), (model_label[(-3)]), (model_label[(-2)]), (model_label[(-1)]), pop=False, value='value_leaf')

            else: # and not str(gts[5]) == "nan":
                if model_exists:
                    if gts[3] == int(model_label[(-1)]) and gts[2] == int(model_label[(-2)]) and gts[1] == int(model_label[(-3)]) and gts[0] == int(model_label[(-4)]):
                        is_L4_hit = True
                    probs = get_classification_probs_per_level(x, model, value='value_l5')
                    priority_q = self.add_level_to_queue(priority_q, probs, (model_label[(-4)]), (model_label[(-3)]), (model_label[(-2)]), (model_label[(-1)]), pop=False)
        
        if len(model_label) == 7:
            try:
                model = model_stack[5][(int(model_label[(-5)]))][(int(model_label[(-4)]))][(int(model_label[(-3)]))][(int(model_label[(-2)]))][(int(model_label[(-1)]))]
            except IndexError:
                model_exists = False

            if model_label[0] == "C":
                if gts[3] == int(model_label[(-2)]) and gts[2] == int(model_label[(-3)]) and gts[1] == int(model_label[(-4)]) and gts[0] == int(model_label[(-5)]):
                    is_leaf_hit = True
            elif not model_exists or model == [] or n_non_na_gts == 5:
                try:
                    leaf_model = model_stack[6][4][(int(model_label[(-5)]))][(int(model_label[(-4)]))][(int(model_label[(-3)]))][(int(model_label[(-2)]))][(int(model_label[(-1)]))]
                    if self.is_rf(leaf_model):
                        probs = get_classification_probs_per_level(x, leaf_model, value='value_leaf')
                        priority_q = self.add_level_to_queue(priority_q, probs, (model_label[(-5)]), (model_label[(-4)]), (model_label[(-3)]), (model_label[(-2)]), (model_label[(-1)]), pop=False, value='value_leaf')

                except (IndexError, AttributeError) as e:
                    if debug: print(f"Didn't find leaf model for {model_label}")
                    priority_q = self.add_level_to_queue(priority_q, [{"votes_perc":1.0, "value_l6": (model_label[(-1)])}], (model_label[(-5)]), (model_label[(-4)]), (model_label[(-3)]), (model_label[(-2)]), (model_label[(-1)]), pop=False, value='value_leaf')

            else: # and not str(gts[5]) == "nan":
                if model_exists:
                    if gts[4] == int(model_label[(-1)]) and gts[3] == int(model_label[(-2)]) and gts[2] == int(model_label[(-3)]) and gts[1] == int(model_label[(-4)]) and gts[0] == int(model_label[(-5)]):
                        is_L5_hit = True
                    probs = get_classification_probs_per_level(x, model, value='value_l6')
                    priority_q = self.add_level_to_queue(priority_q, probs, (model_label[(-5)]), (model_label[(-4)]), (model_label[(-3)]), (model_label[(-2)]), (model_label[(-1)]), pop=False)

        if len(model_label) == 8:
            if model_label[0] == "C":
                if gts[4] == int(model_label[(-2)]) and gts[3] == int(model_label[(-3)]) and gts[2] == int(model_label[(-4)]) and gts[1] == int(model_label[(-5)]) and gts[0] == int(model_label[(-6)]):
                    is_leaf_hit = True
            else:
                try:
                    leaf_model = model_stack[6][5][(int(model_label[(-6)]))][(int(model_label[(-5)]))][(int(model_label[(-4)]))][(int(model_label[(-3)]))][(int(model_label[(-2)]))][(int(model_label[(-1)]))]
                    if self.is_rf(leaf_model):
                        probs = get_classification_probs_per_level(x, leaf_model, value='value_leaf')
                        priority_q = self.add_level_to_queue(priority_q, probs, (model_label[(-6)]), (model_label[(-5)]), (model_label[(-4)]), (model_label[(-3)]), (model_label[(-2)]), (model_label[(-1)]), pop=False, value='value_leaf')
                except (IndexError, AttributeError) as e:
                    if debug: print(f"Didn't find leaf model for {model_label}")
                    priority_q = self.add_level_to_queue(priority_q, [{"votes_perc":1.0, "value_l7": (model_label[(-1)])}], (model_label[(-6)]), (model_label[(-5)]), (model_label[(-4)]), (model_label[(-3)]), (model_label[(-2)]), (model_label[(-1)]), pop=False, value='value_leaf')
                
        if len(model_label) == 9 and model_label[0] == "C":
            if gts[5] == int(model_label[(-2)]) and gts[4] == int(model_label[(-3)]) and gts[3] == int(model_label[(-4)]) and gts[2] == int(model_label[(-5)]) and gts[1] == int(model_label[(-6)]) and gts[0] == int(model_label[(-7)]):
                is_leaf_hit = True
        priority_q = sorted(priority_q, key=(lambda i: list(i.values())), reverse=True)
        if debug:
            print(f"L[2-3] added - PQ: {priority_q}, ...\n")
        return (priority_q, processed_node, is_L1_hit, is_L2_hit, is_L3_hit, is_L4_hit, is_L5_hit, is_L6_hit, is_leaf_hit)
    """

    def process_node_8(self, priority_q, x, gts, gts_preds, mapping, model_stack, existing_buckets=None, ignore_gts=False, custom_classifier=None, debug=False):
        is_L1_hit = False; is_L2_hit = False; is_L3_hit = False; is_L4_hit = False; is_L5_hit = False; is_L6_hit = False; is_L7_hit = False; is_L8_hit = False; is_leaf_hit = False
        popped = priority_q.pop(0)
        node_to_process = list(popped.keys())[0]
        model_label = node_to_process.split('.')
        #print(model_label)
        old_node_to_process = None
        if debug: print(f"Popped {popped}")
        #n_non_na_gts = len(gts)-sum(math.isnan(x) for x in gts)
        #print(f"n_non_na_gts: {n_non_na_gts}")
        s_len = len(priority_q)
        model_exists = True
        #s = time.time()
        if len(model_label) == 3:
            preds_index = [int(model_label[-1])]
            if preds_index in mapping[0]:
                stack_index = mapping[0].index(preds_index)
                model = model_stack[1][stack_index]
                probs = get_classification_probs_per_level(x, model, custom_classifier=custom_classifier, value='value_l2')
                priority_q = self.add_level_to_queue_8(priority_q, probs, (model_label[(-1)]), pop=False)

            #model = model_stack[1][(int(model_label[(-1)]))]
            #except IndexError:
            #    model_exists = False

            #if model_exists:
            if gts[0] == int(model_label[(-1)]):
                is_L1_hit = True
            #s = time.time()
            #probs = get_classification_probs_per_level(x, model, value='value_l2')
            #e = time.time()
            #print(probs)
            #print("here: ", n_non_na_gts)
            #print(probs)
            #priority_q = self.add_level_to_queue_8(priority_q, probs, (model_label[(-1)]), pop=False)

            #priority_q = self.add_level_to_queue_8(priority_q, [{"votes_perc":1.0, "value_leaf": (model_label[(-1)])}], (model_label[(-2)]), (model_label[(-1)]), pop=False, value='value_leaf')

        if len(model_label) == 4:
            preds_index = None
            if len(gts_preds) >= 2: preds_index = [int(model_label[-2]), int(model_label[-1])]
            #print(f"{preds_index} not in {mapping[1]}")
            if preds_index in mapping[1]:
                stack_index = mapping[1].index(preds_index)
                model = model_stack[2][stack_index]
            else:
                model_exists = False
                #print(f"{preds_index} not in mapping[1]")

            #else: # and not str(gts[2]) == "nan":
            if model_exists:
                if gts[1] == int(model_label[(-1)]) and gts[0] == int(model_label[(-2)]):
                    is_L2_hit = True
                #s = time.time()
                probs = get_classification_probs_per_level(x, model, custom_classifier=custom_classifier, value='value_l3')
                #e = time.time()
                priority_q = self.add_level_to_queue_8(priority_q, probs, (model_label[(-2)]), (model_label[(-1)]), pop=False)
            elif (int(model_label[-2]), int(model_label[-1])) in existing_buckets:# if n_non_na_gts == 2:
                node_to_process = "C" + node_to_process[1:]
                #print("here")
                #sort = False
                #priority_q = self.add_level_to_queue_8(priority_q, [{"votes_perc":1.0, "value_leaf": (model_label[(-1)])}], (model_label[(-2)]), (model_label[(-1)]), pop=False, value='value_leaf')

        if len(model_label) == 5:
            preds_index = None
            if len(gts_preds) >= 3: preds_index = [int(model_label[-3]), int(model_label[-2]), int(model_label[-1])]
            if preds_index in mapping[2]:
                stack_index = mapping[2].index(preds_index)
                #print(stack_index)
                model = model_stack[3][stack_index]
            else:
                model_exists = False

            #if not model_exists and n_non_na_gts == 3:
            #priority_q = self.add_level_to_queue_8(priority_q, [{"votes_perc":1.0, "value_leaf": (model_label[(-1)])}], (model_label[(-3)]), existing_buckets, mapping, (model_label[(-2)]), (model_label[(-1)]), pop=False,value='value_leaf')

            if model_exists:# and not str(gts[3]) == "nan":
                if gts[2] == int(model_label[(-1)]) and gts[1] == int(model_label[(-2)]) and gts[0] == int(model_label[(-3)]):
                    is_L3_hit = True
                probs = get_classification_probs_per_level(x, model, custom_classifier=custom_classifier, value='value_l4')
                priority_q = self.add_level_to_queue(priority_q, probs, (model_label[(-3)]), (model_label[(-2)]), (model_label[(-1)]), pop=False)
            elif (int(model_label[-3]), int(model_label[-2]), int(model_label[-1])) in existing_buckets:
                node_to_process = "C" + node_to_process[1:]
                #priority_q = self.add_level_to_queue_8(priority_q, [{"votes_perc":1.0, "value_leaf": (model_label[(-1)])}], (model_label[(-3)]), (model_label[(-2)]), (model_label[(-1)]), pop=False,value='value_leaf')


        if len(model_label) == 6:
            preds_index = None
            if len(gts_preds) >= 4: preds_index = [int(model_label[-4]), int(model_label[-3]), int(model_label[-2]), int(model_label[-1])]
            if preds_index in mapping[3]:
                stack_index = mapping[3].index(preds_index)
                model = model_stack[4][stack_index]
            else:
                model_exists = False

            if model_exists:
                if gts[3] == int(model_label[(-1)]) and gts[2] == int(model_label[(-2)]) and gts[1] == int(model_label[(-3)]) and gts[0] == int(model_label[(-4)]):
                    is_L4_hit = True
                probs = get_classification_probs_per_level(x, model, custom_classifier=custom_classifier, value='value_l5')
                priority_q = self.add_level_to_queue_8(priority_q, probs, (model_label[(-4)]), (model_label[(-3)]), (model_label[(-2)]), (model_label[(-1)]), pop=False)
            elif (int(model_label[-4]), int(model_label[-3]), int(model_label[-2]), int(model_label[-1])) in existing_buckets:
                node_to_process = "C" + node_to_process[1:]
                #priority_q = self.add_level_to_queue_8(priority_q, [{"votes_perc":1.0, "value_leaf": (model_label[(-1)])}], (model_label[(-4)]), (model_label[(-3)]), (model_label[(-2)]), (model_label[(-1)]), pop=False, value='value_leaf')

        if len(model_label) == 7:
            preds_index = None
            if len(gts_preds) >= 5: preds_index = [int(model_label[-5]), int(model_label[-4]), int(model_label[-3]), int(model_label[-2]), int(model_label[-1])]
            if preds_index in mapping[4]:
                stack_index = mapping[4].index(preds_index)
                model = model_stack[5][stack_index]
            else:
                model_exists = False

            if model_exists:
                if gts[4] == int(model_label[(-1)]) and gts[3] == int(model_label[(-2)]) and gts[2] == int(model_label[(-3)]) and gts[1] == int(model_label[(-4)]) and gts[0] == int(model_label[(-5)]):
                    is_L5_hit = True
                probs = get_classification_probs_per_level(x, model, custom_classifier=custom_classifier, value='value_l6')
                priority_q = self.add_level_to_queue_8(priority_q, probs, (model_label[(-5)]), (model_label[(-4)]), (model_label[(-3)]), (model_label[(-2)]), (model_label[(-1)]), pop=False)
            elif (int(model_label[-5]), int(model_label[-4]), int(model_label[-3]), int(model_label[-2]), int(model_label[-1])) in existing_buckets:
                node_to_process = "C" + node_to_process[1:]
                #priority_q = self.add_level_to_queue_8(priority_q, [{"votes_perc":1.0, "value_leaf": (model_label[(-1)])}], (model_label[(-5)]), (model_label[(-4)]), (model_label[(-3)]), (model_label[(-2)]), (model_label[(-1)]), pop=False, value='value_leaf')


        if len(model_label) == 8:
            preds_index = None
            if len(gts_preds) >= 6: preds_index = [int(model_label[-6]), int(model_label[-5]), int(model_label[-4]), int(model_label[-3]), int(model_label[-2]), int(model_label[-1])]
            if preds_index in mapping[5]:
                stack_index = mapping[5].index(preds_index)
                model = model_stack[6][stack_index]
            else:
                model_exists = False

            if model_exists:
                if gts[5] == int(model_label[(-1)]) and gts[4] == int(model_label[(-2)]) and gts[3] == int(model_label[(-3)]) and gts[2] == int(model_label[(-4)]) and gts[1] == int(model_label[(-5)]) and gts[0] == int(model_label[(-6)]):
                    is_L6_hit = True
                probs = get_classification_probs_per_level(x, model, custom_classifier=custom_classifier, value='value_l7')
                priority_q = self.add_level_to_queue_8(priority_q, probs, (model_label[(-6)]), (model_label[(-5)]), (model_label[(-4)]), (model_label[(-3)]), (model_label[(-2)]), (model_label[(-1)]), pop=False)
            elif (int(model_label[-6]), int(model_label[-5]), int(model_label[-4]), int(model_label[-3]), int(model_label[-2]), int(model_label[-1])) in existing_buckets:
                node_to_process = "C" + node_to_process[1:]
                #priority_q = self.add_level_to_queue_8(priority_q, [{"votes_perc":1.0, "value_leaf": (model_label[(-1)])}], (model_label[(-6)]), (model_label[(-5)]), (model_label[(-4)]), (model_label[(-3)]), (model_label[(-2)]), (model_label[(-1)]), pop=False, value='value_leaf')


        if len(model_label) == 9:
            preds_index = None
            if len(gts_preds) >= 7: preds_index = [int(model_label[-7]), int(model_label[-6]), int(model_label[-5]), int(model_label[-4]), int(model_label[-3]), int(model_label[-2]), int(model_label[-1])]
            if preds_index in mapping[6]:
                stack_index = mapping[6].index(preds_index)
                model = model_stack[7][stack_index]
            else:
                model_exists = False
            
            if model_exists:
                if gts[6] == int(model_label[(-1)]) and gts[5] == int(model_label[(-2)]) and gts[4] == int(model_label[(-3)]) and gts[3] == int(model_label[(-4)]) and gts[2] == int(model_label[(-5)]) and gts[1] == int(model_label[(-6)]) and gts[0] == int(model_label[(-7)]):
                    is_L7_hit = True
                probs = get_classification_probs_per_level(x, model, custom_classifier=custom_classifier, value='value_l8')
                priority_q = self.add_level_to_queue_8(priority_q, probs, (model_label[(-7)]),  (model_label[(-6)]), (model_label[(-5)]), (model_label[(-4)]), (model_label[(-3)]), (model_label[(-2)]), (model_label[(-1)]), pop=False)
            elif (int(model_label[-7]), int(model_label[-6]), int(model_label[-5]), int(model_label[-4]), int(model_label[-3]), int(model_label[-2]), int(model_label[-1])) in existing_buckets:
                node_to_process = "C" + node_to_process[1:]
                #priority_q = self.add_level_to_queue_8(priority_q, [{"votes_perc":1.0, "value_leaf": (model_label[(-1)])}], (model_label[(-7)]), (model_label[(-6)]), (model_label[(-5)]), (model_label[(-4)]), (model_label[(-3)]), (model_label[(-2)]), (model_label[(-1)]), pop=False, value='value_leaf')


        if len(model_label) == 10:
            if gts[7] == int(model_label[(-1)]) and gts[6] == int(model_label[(-2)]) and gts[5] == int(model_label[(-3)]) and gts[4] == int(model_label[(-4)]) and gts[3] == int(model_label[(-5)]) and gts[2] == int(model_label[(-6)]) and gts[1] == int(model_label[(-7)]) and gts[0] == int(model_label[(-8)]):
                is_L8_hit = True
                #if n_non_na_gts == 8:
                node_to_process = "C" + node_to_process[1:]
                #priority_q = self.add_level_to_queue_8(priority_q, [{"votes_perc":1.0, "value_leaf": (model_label[(-1)])}], (model_label[(-8)]), (model_label[(-7)]), (model_label[(-6)]), (model_label[(-5)]), (model_label[(-4)]), (model_label[(-3)]), (model_label[(-2)]), (model_label[(-1)]), pop=False, value='value_leaf')

        #s = time.time()
        if node_to_process[0] != "C" and s_len != len(priority_q):
            #priority_q = sorted(priority_q, key=(lambda i: list(i.values())), reverse=True)
            priority_q.sort(key=(lambda i: list(i.values())), reverse=True)
            if debug:
                print(f"L[2-3] added - PQ: {priority_q}, ...\n")
        return (priority_q, node_to_process, is_L1_hit, is_L2_hit, is_L3_hit, is_L4_hit, is_L5_hit, is_L6_hit, is_L7_hit, is_L8_hit)

    def process_node_6(self, priority_q, x, gts, gts_preds, mapping, model_stack, existing_buckets=None, ignore_gts=False, custom_classifier=None, debug=False):
        is_L1_hit = False; is_L2_hit = False; is_L3_hit = False; is_L4_hit = False; is_L5_hit = False; is_L6_hit = False; is_leaf_hit = False
        popped = priority_q.pop(0)
        node_to_process = list(popped.keys())[0]
        model_label = node_to_process.split('.')
        #print(model_label)
        old_node_to_process = None
        if debug: print(f"Popped {popped}")
        #n_non_na_gts = len(gts)-sum(math.isnan(x) for x in gts)
        #print(f"n_non_na_gts: {n_non_na_gts}")
        s_len = len(priority_q)
        model_exists = True
        #s = time.time()
        if len(model_label) == 3:
            preds_index = [int(model_label[-1])]
            if preds_index in mapping[0]:
                stack_index = mapping[0].index(preds_index)
                model = model_stack[1][stack_index]
                probs = get_classification_probs_per_level(x, model, custom_classifier=custom_classifier, value='value_l2')
                priority_q = self.add_level_to_queue_8(priority_q, probs, (model_label[(-1)]), pop=False)

            if gts[0] == int(model_label[(-1)]):
                is_L1_hit = True

        if len(model_label) == 4:
            preds_index = None
            if len(gts_preds) >= 2: preds_index = [int(model_label[-2]), int(model_label[-1])]
            #print(f"{preds_index} not in {mapping[1]}")
            if preds_index in mapping[1]:
                stack_index = mapping[1].index(preds_index)
                model = model_stack[2][stack_index]
            else:
                model_exists = False
                #print(f"{preds_index} not in mapping[1]")

            #else: # and not str(gts[2]) == "nan":
            if model_exists:
                if gts[1] == int(model_label[(-1)]) and gts[0] == int(model_label[(-2)]):
                    is_L2_hit = True
                #s = time.time()
                probs = get_classification_probs_per_level(x, model, custom_classifier=custom_classifier, value='value_l3')
                #e = time.time()
                priority_q = self.add_level_to_queue_8(priority_q, probs, (model_label[(-2)]), (model_label[(-1)]), pop=False)
            elif (int(model_label[-2]), int(model_label[-1])) in existing_buckets:# if n_non_na_gts == 2:
                node_to_process = "C" + node_to_process[1:]
                #print("here")
                #sort = False
                #priority_q = self.add_level_to_queue_8(priority_q, [{"votes_perc":1.0, "value_leaf": (model_label[(-1)])}], (model_label[(-2)]), (model_label[(-1)]), pop=False, value='value_leaf')

        if len(model_label) == 5:
            preds_index = None
            if len(gts_preds) >= 3: preds_index = [int(model_label[-3]), int(model_label[-2]), int(model_label[-1])]
            if preds_index in mapping[2]:
                stack_index = mapping[2].index(preds_index)
                #print(stack_index)
                model = model_stack[3][stack_index]
            else:
                model_exists = False

            if model_exists:# and not str(gts[3]) == "nan":
                if gts[2] == int(model_label[(-1)]) and gts[1] == int(model_label[(-2)]) and gts[0] == int(model_label[(-3)]):
                    is_L3_hit = True
                probs = get_classification_probs_per_level(x, model, custom_classifier=custom_classifier, value='value_l4')
                priority_q = self.add_level_to_queue(priority_q, probs, (model_label[(-3)]), (model_label[(-2)]), (model_label[(-1)]), pop=False)
            elif (int(model_label[-3]), int(model_label[-2]), int(model_label[-1])) in existing_buckets:
                node_to_process = "C" + node_to_process[1:]
                #priority_q = self.add_level_to_queue_8(priority_q, [{"votes_perc":1.0, "value_leaf": (model_label[(-1)])}], (model_label[(-3)]), (model_label[(-2)]), (model_label[(-1)]), pop=False,value='value_leaf')


        if len(model_label) == 6:
            preds_index = None
            if len(gts_preds) >= 4: preds_index = [int(model_label[-4]), int(model_label[-3]), int(model_label[-2]), int(model_label[-1])]
            if preds_index in mapping[3]:
                stack_index = mapping[3].index(preds_index)
                model = model_stack[4][stack_index]
            else:
                model_exists = False

            if model_exists:
                if gts[3] == int(model_label[(-1)]) and gts[2] == int(model_label[(-2)]) and gts[1] == int(model_label[(-3)]) and gts[0] == int(model_label[(-4)]):
                    is_L4_hit = True
                probs = get_classification_probs_per_level(x, model, custom_classifier=custom_classifier, value='value_l5')
                priority_q = self.add_level_to_queue_8(priority_q, probs, (model_label[(-4)]), (model_label[(-3)]), (model_label[(-2)]), (model_label[(-1)]), pop=False)
            elif (int(model_label[-4]), int(model_label[-3]), int(model_label[-2]), int(model_label[-1])) in existing_buckets:
                node_to_process = "C" + node_to_process[1:]
                #priority_q = self.add_level_to_queue_8(priority_q, [{"votes_perc":1.0, "value_leaf": (model_label[(-1)])}], (model_label[(-4)]), (model_label[(-3)]), (model_label[(-2)]), (model_label[(-1)]), pop=False, value='value_leaf')

        if len(model_label) == 7:
            preds_index = None
            if len(gts_preds) >= 5: preds_index = [int(model_label[-5]), int(model_label[-4]), int(model_label[-3]), int(model_label[-2]), int(model_label[-1])]
            if preds_index in mapping[4]:
                stack_index = mapping[4].index(preds_index)
                model = model_stack[5][stack_index]
            else:
                model_exists = False

            if model_exists:
                if gts[4] == int(model_label[(-1)]) and gts[3] == int(model_label[(-2)]) and gts[2] == int(model_label[(-3)]) and gts[1] == int(model_label[(-4)]) and gts[0] == int(model_label[(-5)]):
                    is_L5_hit = True
                probs = get_classification_probs_per_level(x, model, custom_classifier=custom_classifier, value='value_l6')
                priority_q = self.add_level_to_queue_8(priority_q, probs, (model_label[(-5)]), (model_label[(-4)]), (model_label[(-3)]), (model_label[(-2)]), (model_label[(-1)]), pop=False)
            elif (int(model_label[-5]), int(model_label[-4]), int(model_label[-3]), int(model_label[-2]), int(model_label[-1])) in existing_buckets:
                node_to_process = "C" + node_to_process[1:]

        if len(model_label) == 8:
            if gts[7] == int(model_label[(-1)]) and gts[6] == int(model_label[(-2)]) and gts[5] == int(model_label[(-3)]) and gts[4] == int(model_label[(-4)]) and gts[3] == int(model_label[(-5)]) and gts[2] == int(model_label[(-6)]) and gts[1] == int(model_label[(-7)]) and gts[0] == int(model_label[(-8)]):
                is_L8_hit = True
                node_to_process = "C" + node_to_process[1:]

        #s = time.time()
        if node_to_process[0] != "C" and s_len != len(priority_q):
            #priority_q = sorted(priority_q, key=(lambda i: list(i.values())), reverse=True)
            priority_q.sort(key=(lambda i: list(i.values())), reverse=True)
            if debug:
                print(f"L[2-3] added - PQ: {priority_q}, ...\n")
        return (priority_q, node_to_process, is_L1_hit, is_L2_hit, is_L3_hit, is_L4_hit, is_L5_hit, is_L6_hit)


    def get_number_of_leaf_node_models(self, popped_nodes):
        set_ = set([e for e in popped_nodes if e[0] == 'C'])
        return len(set_)
        """
        c = 0
        for popped_node in popped_nodes:
            #for k, v in popped_node.items():
            if popped_node[0] == 'C':
                c += 1
        return c
        """

    def knn_search_2(self, df_1k, stop_cond_leaf, knns, model_stack, df_res, over_approx_dict=None, n_objects=1000, row=None, n_candidates=None, knn=30, debug=False):
        stats = []
        if row is not None:
            iterate = row
        else:
            iterate = df_1k[:n_objects]
        all_gts = []
        for i, o in iterate.iterrows():
            o_df_1 = int(o['object_id'])
            c_L1_L2_L3 = 0
            c_L1_L2 = 0
            c_L1 = 0
            if debug:
                print(f"Orig object: {o_df_1}")
            search_res = self.approximate_search_2(model_stack, df_res, o_df_1, over_approx_dict=over_approx_dict, steps_limit_leaf=stop_cond_leaf, end_on_exact_hit=False, debug=debug)
            if search_res['is_hit']:
                c_L1_L2_L3 += 1
            if debug:
                print(f"\nTrying to hit {search_res['popped_nodes']} n_leaf_models: {self.get_number_of_leaf_node_models(search_res['popped_nodes'])}")
            #popped_keys = [list(p_n.keys())[0] for p_n in search_res['popped_nodes']]
            popped_keys = [p_n for p_n in search_res['popped_nodes']]

            gts = []; gts_L1 = []; gts_L2 = []
            n_last_level = 0
            for p in popped_keys:
                model_label = p.split('.')
                if len(model_label) == 3:
                    gts_L1.append(int(model_label[(-1)]))
                if len(model_label) == 4:
                    if int(model_label[(-2)]) in gts_L1:
                        if model_label[0] == "C":
                            gts.append((int(model_label[(-2)])))
                        else:
                            gts_L2.append((int(model_label[(-2)]), int(model_label[(-1)])))
                if len(model_label) == 5:
                    if (int(model_label[(-3)]), int(model_label[(-2)])) in gts_L2:
                        if model_label[0] == "C":
                            gts.append((int(model_label[(-3)]), int(model_label[(-2)])))
            all_gts.append(gts)
            #if debug:
            if len(gts) == 0:
                print(f"\nIdentified pool of ids to hit: {gts}")
            df_subset = []
            for gt in gts:
                if type(gt) is int:
                    df_subset.append(df_res[(df_res['L1'] == gt)])
                elif len(gt) == 2:
                    df_subset.append(df_res[(df_res['L1_pred'] == gt[0]) & (df_res['L2_pred'] == gt[1])])

            df_subset = pd.concat(df_subset).drop_duplicates()
            if debug:
                print(df_subset.shape)
            if n_candidates is not None:
                df_subset = df_subset[:n_candidates]
                if debug:
                    print(df_subset)
            nn_object_ids = np.array((list(knns[str(o_df_1)].keys())), dtype=(np.int64))
            intersect = np.intersect1d(df_subset['object_id'].values, nn_object_ids)
            stats.append((o_df_1, intersect.shape[0] / knn))

        return stats, all_gts

    def get_sample_1k_objects(self, df_res):
        return df_res[df_res["object_id"].isin(self.get_knn_objects(path="/storage/plzen1/home/tslaninakova/learned-indexes/datasets/queries.data"))]

    def get_sample_1k_objects_profiset(self, df_res):
        return df_res[df_res["object_id"].isin(self.get_knn_objects(path="/storage/plzen1/home/tslaninakova/learned-indexes/datasets/profiset-queries.data"))]
    
    def get_sample_1k_objects_mocap(self, df_res):
        return df_res[df_res["object_id"].isin(self.get_knn_objects(path="/storage/brno6/home/tslaninakova/learned-indexes/datasets/mocap-queries.data", should_be_int=False))]

    def get_knn_gts(self, path=None):
        """ Loads object 30 knns ground truth file (json).

        Returns
        -------
        gt_knns: dict
            Ground truth for 30 kNN query
        """
        if path is None:
            path = self.knn_gts_file
        with open(path) as json_file:
            gt_knns = json.load(json_file)
        return gt_knns

    def knn_search_8(self, df_1k, mappings, knns, model_stack, df_res, existing_buckets, df_orig = None, struct_df=None, stop_cond_model=None,custom_classifier=None, existing_regions_dict=None, stop_cond_leaf=None, over_approx_dict=None, n_objects=1000, row=None, mindex=False, n_candidates=None, merged_paths=None, knn=30, debug=False):
        stats = []; intersects = []; times = []
        if row is not None:
            iterate = pd.DataFrame(row)
        else:
            iterate = df_1k[:n_objects]
        all_gts = []
        if df_orig is not None:
            ref_df = df_orig#.copy()
            for l in ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"]:
                df_orig[f"{l}_pred"] = df_orig[l]
            df_orig = df_orig.drop([f"{l}_pred" for l in ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"]], axis=1)
        else:
            ref_df = df_res
        ref_df_1 = ref_df[ref_df['L2_pred'].isna()]
        ref_df_2 = ref_df[ref_df['L3_pred'].isna()]
        ref_df_3 = ref_df[(ref_df['L4_pred'].isna()) & (~ref_df['L3_pred'].isna())]
        ref_df_4 = ref_df[ref_df['L5_pred'].isna() & (~ref_df['L4_pred'].isna()) & (~ref_df['L3_pred'].isna())]
        ref_df_5 = ref_df[ref_df['L6_pred'].isna() & (~ref_df['L5_pred'].isna()) & (~ref_df['L4_pred'].isna()) & (~ref_df['L3_pred'].isna())]
        ref_df_6 = ref_df[ref_df['L7_pred'].isna() & (~ref_df['L6_pred'].isna()) & (~ref_df['L5_pred'].isna()) & (~ref_df['L4_pred'].isna()) & (~ref_df['L3_pred'].isna())]
        ref_df_7 = ref_df[ref_df['L8_pred'].isna() & (~ref_df['L7_pred'].isna()) & (~ref_df['L6_pred'].isna()) & (~ref_df['L5_pred'].isna()) & (~ref_df['L4_pred'].isna()) & (~ref_df['L3_pred'].isna())]
        ref_df_8 = ref_df[~ref_df['L8_pred'].isna()]

        for n, (i, o) in enumerate(iterate.iterrows()):
            if mindex: print(f"{n}/{n_objects}")
            o_df_1 = int(o['object_id'])
            c_L1_L2_L3 = 0
            c_L1_L2 = 0
            c_L1 = 0
            if debug:
                print(f"Orig object: {o_df_1}")
            if mindex:
                s = time.time()
                #df_orig, 36126726, pivots_df, [], existing_regions_unique, labels=mindex.get_descriptive_col_names()+ ["object_id"], max_visited_models=1000, bucket_level=8, is_profi=False, debug=False
                search_res = approximate_search_mindex(df_orig, o_df_1, struct_df, L1_only_pivots= [], existing_regions=[], existing_regions_dict=existing_regions_dict, max_visited_models=stop_cond_model, bucket_level=8, labels=self.get_descriptive_col_names()+ ["object_id"], is_profi=False, debug=debug)
                #print(search_res)
                times.append(time.time() - s)
            else:
                s = time.time()
                if stop_cond_model is not None:
                    #%time li.approximate_search_8(stack, df_res, 81178044, mapping, steps_limit=5, existing_buckets=existing_bucket_models, end_on_exact_hit=False, debug=True)
                    search_res = self.approximate_search_8(model_stack, df_res, o_df_1, mappings, existing_buckets=existing_buckets, over_approx_dict=over_approx_dict, steps_limit=stop_cond_model, custom_classifier=custom_classifier, end_on_exact_hit=False, merged_paths=merged_paths, debug=debug)
                    times.append(time.time() - s)
                if search_res['is_hit']:
                    c_L1_L2_L3 += 1
                #print(f"n of leaf nodes: {search_res['leaf_nodes_hit']}")
            if debug:
                print(f"\nTrying to hit {search_res['popped_nodes']}")
            popped_keys = [p_n for p_n in search_res['popped_nodes']]
            gts = []; gts_L1 = []; gts_L2 = []; gts_L3 = []; gts_L4 = []; gts_L5 = []; gts_L6 = []; gts_L7 = []; gts_L8 = []
            n_last_level = 0
            for p in popped_keys:
                model_label = p.split('.')
                if len(model_label) == 3:
                    gts_L1.append(int(model_label[(-1)]))
                if len(model_label) == 4:
                    if int(model_label[(-2)]) in gts_L1:
                        #if model_label[0] == "C":
                        #    gts.append((int(model_label[(-2)])))
                        #else:
                        gts_L2.append((int(model_label[(-2)]), int(model_label[(-1)])))
                if len(model_label) == 5:
                    if (int(model_label[(-3)]), int(model_label[(-2)])) in gts_L2:
                        #if model_label[0] == "C":
                        #    gts.append((int(model_label[(-3)]), int(model_label[(-2)])))
                        #else:
                        gts_L3.append((int(model_label[(-3)]), int(model_label[(-2)]), int(model_label[(-1)])))
                if len(model_label) == 6:
                    if (int(model_label[(-4)]), int(model_label[(-3)]), int(model_label[(-2)])) in gts_L3:
                        #if model_label[0] == "C":
                        #    gts.append((int(model_label[(-4)]), int(model_label[(-3)]), int(model_label[(-2)])))
                        #lse:
                        gts_L4.append((int(model_label[(-4)]), int(model_label[(-3)]), int(model_label[(-2)]), int(model_label[(-1)])))
                if len(model_label) == 7:
                    if (int(model_label[(-5)]), int(model_label[(-4)]), int(model_label[(-3)]), int(model_label[(-2)])) in gts_L4:
                        #if model_label[0] == "C":
                        #    gts.append((int(model_label[(-5)]), int(model_label[(-4)]), int(model_label[(-3)]), int(model_label[(-2)])))
                        #else:
                        gts_L5.append((int(model_label[(-5)]), int(model_label[(-4)]), int(model_label[(-3)]), int(model_label[(-2)]), int(model_label[(-1)])))
                if len(model_label) == 8:
                    if (int(model_label[(-6)]), int(model_label[(-5)]), int(model_label[(-4)]), int(model_label[(-3)]), int(model_label[(-2)])) in gts_L5:
                        #if model_label[0] == "C":
                        #    gts.append((int(model_label[(-6)]), int(model_label[(-5)]), int(model_label[(-4)]), int(model_label[(-3)]), int(model_label[(-2)])))
                        #else:
                        gts_L6.append((int(model_label[(-6)]), int(model_label[(-5)]), int(model_label[(-4)]), int(model_label[(-3)]), int(model_label[(-2)]), int(model_label[(-1)])))
                if len(model_label) == 9:
                    if (int(model_label[(-7)]), int(model_label[(-6)]), int(model_label[(-5)]), int(model_label[(-4)]), int(model_label[(-3)]), int(model_label[(-2)])) in gts_L6:
                        #if model_label[0] == "C":
                        #    gts.append((int(model_label[(-7)]), int(model_label[(-6)]), int(model_label[(-5)]), int(model_label[(-4)]), int(model_label[(-3)]), int(model_label[(-2)])))
                        #else:
                        gts_L7.append((int(model_label[(-7)]), int(model_label[(-6)]), int(model_label[(-5)]), int(model_label[(-4)]), int(model_label[(-3)]), int(model_label[(-2)]), int(model_label[(-1)])))
                if len(model_label) == 10:
                    if (int(model_label[(-8)]), int(model_label[(-7)]), int(model_label[(-6)]), int(model_label[(-5)]), int(model_label[(-4)]), int(model_label[(-3)]), int(model_label[(-2)])) in gts_L7:
                        #if model_label[0] == "C":
                        #    gts.append((int(model_label[(-8)]), int(model_label[(-7)]), int(model_label[(-6)]), int(model_label[(-5)]), int(model_label[(-4)]), int(model_label[(-3)]), int(model_label[(-2)])))
                        #else:
                        gts_L8.append((int(model_label[(-8)]), int(model_label[(-7)]), int(model_label[(-6)]), int(model_label[(-5)]), int(model_label[(-4)]), int(model_label[(-3)]), int(model_label[(-2)]), int(model_label[(-1)])))
  
                if len(model_label) == 11 and model_label[0] == "C":
                    gts.append((int(model_label[(-9)]),int(model_label[(-8)]),int(model_label[(-7)]), int(model_label[(-6)]), int(model_label[(-5)]), int(model_label[(-4)]), int(model_label[(-3)]), int(model_label[(-2)]),int(model_label[(-1)])))
                
            all_gts.append(gts)
            for g in [gts_L1, gts_L2, gts_L3, gts_L4, gts_L5, gts_L6, gts_L7, gts_L8]:
                if g != []: gts.extend(g)
            if debug:
                print(f"\nIdentified pool of ids to hit: {gts}")# {gts_L1} {gts_L2} {gts_L3}, {gts_L4} {gts_L5} {gts_L6} {gts_L7}")
            df_subset = []
            s = time.time()
            for gt in gts:
                if type(gt) is int:
                    df_subset.append(ref_df_1[(ref_df_1['L1_pred'] == gt)])# & (ref_df['L2'].isna())])
                elif len(gt) == 2:
                    df_subset.append(ref_df_2[(ref_df_2['L1_pred'] == gt[0]) & (ref_df_2['L2_pred'] == gt[1])])
                elif len(gt) == 3:
                    df_subset.append(ref_df_3[(ref_df_3['L1_pred'] == gt[0]) & (ref_df_3['L2_pred'] == gt[1]) & (ref_df_3['L3_pred'] == gt[2])])
                elif len(gt) == 4:
                    df_subset.append(ref_df_4[(ref_df_4['L1_pred'] == gt[0]) & (ref_df_4['L2_pred'] == gt[1]) & (ref_df_4['L3_pred'] == gt[2]) & (ref_df_4['L4_pred'] == gt[3])])
                elif len(gt) == 5:
                    df_subset.append(ref_df_5[(ref_df_5['L1_pred'] == gt[0]) & (ref_df_5['L2_pred'] == gt[1]) & (ref_df_5['L3_pred'] == gt[2]) & (ref_df_5['L4_pred'] == gt[3]) & (ref_df_5['L5_pred'] == gt[4])])
                elif len(gt) == 6:
                    df_subset.append(ref_df_6[(ref_df_6['L1_pred'] == gt[0]) & (ref_df_6['L2_pred'] == gt[1]) & (ref_df_6['L3_pred'] == gt[2]) & (ref_df_6['L4_pred'] == gt[3]) & (ref_df_6['L5_pred'] == gt[4]) & (ref_df_6['L6_pred'] == gt[5])])
                elif len(gt) == 7:
                    df_subset.append(ref_df_7[(ref_df_7['L1_pred'] == gt[0]) & (ref_df_7['L2_pred'] == gt[1]) & (ref_df_7['L3_pred'] == gt[2]) & (ref_df_7['L4_pred'] == gt[3]) & (ref_df_7['L5_pred'] == gt[4]) & (ref_df_7['L6_pred'] == gt[5]) & (ref_df_7['L7_pred'] == gt[6])])
                elif len(gt) == 8:
                    df_subset.append(ref_df_8[(ref_df_8['L1_pred'] == gt[0]) & (ref_df_8['L2_pred'] == gt[1]) & (ref_df_8['L3_pred'] == gt[2]) & (ref_df_8['L4_pred'] == gt[3]) & (ref_df_8['L5_pred'] == gt[4]) & (ref_df_8['L6_pred'] == gt[5]) & (ref_df_8['L7_pred'] == gt[6]) & (ref_df_8['L8_pred'] == gt[7])])
            #print(time.time() - s)
            if df_subset != []:
                df_subset = pd.concat(df_subset).drop_duplicates()
                #return df_subset
                #return search_res, df_subset
                if debug:
                    print(df_subset.shape)
                if n_candidates is not None:
                    df_subset = df_subset[:n_candidates]
                    if debug:
                        print(df_subset)
                nn_object_ids = np.array((list(knns[str(o_df_1)].keys())), dtype=(np.int64))
                if df_subset.shape[0] != 0:
                    intersect = np.intersect1d(df_subset['object_id'].values, nn_object_ids)
                    if debug: print(intersect.shape[0] / knn)
                    intersects.append(intersect)
                    stats.append((o_df_1, intersect.shape[0] / knn, gts))
                else:
                    stats.append((o_df_1, 0, gts))
            else:
                stats.append((o_df_1, 0, None))

        return stats, np.array(times).mean()

    def knn_search_6(self, df_1k, mappings, knns, model_stack, df_res, existing_buckets, df_orig = None, struct_df=None, stop_cond_model=None,custom_classifier=None, existing_regions_dict=None, stop_cond_leaf=None, over_approx_dict=None, n_objects=1000, row=None, mindex=False, n_candidates=None, knn=30, debug=False):
        stats = []; intersects = []; times = []
        if row is not None:
            iterate = pd.DataFrame(row)
        else:
            iterate = df_1k[:n_objects]
        all_gts = []
        if df_orig is not None:
            ref_df = df_orig#.copy()
            for l in ["L1", "L2", "L3", "L4", "L5", "L6"]:
                df_orig[f"{l}_pred"] = df_orig[l]
            df_orig = df_orig.drop([f"{l}_pred" for l in ["L1", "L2", "L3", "L4", "L5", "L6"]], axis=1)
        else:
            ref_df = df_res
        ref_df_1 = ref_df[ref_df['L2_pred'].isna()]
        ref_df_2 = ref_df[ref_df['L3_pred'].isna()]
        ref_df_3 = ref_df[(ref_df['L4_pred'].isna()) & (~ref_df['L3_pred'].isna())]
        ref_df_4 = ref_df[ref_df['L5_pred'].isna() & (~ref_df['L4_pred'].isna()) & (~ref_df['L3_pred'].isna())]
        ref_df_5 = ref_df[ref_df['L6_pred'].isna() & (~ref_df['L5_pred'].isna()) & (~ref_df['L4_pred'].isna()) & (~ref_df['L3_pred'].isna())]
        ref_df_6 = ref_df[~ref_df['L6_pred'].isna()]

        for n, (i, o) in enumerate(iterate.iterrows()):
            if mindex: print(f"{n}/{n_objects}")
            o_df_1 = int(o['object_id'])
            c_L1_L2_L3 = 0
            c_L1_L2 = 0
            c_L1 = 0
            if debug:
                print(f"Orig object: {o_df_1}")
            if mindex:
                s = time.time()
                #df_orig, 36126726, pivots_df, [], existing_regions_unique, labels=mindex.get_descriptive_col_names()+ ["object_id"], max_visited_models=1000, bucket_level=8, is_profi=False, debug=False
                search_res = approximate_search_mindex(df_orig, o_df_1, struct_df, L1_only_pivots= [], existing_regions=[], existing_regions_dict=existing_regions_dict, max_visited_models=stop_cond_model, bucket_level=8, labels=self.get_descriptive_col_names()+["object_id"], is_profi=False, debug=debug)
                #print(search_res)
                times.append(time.time() - s)
            else:
                s = time.time()
                if stop_cond_model is not None:
                    #%time li.approximate_search_8(stack, df_res, 81178044, mapping, steps_limit=5, existing_buckets=existing_bucket_models, end_on_exact_hit=False, debug=True)
                    search_res = self.approximate_search_6(model_stack, df_res, o_df_1, mappings, existing_buckets=existing_buckets, over_approx_dict=over_approx_dict, steps_limit=stop_cond_model, custom_classifier=custom_classifier, end_on_exact_hit=False, debug=debug)
                    times.append(time.time() - s)
                if search_res['is_hit']:
                    c_L1_L2_L3 += 1
                #print(f"n of leaf nodes: {search_res['leaf_nodes_hit']}")
            if debug:
                print(f"\nTrying to hit {search_res['popped_nodes']}")
            popped_keys = [p_n for p_n in search_res['popped_nodes']]
            gts = []; gts_L1 = []; gts_L2 = []; gts_L3 = []; gts_L4 = []; gts_L5 = []; gts_L6 = []
            n_last_level = 0
            for p in popped_keys:
                model_label = p.split('.')
                if len(model_label) == 3:
                    gts_L1.append(int(model_label[(-1)]))
                if len(model_label) == 4:
                    if int(model_label[(-2)]) in gts_L1:
                        #if model_label[0] == "C":
                        #    gts.append((int(model_label[(-2)])))
                        #else:
                        gts_L2.append((int(model_label[(-2)]), int(model_label[(-1)])))
                if len(model_label) == 5:
                    if (int(model_label[(-3)]), int(model_label[(-2)])) in gts_L2:
                        #if model_label[0] == "C":
                        #    gts.append((int(model_label[(-3)]), int(model_label[(-2)])))
                        #else:
                        gts_L3.append((int(model_label[(-3)]), int(model_label[(-2)]), int(model_label[(-1)])))
                if len(model_label) == 6:
                    if (int(model_label[(-4)]), int(model_label[(-3)]), int(model_label[(-2)])) in gts_L3:
                        #if model_label[0] == "C":
                        #    gts.append((int(model_label[(-4)]), int(model_label[(-3)]), int(model_label[(-2)])))
                        #lse:
                        gts_L4.append((int(model_label[(-4)]), int(model_label[(-3)]), int(model_label[(-2)]), int(model_label[(-1)])))
                if len(model_label) == 7:
                    if (int(model_label[(-5)]), int(model_label[(-4)]), int(model_label[(-3)]), int(model_label[(-2)])) in gts_L4:
                        #if model_label[0] == "C":
                        #    gts.append((int(model_label[(-5)]), int(model_label[(-4)]), int(model_label[(-3)]), int(model_label[(-2)])))
                        #else:
                        gts_L5.append((int(model_label[(-5)]), int(model_label[(-4)]), int(model_label[(-3)]), int(model_label[(-2)]), int(model_label[(-1)])))
                if len(model_label) == 8:
                    if (int(model_label[(-6)]), int(model_label[(-5)]), int(model_label[(-4)]), int(model_label[(-3)]), int(model_label[(-2)])) in gts_L5:
                        #if model_label[0] == "C":
                        #    gts.append((int(model_label[(-6)]), int(model_label[(-5)]), int(model_label[(-4)]), int(model_label[(-3)]), int(model_label[(-2)])))
                        #else:
                        gts_L6.append((int(model_label[(-6)]), int(model_label[(-5)]), int(model_label[(-4)]), int(model_label[(-3)]), int(model_label[(-2)]), int(model_label[(-1)])))
                if len(model_label) == 9 and model_label[0] == "C":
                    gts.append((int(model_label[(-7)]), int(model_label[(-6)]), int(model_label[(-5)]), int(model_label[(-4)]), int(model_label[(-3)]), int(model_label[(-2)]), int(model_label[(-1)])))
                
            all_gts.append(gts)
            for g in [gts_L1, gts_L2, gts_L3, gts_L4, gts_L5, gts_L6]:
                if g != []: gts.extend(g)
            if debug:
                print(f"\nIdentified pool of ids to hit: {gts}")# {gts_L1} {gts_L2} {gts_L3}, {gts_L4} {gts_L5} {gts_L6} {gts_L7}")
            df_subset = []
            s = time.time()
            for gt in gts:
                if type(gt) is int:
                    df_subset.append(ref_df_1[(ref_df_1['L1_pred'] == gt)])# & (ref_df['L2'].isna())])
                elif len(gt) == 2:
                    df_subset.append(ref_df_2[(ref_df_2['L1_pred'] == gt[0]) & (ref_df_2['L2_pred'] == gt[1])])
                elif len(gt) == 3:
                    df_subset.append(ref_df_3[(ref_df_3['L1_pred'] == gt[0]) & (ref_df_3['L2_pred'] == gt[1]) & (ref_df_3['L3_pred'] == gt[2])])
                elif len(gt) == 4:
                    df_subset.append(ref_df_4[(ref_df_4['L1_pred'] == gt[0]) & (ref_df_4['L2_pred'] == gt[1]) & (ref_df_4['L3_pred'] == gt[2]) & (ref_df_4['L4_pred'] == gt[3])])
                elif len(gt) == 5:
                    df_subset.append(ref_df_5[(ref_df_5['L1_pred'] == gt[0]) & (ref_df_5['L2_pred'] == gt[1]) & (ref_df_5['L3_pred'] == gt[2]) & (ref_df_5['L4_pred'] == gt[3]) & (ref_df_5['L5_pred'] == gt[4])])
                elif len(gt) == 6:
                    df_subset.append(ref_df_6[(ref_df_6['L1_pred'] == gt[0]) & (ref_df_6['L2_pred'] == gt[1]) & (ref_df_6['L3_pred'] == gt[2]) & (ref_df_6['L4_pred'] == gt[3]) & (ref_df_6['L5_pred'] == gt[4]) & (ref_df_6['L6_pred'] == gt[5])])
           #print(time.time() - s)
            if df_subset != []:
                df_subset = pd.concat(df_subset).drop_duplicates()
                #return df_subset
                #return search_res, df_subset
                if debug:
                    print(df_subset.shape)
                if n_candidates is not None:
                    df_subset = df_subset[:n_candidates]
                    if debug:
                        print(df_subset)
                nn_object_ids = np.array((list(knns[str(o_df_1)].keys())), dtype=(np.int64))
                if df_subset.shape[0] != 0:
                    intersect = np.intersect1d(df_subset['object_id'].values, nn_object_ids)
                    if debug: print(intersect.shape[0] / knn)
                    intersects.append(intersect)
                    stats.append((o_df_1, intersect.shape[0] / knn, gts))
                else:
                    stats.append((o_df_1, 0, gts))
            else:
                stats.append((o_df_1, 0, None))

        return stats, np.array(times).mean()


    def get_1k_out_of_dataset_cophir(self, path="./datasets/cophir-queries-out-of-1M.data", scale=True):
        labels, numerical, attr_lengths = parse_objects(path, is_filter=False)
        df_orig = pd.DataFrame(numerical)
        df_orig['object_id'] = labels
        if scale:
            df = scale_per_descriptor(df_orig, [], attr_lengths)
        else:
            df = df_orig
        for l in self.labels:
            df[l] = np.nan
        return df

    def get_1k_out_of_dataset_profi(self, objects_path="./datasets/profiset-queries-out-of-1M-odd.data"):
        df_odd = pd.read_csv(objects_path, header=None)
        arr_full = [np.fromstring(arr, dtype=float, sep=" ") for arr in df_odd[0].values]
        df = pd.DataFrame(arr_full)
        obj_ids = pd.read_csv(objects_path.replace("odd", "even"),  sep=r'[\s+]', engine='python', header=None)
        df["object_id"] = obj_ids[2].values
        #df_orig = pd.DataFrame(numerical)
        #df_orig['object_id'] = labels
        #if scale:
        #    df = scale_per_descriptor(df_orig, [], attr_lengths)
        #else:
        #    df = df_orig
        for l in self.labels:
            df[l] = np.nan
        return df

    def load_knn_gt_8(self, path="./mindex/mindex_knn_gt_1M_200_parsed.json"):
        gt_knns = None
        with open(path) as json_file:
            gt_knns = json.load(json_file)
        return gt_knns

    def get_knn_objects(self, path="./queries.data", should_be_int=True):
        knn_object_ids = []
        with open(path) as f:
            for line in f.readlines():
                #z_1 = re.findall(r"AbstractObjectKey ([.*])", line)
                z_1 = re.findall(r"AbstractObjectKey ([\d\-_]+)", line)
                #print(z_1)
                if z_1:
                    if should_be_int:
                        knn_object_ids.append(int(z_1[0]))
                    else:
                        knn_object_ids.append(z_1[0])
        if should_be_int:
            return np.array(knn_object_ids, dtype=np.int64)
        else:
            return np.array(knn_object_ids)

    def create_overlaps_approx_dict(self, class_comb_dict, df_res_l1_miss, L1_label="L1", L2_label="L2"):
        approx_dict = {}
        L1_pred_label = f"{L1_label}_pred"
        for n,g in df_res_l1_miss.groupby([f"{L1_label}_pred"]):
            for i, obj in g.iterrows():
                group_c = class_comb_dict[(int(obj[f"{L1_label}_pred"]), int(obj[f"{L2_label}_pred"]),int(obj[f"{L1_label}"]))]
                approx_dict[f"M.1.{int(obj[L1_pred_label])}.{group_c}"] = f"M.1.{int(obj[L1_label])}.{int(obj[L2_label])}"
        return approx_dict
    """
    def create_overlaps_approx_dict_L23(self, class_comb_dict, df_res_l1_miss, L1_label="L1", L2_label="L2", L3_label="L3"):
        approx_dict = {}
        L1_pred_label = f"{L1_label}_pred"; L2_pred_label = f"{L2_label}_pred"
        for n,g in df_res_l1_miss.groupby([f"{L1_label}_pred", f"{L2_label}_pred"]):
            for i, obj in g.iterrows():
                group_c = class_comb_dict[(int(obj[f"{L2_label}_pred"]), int(obj[f"{L3_label}_pred"]), int(obj[f"{L2_label}"]))]
                approx_dict[f"M.1.{int(obj[L1_pred_label])}.{int(obj[L1_pred_label])}.{group_c}"] = f"M.1.{int(obj[L1_pred_label])}.{int(obj[L2_label])}.{int(obj[L3_label])}"
        return approx_dict

    def correct_training_labels_L23(self, df_res, df_res_l1_miss, class_comb_dict, L1_label="L1", L2_label="L2", L3_label="L3"):
        mismatched_objs = df_res_l1_miss[["object_id", f"{L2_label}_pred", f"{L3_label}_pred", f"{L2_label}"]].values
        split_dfs = [df_res[(df_res[f"{L1_label}_pred"] == i) & (df_res[f"{L2_label}_pred"] == j)] for i in range(df_res[f"{L1_label}_pred"].max()+1) for j in range(df_res[f"{L2_label}_pred"].max()+1)]
        mean = df_res["L2"].value_counts().values.mean()
        for m_o in mismatched_objs:
            r = split_dfs[int(m_o[1])][int(m_o[2])].loc[split_dfs[int(m_o[1])][int(m_o[2])]["object_id"] == int(m_o[0])].copy() #= class_comb_dict[(int(m_o[1]), int(m_o[2]))]
            r[L3_label] = class_comb_dict[(int(m_o[1]), int(m_o[2]), int(m_o[3]))]
            r_all = np.array([r.values[0],]*int(mean))
            r_all_df = pd.DataFrame(r_all, columns=df_res.columns)
            split_dfs[int(m_o[1])][int(m_o[2])] = pd.concat([split_dfs[int(m_o[1])][int(m_o[2])], r_all_df])
        return split_dfs
    """
    def correct_training_labels(self, df_res, df_res_l1_miss, class_comb_dict, L1_label="L1", L2_label="L2"):
        mismatched_objs = df_res_l1_miss[["object_id", f"{L1_label}_pred", f"{L2_label}_pred", f"{L1_label}"]].values
        split_dfs = [df_res[df_res[f"{L1_label}_pred"] == i] for i in range(df_res[f"{L1_label}_pred"].max()+1)]
        mean = df_res["L1"].value_counts().values.mean()
        for m_o in mismatched_objs:
            r = split_dfs[int(m_o[1])].loc[split_dfs[int(m_o[1])]["object_id"] == int(m_o[0])].copy() #= class_comb_dict[(int(m_o[1]), int(m_o[2]))]
            r[L2_label] = class_comb_dict[(int(m_o[1]), int(m_o[2]), int(m_o[3]))]
            r_all = np.array([r.values[0],]*int(mean))
            r_all_df = pd.DataFrame(r_all, columns=df_res.columns)
            split_dfs[int(m_o[1])] = pd.concat([split_dfs[int(m_o[1])], r_all_df])
        return split_dfs

    def create_splits(self, split_dfs):
        split_data = []
        for i, df_ in enumerate(split_dfs):
            X = df_.drop(self.get_descriptive_col_names_pred() + ["object_id"], axis=1, errors='ignore')
            assert X.shape[1] == 282
            y = df_[["L1", "L2", "object_id"]].values
            split_data.append({'X': X, 'y_2': y[:,1], 'y_1': y[:,0], 'object_id': y[:,2]})
        return split_data

    def create_overlaps_L12(self, df_res, stack_l1, L1_label="L1", L2_label="L2"):
        df_res_l1_miss = df_res[df_res[f"{L1_label}_pred"] != df_res[f"{L1_label}"]]
        
        class_comb_dict = {}
        counters = [(int(self.L2_range[1]) + 1) for i in range(int(self.L1_range[1]) + 1)]
        for class_comb in np.unique(np.array(list(df_res_l1_miss.groupby([f"{L1_label}_pred", f"{L2_label}_pred", f"{L1_label}"]).groups.keys())), axis=0):
            class_comb_dict[(int(class_comb[0]), int(class_comb[1]), int(class_comb[2]))] = counters[int(class_comb[0])]
            counters[int(class_comb[0])] += 1
        approx_dict = self.create_overlaps_approx_dict(class_comb_dict, df_res_l1_miss, L1_label, L2_label)
        #return class_comb_dict, approx_dict
        split_dfs_corrected = self.correct_training_labels(df_res, df_res_l1_miss, class_comb_dict, L1_label, L2_label)
        splits = self.create_splits(split_dfs_corrected)
        
        models_to_retrain = np.unique(np.array(list(class_comb_dict.keys()))[:, 0], axis=0)
        print(f"Models to retrain: {models_to_retrain}")
        for i in models_to_retrain:
            print(f"Training model {i} ")
            stack_l1[i-1].fit(splits[i]['X'], splits[i]["y_2"])    
        
        return approx_dict, splits, stack_l1

    def create_overlaps_approx_dict_L23(self, class_comb_dict, df_res_l1_miss, L1_label="L1", L2_label="L2", L3_label="L3"):
        approx_dict = {}
        L2_pred_label = f"{L2_label}_pred"
        for n,g in df_res_l1_miss.groupby([f"{L2_label}_pred"]):
            for i, obj in g.iterrows():
                group_c = class_comb_dict[(int(obj[f"{L2_label}_pred"]), int(obj[f"{L3_label}_pred"]), int(obj[f"{L2_label}"]))]
                L1_pred = f"{L1_label}_pred"
                approx_dict[f"M.1.{int(obj[L1_pred])}.{int(obj[L2_pred_label])}.{group_c}"] = f"M.1.{int(obj[L1_pred])}.{int(obj[L2_label])}.{int(obj[L3_label])}"
        return approx_dict

    def create_splits_L23(self, split_dfs):
        split_data = []
        for i, df_ in enumerate(split_dfs):
            split_data.append([])
            for j, df__ in enumerate(df_):
                X = df__.drop(self.get_descriptive_col_names_pred() + ["object_id"], axis=1, errors='ignore')
                assert X.shape[1] == 282
                y = df__[["L1", "L2", "L3", "object_id"]].values
                split_data[i].append({'X': X, 'y_3': y[:,2], 'y_2': y[:,1], 'y_1': y[:,0], 'object_id': y[:,-1]})
        return split_data

    def correct_training_labels_L23(self, df_res, df_res_l1_miss, class_comb_dict, n_objects_in_new_classes, L1_label="L1", L2_label="L2", L3_label="L3"):
        capacity_dict = get_avg_capacities(df_res, L2_label, L3_label)
        mismatched_objs = df_res_l1_miss[["object_id", f"{L2_label}_pred", f"{L3_label}_pred", f"{L2_label}", f"{L1_label}_pred"]].values
        #split_dfs = [df_res[(df_res[f"{L1_label}_pred"] == i) & (df_res[f"{L2_label}_pred"] == j)] for i in range(df_res[f"{L1_label}_pred"].max()+1) for j in range(df_res[f"{L2_label}_pred"].max()+1)]
        split_dfs = []
        for i in range(df_res[f"{L1_label}_pred"].max()+1):
            split_dfs.append([])
            for j in range(df_res[f"{L2_label}_pred"].max()+1):
                split_dfs[i].append(df_res[(df_res[f"{L1_label}_pred"] == i) & (df_res[f"{L2_label}_pred"] == j)])
        for m_o in mismatched_objs:
            #print(f"Cap for {m_o[1]} == {capacity_dict[m_o[1]]-1}")
            #r = split_dfs[m_o[1]].loc[split_dfs[m_o[1]]["object_id"] == m_o[0]].copy()
            #r[L2_label] = class_comb_dict[(m_o[1], m_o[2], m_o[3])]
            #print(f"Class for {m_o[1]}{m_o[2]} == {class_comb_dict[(m_o[1], m_o[2])]}")
            #for i in range(int((capacity_dict[m_o[1]])/n_objects_in_new_classes[(m_o[1], m_o[2], m_o[3])])):
            #    split_dfs[m_o[1]] = split_dfs[m_o[1]].append(r)
            r = split_dfs[m_o[-1]][m_o[1]].loc[split_dfs[m_o[-1]][m_o[1]]["object_id"] == m_o[0]].copy()
            r[L3_label] = class_comb_dict[(m_o[1], m_o[2], m_o[3])]
            r[L2_label] = m_o[1]
            r_all = np.array([r.values[0],]*int((capacity_dict[m_o[1]])/n_objects_in_new_classes[(m_o[1], m_o[2], m_o[3])]))
            r_all_df = pd.DataFrame(r_all, columns=df_res.columns)
            print(m_o, int((capacity_dict[m_o[1]])/n_objects_in_new_classes[(m_o[1], m_o[2], m_o[3])]))
            split_dfs[m_o[-1]][m_o[1]] = pd.concat([split_dfs[m_o[-1]][m_o[1]], r_all_df])
        return split_dfs

    def create_overlaps_L23(self, df_res, stack_l2, L1_label="L1", L2_label="L2", L3_label="L3"):
        df_res_l2_miss = df_res[df_res[f"{L2_label}_pred"] != df_res[f"{L2_label}"]]
        
        class_comb_dict = {}; n_objects_in_new_classes = {}
        counters = [(df_res[L3_label].max() + 1) for i in range(df_res[L2_label].max() + 1)]
        for class_comb in np.unique(np.array(list(df_res_l2_miss.groupby([f"{L2_label}_pred", f"{L3_label}_pred", f"{L2_label}"]).groups.keys())), axis=0):
            class_comb_dict[(class_comb[0], class_comb[1], class_comb[2])] = counters[class_comb[0]]
            n_objects_in_new_classes[(class_comb[0], class_comb[1], class_comb[2])] = df_res_l2_miss[(df_res_l2_miss[f"{L2_label}"] == class_comb[2]) & (df_res_l2_miss[f"{L2_label}_pred"] == class_comb[0]) & (df_res_l2_miss[f"{L3_label}_pred"] == class_comb[1])].shape[0]
            counters[class_comb[0]] += 1
        approx_dict = self.create_overlaps_approx_dict_L23(class_comb_dict, df_res_l2_miss, L1_label, L2_label)
        #return approx_dict
        split_dfs_corrected = self.correct_training_labels_L23(df_res, df_res_l2_miss, class_comb_dict, n_objects_in_new_classes, L1_label, L2_label)
        splits = self.create_splits_L23(split_dfs_corrected)
        models_retrain = []
        for class_comb in np.unique(np.array(list(df_res_l2_miss.groupby([f"{L1_label}_pred", f"{L2_label}_pred"]).groups.keys())), axis=0):
            models_retrain.append((class_comb[0], class_comb[1]))
        models_to_retrain = np.unique(np.array(models_retrain), axis=0)
        for (i,j) in models_to_retrain:
            print(f"Training model {i},{j}")
            stack_l2[i][j].fit(splits[i][j]['X'], splits[i][j]["y_3"])    
        
        return approx_dict, stack_l2, splits
        
    def correct_training_labels_L23_leaf(self, df_res, df_res_l2_miss, class_comb_dict, L1_label="L1", L2_label="L2"):
        mismatched_objs = df_res_l2_miss[["object_id", f"{L1_label}_pred", f"{L2_label}_pred", L1_label]].values
        df_res_c = df_res.copy()
        mean = df_res_l2_miss[ f"{L2_label}_pred"].value_counts().values.mean()
        for m_o in mismatched_objs:

            #r = df_res_c.loc[df_res_c["object_id"] == m_o[0]] #= class_comb_dict[(m_o[1], m_o[2], m_o[2])]
            r = df_res_c.loc[df_res_c["object_id"] == m_o[0]].copy()
            r[L2_label] == class_comb_dict[(m_o[1], m_o[2], m_o[3])]
            r_all = np.array([r.values[0],]*int(mean))
            #print(r_all)
            r_all_df = pd.DataFrame(r_all, columns=df_res.columns)
            df_res_c = pd.concat([df_res_c, r_all_df])
            #= class_comb_dict[(m_o[1], m_o[1])]
        return df_res_c

    def correct_training_labels_L12_leaf(self, df_res, df_res_l1_miss, class_comb_dict, L1_label="L1"):
        mismatched_objs = df_res_l1_miss[["object_id", f"{L1_label}_pred"]].values
        mean = df_res["L1"].value_counts().values.mean()
        df_res_c = df_res.copy()
        for m_o in mismatched_objs:
            r = df_res_c.loc[df_res_c["object_id"] == m_o[0]].copy()
            r[L1_label] = class_comb_dict[(m_o[1], m_o[1])]
            r_all = np.array([r.values[0],]*int(mean))
            #print(r_all)
            r_all_df = pd.DataFrame(r_all, columns=df_res.columns)
            df_res_c = pd.concat([df_res_c, r_all_df])
            #= class_comb_dict[(m_o[1], m_o[1])]
        return df_res_c

    def create_overlaps_approx_dict_L23_leaf(self, class_comb_dict, df_res_l1_miss, L1_label="L1", L2_label="L2"):
        approx_dict = {}
        L1_pred_label = f"{L1_label}_pred"; L2_pred_label = f"{L2_label}_pred"
        for n,g in df_res_l1_miss.groupby([f"{L1_label}_pred"]):
            for i, obj in g.iterrows():
                group_c = class_comb_dict[(int(obj[f"{L1_label}_pred"]), int(obj[f"{L2_label}_pred"]), int(obj[f"{L1_label}"]))]
                approx_dict[f"C.1.{int(obj[L1_pred_label])}.{int(obj[L2_pred_label])}.{int(group_c)}"] = f"C.1.{int(obj[L1_label])}.{int(obj[L2_label])}.{int(obj[L2_label])}"
        return approx_dict

    def create_overlaps_approx_dict_L12_leaf(self, class_comb_dict, df_res_l1_miss, L1_label="L1"):
        approx_dict = {}
        L1_pred_label = f"{L1_label}_pred"
        for n,g in df_res_l1_miss.groupby([f"{L1_label}_pred"]):
            for i, obj in g.iterrows():
                print(int(obj[f"{L1_label}_pred"]), int(obj[f"{L1_label}"]))
                group_c = class_comb_dict[(int(obj[f"{L1_label}_pred"]), int(obj[f"{L1_label}_pred"]))]
                approx_dict[f"C.1.{int(obj[L1_pred_label])}.{group_c}"] = f"C.1.{int(obj[L1_label])}.{int(obj[L1_label])}"
        # manual hack
        approx_dict['C.1.116.122'] = 'C.1.114.114'
        return approx_dict

    def create_splits_groupby(self, df_res_c, L1_label="L1", L2_label="L2"):
        split_data = []; split_data_mapping = {}
        L1_pred_label = f"{L1_label}_pred"; L2_pred_label = f"{L2_label}_pred"
        split_dfs = df_res_c.groupby([L1_pred_label, L2_pred_label])
        for i, g in enumerate(split_dfs.groups):
            X = df_res_c[(df_res_c[L1_pred_label] == g[0]) & (df_res_c[L2_pred_label] == g[1])].drop(get_descriptive_col_names_with_predictions(), axis=1, errors='ignore')
            assert X.shape[1] == 282
            y = df_res_c[(df_res_c[L1_pred_label] == g[0]) & (df_res_c[L2_pred_label] == g[1])][get_descriptive_col_names()].values
            split_data.append({'X': X, 'y_2': y[:,1], 'y_1': y[:,0], 'object_id': y[:,2]})
            split_data_mapping[g] = i
        return split_data, split_data_mapping

    def create_splits_groupby_l12_leaf(self, df_res_c, L1_label="L1"):
        split_data = []; split_data_mapping = {}
        L1_pred_label = f"{L1_label}_pred"
        split_dfs = df_res_c.groupby([L1_pred_label])
        for i, g in enumerate(split_dfs.groups):
            X = df_res_c[(df_res_c[L1_pred_label] == g) ].drop(self.get_descriptive_col_names_pred() + ["object_id"], axis=1, errors='ignore')
            assert X.shape[1] == 282
            y = df_res_c[(df_res_c[L1_pred_label] == g) ][["L1", "object_id"]].values
            split_data.append({'X': X, 'y_1': y[:,0], 'object_id': y[:,-1]})
            split_data_mapping[g] = i
        return split_data, split_data_mapping


    def create_splits_groupby_l23_leaf(self, df_res_c, L1_label="L1", L2_label="L2"):
        split_data = []; split_data_mapping = {}
        L1_pred_label = f"{L1_label}_pred"; L2_pred_label = f"{L2_label}_pred"
        split_dfs = df_res_c.groupby([L1_pred_label,L2_pred_label])
        for i, g in enumerate(split_dfs.groups):
            print(g)
            X = df_res_c[(df_res_c[L1_pred_label] == g[0]) &(df_res_c[L2_pred_label] == g[1])].drop(self.get_descriptive_col_names_pred() + ["object_id"], axis=1, errors='ignore')
            assert X.shape[1] == 282
            y = df_res_c[(df_res_c[L1_pred_label] == g[0]) &(df_res_c[L2_pred_label] == g[1])][["L1", "L2", "object_id"]].values
            split_data.append({'X': X, 'y_2': y[:,1], 'y_1': y[:,0], 'object_id': y[:,-1]})
            split_data_mapping[g] = i
        return split_data, split_data_mapping


    def create_overlaps_L12_leaf(self, df_res, stack_l1, L1_label="L1"):
        df_res_l1_miss = df_res[df_res[f"{L1_label}_pred"] != df_res[f"{L1_label}"]]
        
        class_comb_dict = {}
        counters = [(df_res[f"{L1_label}_pred"].max() + 1) for i in range(df_res[f"{L1_label}_pred"].max() + 1)]
        for class_comb in np.unique(np.array(list(df_res_l1_miss.groupby([f"{L1_label}_pred"]).groups.keys())), axis=0):
            # class_comb[0], class_comb[1], class_comb[1]) -> repeating the same class, cause it's a leaf
            class_comb_dict[(class_comb, class_comb)] = counters[class_comb]
            counters[class_comb] += 1
        approx_dict = self.create_overlaps_approx_dict_L12_leaf(class_comb_dict, df_res_l1_miss, L1_label)
        print(approx_dict)
        df_res_corrected = self.correct_training_labels_L12_leaf(df_res, df_res_l1_miss, class_comb_dict, L1_label)
        splits, split_data_mapping = self.create_splits_groupby_l12_leaf(df_res_corrected)
        models_to_retrain = np.unique(np.array(list(class_comb_dict.keys()))[:, 0], axis=0)
        print(models_to_retrain)
        for i in models_to_retrain:
            print(f"Training model {i} using stack[6][0][{i}]")
            stack_l1[i].fit(splits[split_data_mapping[i]]['X'], splits[split_data_mapping[i]]["y_1"])    
        
        return approx_dict, stack_l1, splits, split_data_mapping
        

    def create_overlaps_L23_leaf(self, df_res, stack_l2, L1_label="L1", L2_label="L2"):
        df_res_l2_miss = df_res[df_res[f"{L2_label}_pred"] != df_res[f"{L2_label}"]]
        
        class_comb_dict = {}; n_objects_in_new_classes = {}
        counters = [(df_res[L2_label].max() + 1) for i in range(df_res[L1_label].max() + 1)]
        for class_comb in np.unique(np.array(list(df_res_l2_miss.groupby([f"{L1_label}_pred", f"{L2_label}_pred", f"{L1_label}"]).groups.keys())), axis=0):
            print(class_comb)
            # class_comb[0], class_comb[1], class_comb[1]) -> repeating the same class, cause it's a leaf
            class_comb[0] = int(class_comb[0]); class_comb[1] = int(class_comb[1]); class_comb[2] = int(class_comb[2])
            class_comb_dict[(int(class_comb[0]), int(class_comb[1]), int(class_comb[2]))] = counters[int(class_comb[0])]
            n_objects_in_new_classes[(int(class_comb[0]), int(class_comb[1]), int(class_comb[1]))] = df_res_l2_miss[(df_res_l2_miss[f"{L1_label}_pred"] == class_comb[0]) & (df_res_l2_miss[f"{L2_label}_pred"] == class_comb[1]) & (df_res_l2_miss[f"{L1_label}"] == class_comb[2])].shape[0]
            counters[int(class_comb[0])] += 1
        approx_dict = self.create_overlaps_approx_dict_L23_leaf(class_comb_dict, df_res_l2_miss, L1_label, L2_label)
        #return approx_dict
        df_res_corrected = self.correct_training_labels_L23_leaf(df_res, df_res_l2_miss, class_comb_dict, L1_label, L2_label)
        splits, split_data_mapping = self.create_splits_groupby_l23_leaf(df_res_corrected)
        models_to_retrain = np.unique(np.array(list(class_comb_dict.keys()))[:, :2], axis=0)
        for i in models_to_retrain:
            print(f"Training model {i[0]}, {i[1]} using stack[2][{i[0]}][{i[0]}]")
            stack_l2[i[0]][i[1]].fit(splits[split_data_mapping[(i[0], i[1])]]['X'], splits[split_data_mapping[(i[0], i[1])]]["y_2"])    
        
        return approx_dict, stack_l2, splits

    def find_unique_nn_combinations(self, obj, obj_gts, nn_gts, gt_2nns):
        knn_classes = []; # knn_classes_L2 = []
        for o, gts in zip(obj,obj_gts):
            nn_gts_ = nn_gts[gt_2nns[o]]
            if not (gts[0] == nn_gts_[0] and gts[1] == nn_gts_[1]):
                knn_classes.append((gts[0], nn_gts_[0], nn_gts_[1]))
                #knn_classes_L2.append((gts[0], gts[1], nn_gts_[0], nn_gts_[1]))
        return np.unique(np.array(knn_classes), axis=0)#, np.unique(np.array(knn_classes_L2), axis=0)
