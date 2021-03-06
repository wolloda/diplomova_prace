import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import train_test_split

from imports import logging

from pathlib import Path
from timeit import default_timer as timer

from utils import get_knn_objects, get_sample_1k_objects
from enums import DatasetDirs, DEFAULT_DESCRIPTORS, KNN_OBJECTS

# BEGIN NEW IMPORTS
from mindex_new import MIndex
from profiset import *
# END

from LMI import LMI
from knn_search import evaluate_knn_per_query

def rounded_accuracy(y_true, y_pred):
    import tensorflow as tf
    from tensorflow import keras

    return keras.metrics.binary_accuracy(tf.round(y_true), tf.round(y_pred))

class Benchmark:
    def __init__(self, model, dataset = DatasetDirs.COPHIR_1M, descriptors = None, buckets = [], normalize = True):
        self.model = model
        self.dataset = dataset

        if not descriptors:
            descriptors = DEFAULT_DESCRIPTORS[dataset]

        self.li = LMI(PATH = f"{dataset.value}", desc_values = descriptors)
        self.df = self.li.get_dataset(normalize = normalize)

        #self.li, self.df = self.prepare_dataset(dataset, descriptors)
        logging.info(f"Dataset {dataset.value} loaded")

        self.buckets = buckets if buckets else [self.df["L1"].max(), self.df["L2"].max()]
        self.destination = self.output_destination(dataset)

        if (descriptors != DEFAULT_DESCRIPTORS[dataset]):
            self.destination += f"/ENCODED-{descriptors}"

        Path(self.destination).mkdir(parents = True, exist_ok = True)

        self.logger = self.get_logger()
        self.logger.info("Benchmark initialized")

        if (descriptors != DEFAULT_DESCRIPTORS[dataset]):
            self.logger.debug("Number of output columns differs from number of columns in dataset -> training an encoder")
            self.df = self.get_encoded_df(descriptors)

        self.logger.info(f"Buckets: {self.buckets}")

    def prepare_dataset(self, dataset, descriptors, normalize = True):
        DATA_LOCATION = "/storage/brno6/home/tslaninakova/learned-indexes"

        if dataset in [DatasetDirs.COPHIR_100k, DatasetDirs.COPHIR_1M]:
            li = LMI(dataset.value, desc_value = descriptors)
            df = self.li.get_dataset(normalize = normalize)

        elif dataset == DatasetDirs.PROFI_1M:
            li = MIndex(PATH = f"{DATA_LOCATION}/", mindex = "MIndex1M-Profiset-leaf2000/")
            li.labels = ["L1", "L2"]
            index_df = load_indexes_profiset(f"{DATA_LOCATION}/MIndex1M-Profiset-leaf2000/", li.labels,  filenames=[f'level-{l}.txt' for l in range(1,3)])
            index_df = index_df.sort_values(by=["object_id"])
            objects = get_1M_profiset(index_path=f"{DATA_LOCATION}/MIndex1M-Profiset-leaf2000/", objects_path="f{DATA_LOCATION}/datasets/descriptors-decaf-odd-5M-1.data", labels=li.labels)

            df = pd.DataFrame(objects)
            df["L1"] = index_df["L1"].values
            df["L2"] = index_df["L2"].values
            df["object_id"] = index_df["object_id"].values

        elif dataset == DatasetDirs.PROFI_100k:
            li = MIndex(PATH = f"{DATA_LOCATION}/", mindex = "MIndex1M-Profiset-leaf200/")
            li.labels = ["L1", "L2"]
            index_df = load_indexes_profiset(f"{DATA_LOCATION}/MIndex1M-Profiset-leaf200/", li.labels,  filenames=[f'level-{l}.txt' for l in range(1,3)])
            index_df = index_df.sort_values(by=["object_id"])
            #TODO
            objects = get_1M_profiset(index_path=f"{DATA_LOCATION}/MIndex1M-Profiset-leaf200/", objects_path="f{DATA_LOCATION}/datasets/descriptors-decaf-odd-5M-1.data", labels=li.labels)

        return (li, df)

    def get_logger(self):
        logger = logging.getLogger('benchmark')
        shared_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', "%Y-%m-%d %H:%M:%S")

        file_logger = logging.FileHandler(f"{self.destination}/benchmark.log")
        file_logger.setLevel(logging.INFO)
        file_logger.setFormatter(shared_formatter)

        stream_logger = logging.StreamHandler()
        stream_logger.setLevel(logging.DEBUG)
        stream_logger.setFormatter(shared_formatter)

        logger.addHandler(file_logger)
        logger.addHandler(stream_logger)
        return logger

    def get_encoded_df(self, output_columns):
        import tensorflow as tf
        from tensorflow import keras

        start = timer()

        # AUTOENCODER TRAINING
        cols = len(self.df.drop(["L1", "L2", "object_id"], axis = 1).columns)

        X_train_full, X_test, y_train_full, y_test = train_test_split(self.df.drop(["L1", "L2", "object_id"], axis = 1).values, self.df["L1"].values)
        X = np.concatenate((X_train_full, X_test))
        y = np.concatenate((y_train_full, y_test))

        stacked_encoder = keras.models.Sequential([
            keras.layers.Flatten(input_shape = (cols,)),
            keras.layers.Dense(cols // 3, activation = "selu"),
            keras.layers.Dense(output_columns, activation = "sigmoid"),
            keras.layers.ActivityRegularization(l1 = 1e-3)
            ])

        stacked_decoder = keras.models.Sequential([
            keras.layers.Dense(cols // 3, activation = "selu", input_shape = [output_columns]),
            keras.layers.Dense(cols * 1, activation="sigmoid"),
            keras.layers.Reshape((cols,))
            ])

        stacked_ae = keras.models.Sequential([stacked_encoder, stacked_decoder])
        stacked_ae.compile(loss = "binary_crossentropy",
                optimizer=keras.optimizers.SGD(lr = 1.5), metrics = [rounded_accuracy])

        history = stacked_ae.fit(X, X, epochs = 15)

        end = timer()
        self.logger.info(f"Autoencoder from {cols} -> {output_columns} trained on {self.dataset.value} in {end - start} seconds ({(end - start) / 60} minutes)")

        # "PREDICT" NEW VALUES (DIMENSION REDUCTION)
        encoded_array = stacked_encoder.predict(self.df.drop(["L1", "L2", "object_id"], axis = 1))
        df_encoded = pd.DataFrame(encoded_array)

        df_encoded["L1"] = self.df["L1"] # can do this because order of dataframe rows is preserved
        df_encoded["L2"] = self.df["L2"]
        df_encoded["object_id"] = self.df["object_id"]

        # reorder columns
        cols = df_encoded.columns.tolist()
        cols = cols[-3:] + cols[:-3]
        df_encoded = df_encoded[cols]

        return df_encoded

    def load_config(self, model):
        training_specs = []

        if model == "GMM":
            covariance_types = ['spherical', 'diag', 'tied', 'full']
            max_iter = [1, 2, 5]
            init_params = ['kmeans', 'random']

            for covariance_type in covariance_types:
                for iterations in max_iter:
                    for init_param in init_params:
                        tmp = []
                        for i in range(len(self.buckets)):
                            tmp.append({'comp': self.buckets[i], 'cov_type': covariance_type, 'max_iter': iterations, 'init_params': init_param})

                        training_specs.append({model: tmp})

        elif model == "BayesianGMM":
            covariance_types = ['spherical', 'tied', 'diag', 'full']
            max_iter = [1, 2]
            init_params = ['kmeans', 'random']
            weight_concentration_prior_types = ['dirichlet_process', 'dirichlet_distribution']

            for covariance_type in covariance_types:
                for iterations in max_iter:
                    for init_param in init_params:
                        for weight_concentration_prior_type in weight_concentration_prior_types:
                            tmp = []
                            for i in range(len(self.buckets)):
                                tmp.append({'comp': self.buckets[i], 'cov_type': covariance_type, 'max_iter': iterations, 'init_params': init_param, 'weight_concentration_prior_type': weight_concentration_prior_type})

                            training_specs.append({model: tmp})

        elif model == "KMeans":
            max_iter = [5, 10, 25]
            n_init = [1, 5, 10]
            
            for iterations in max_iter:
                for initializations in n_init:
                    tmp = []
                    for i in range(len(self.buckets)):
                        tmp.append({'n_clusters': self.buckets[i], 'n_init': initializations, 'max_iter': iterations})

                    training_specs.append({model: tmp})

        elif model == "Faiss":
            n_inits = [5, 10, 15, 20]
            log_reg_iterations = [5, 10, 15, 20]

            buckets = list(map(lambda x: int(x), self.buckets))

            for init in n_inits:
                for iteration in log_reg_iterations:
                    tmp = []
                    for i in range(len(buckets)):
                        tmp.append({'n_clusters': buckets[i], 'n_init': init, "LogReg": {"ep": iteration}})

                    training_specs.append({model: tmp})

        self.logger.info(f"Total specs to train: {len(training_specs)}")
        return training_specs

    def output_destination(self, dataset):
        output_destination = f"/storage/brno6/home/wollf/learned-indexes/learned-indexes/performance/{str(dataset)[12:]}/{self.model}/"
        output_destination += "-".join(map(lambda x: str(x), self.buckets))
        return output_destination

    def create_identifier(self, spec):
        identifier = ""
        spec_keys = list(spec.keys())
        if "comp" in spec_keys:
            spec_keys.remove("comp")

        for i in range(len(spec_keys)):
            identifier += f"{spec_keys[i]}={spec[spec_keys[i]]}"
            if i != len(spec_keys) - 1:
                identifier += "_"

        return identifier

    def mocap_prepare_knns(self, knns):
        new_knns = {}
        for k, v in knns.items():
            knn = {}
            for k1, v1 in v.items():
                knn[k1.replace("-", "_").replace("_", "")] = v1
            new_knns[k.replace("-", "_").replace("_", "")] = knn

        return new_knns


    def evaluate(self, model_specs = []):
        final_results = {}

        object_checkpoints = [500, 1000, 3000, 5000, 10000, 50000, 100000, 200000, 300000, 500000, 750000]
        knns = self.li.load_knns()

        if self.dataset == DatasetDirs.MOCAP:
            object_checkpoints = [500, 1000, 3000, 5000, 10000, 50000, 100000, 200000, 300000]
            #object_checkpoints = [500, 1000, 3000, 5000, 10000, 50000, 100000, 200000, 260000]
            knns = self.mocap_prepare_knns(knns)

        training_specs = model_specs if model_specs else self.load_config(self.model) 

        for training_spec in training_specs:
            for model_rep in range(5):
                self.logger.debug(training_spec)
                spec = training_spec[self.model][0]

                identifier = self.create_identifier(spec)
                identifier += str(model_rep)

                start = timer()
                try:
                    df_result = self.li.train(self.df, training_spec, should_erase = True)
                    queries = get_sample_1k_objects(df_result, KNN_OBJECTS[self.dataset])

                    if self.dataset == DatasetDirs.MOCAP:
                        df_result["object_id"] = df_result["object_id"].str.replace("-", "_").apply(int)

                except Exception as e:
                    self.logger.error(f"Training_spec {training_spec} exception: {str(e)}")
                    continue

                end = timer()
                self.logger.info(f"{self.model} with training spec: {training_spec} trained on {self.dataset.value} in {end - start} seconds ({(end - start) / 60} minutes)")

                checkpoint_count = len(object_checkpoints) 
                query_count = len(queries)

                model_times = [0] * checkpoint_count
                model_recall = [0] * checkpoint_count

                errors = []

                for i in range(query_count):
                    search_result = self.li.search(df_result, queries.iloc[i]["object_id"], stop_cond_objects = object_checkpoints, debug = False)

                    time_checkpoints = search_result['time_checkpoints']
                    try:
                        recall = evaluate_knn_per_query(search_result, df_result, knns)

                        for j in range(checkpoint_count):
                            model_times[j] += time_checkpoints[j]
                            model_recall[j] += recall[j]
                    except IndexError as e:
                        #self.logger.error(f"{self.model} `evaluate_knn_per_query` failed on query {queries[i]} with error {str(e)}")
                        errors.append(i)

                if errors:
                    self.logger.error(f"{self.model} `evaluate_knn_per_query` failed on {len(errors)} queries")

                model_times = list(map(lambda x: x / (query_count - len(errors)), model_times))
                model_recall = list(map(lambda x: x / (query_count - len(errors)), model_recall))

                final_results[identifier] = {"model_times": model_times, "model_recall": model_recall}

                with open(f"{self.destination}/{identifier}.csv", 'w') as f:
                    f.write("model_times,model_recall\n")
                    for i in range(checkpoint_count):
                        f.write("%f,%f\n"%(model_times[i], model_recall[i]))
