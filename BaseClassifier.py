from utils import label_encode_data
from random import shuffle

# Logging
import logging
logging.basicConfig(datefmt='%d-%m-%y %H:%M', format='%(asctime)-15s%(levelname)s: %(message)s', level=logging.INFO)

from sklearn.tree import DecisionTreeClassifier


class BaseClassifier(object):

    def __init__(self):
        pass

    def train(self, model, X, y, descriptor_values):
        """ Encodes the data for training, trains a scikit-learn classifier and collects predictions

        Parameters
        -------
        mode:
            Dictionary of model specification
        X: Numpy array
            Training values
        y: Numpy array
            Training labels

        Returns
        -------
        predictions: Numpy array
            Array of model predictions

        encoder: LabelEncoder
            Mapping of actual labels to labels used in training
        """

        assert X.shape[1] == descriptor_values
        #if not y:
        if y is None:
            return self.train_unsupervised(model, X)

        y, encoder = label_encode_data(y)
        try:
            model.fit(X, y)
            predictions = model.predict(X)
        except ValueError as e:
            model = None
            predictions = [0 for i in range(X.shape[0])]
        predictions = encoder.inverse_transform(predictions)

        return model, predictions, encoder

    def train_unsupervised(self, model, X):
        try:
            model.fit(X)
            predictions = model.predict(X)
        except ValueError as e:
            model = None
            predictions = [0 for i in range(X.shape[0])]
        return model, predictions, None

    def predict(self, prob_distr, encoder, value="value_l"):
        classes_votes = []
        for i,p in enumerate(prob_distr):
            if encoder and len(encoder.classes_) > i:
                classes_votes.append({value: int(encoder.classes_[i]), "votes_perc": p})
            else:
                classes_votes.append({value: i, "votes_perc": p})

        shuffle(classes_votes)
        classes_votes = sorted(classes_votes, key = lambda i: i['votes_perc'], reverse=True)
        return classes_votes

