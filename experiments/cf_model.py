"""
Fit and evaluate the LSI-LR model.
"""

import array
import collections
import itertools
import logging


import numpy as np
import scipy.sparse as sp
from sklearn.cross_validation import ShuffleSplit
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


logger = logging.getLogger(__name__)


class CFModel(object):
    """
    The LSI-LR model.
    """

    def __init__(self, dim=64):

        self.dim = dim
        self.model = None
        self.item_latent_features = None

    def fit_svd(self, mat):
        """
        Fit the feature latent factors.
        """

        model = TruncatedSVD(n_components=self.dim)
        model.fit(mat)

        self.model = model

    def fit_latent_features(self, feature_matrix):
        """
        Project items into the latent space.
        """

        self.item_latent_features = self.model.transform(feature_matrix)

    def fit_user(self, item_ids, y):
        """
        Fit a logistic regression model for a single user.
        """

        model = LogisticRegression()
        model.fit(self.item_latent_features[item_ids], y)

        return model

    def predict_user(self, model, item_ids):
        """
        Predict positive interaction probability for user represented by model.
        """

        return model.decision_function(self.item_latent_features[item_ids])



def evaluate_cf_model(model, feature_matrix, train_user_ids, train_item_ids, train_data,
                      test_user_ids, test_item_ids, test_data):
    """
    LSI-LR model: perform LSI (via truncated SVD on the item-feature matrix), then computer user models
    by fitting a logistic regression model to items represented as mixtures of LSI topics.
    """

    train_aucs = []
    test_aucs = []

    train_y_dict = collections.defaultdict(lambda: array.array('d'))
    train_iid_dict = collections.defaultdict(lambda: array.array('i'))

    test_y_dict = collections.defaultdict(lambda: array.array('d'))
    test_iid_dict = collections.defaultdict(lambda: array.array('i'))

    # Gather training data in user-sized chunks
    for i, (uid, iid, y) in enumerate(itertools.izip(train_user_ids, train_item_ids, train_data)):
        train_y_dict[uid].append(y)
        train_iid_dict[uid].append(iid)

    # Gather test data in user-sized chunks
    for i, (uid, iid, y) in enumerate(itertools.izip(test_user_ids, test_item_ids, test_data)):
        test_y_dict[uid].append(y)
        test_iid_dict[uid].append(iid)

    # Only use the items in the training set for LSI
    model.fit_svd(feature_matrix[np.unique(train_item_ids)])
    model.fit_latent_features(feature_matrix)

    # Fit models and generate predictions
    for uid in train_y_dict:
        train_iids = np.frombuffer(train_iid_dict[uid], dtype=np.int32)
        train_y = np.frombuffer(train_y_dict[uid], dtype=np.float64)

        test_iids = np.frombuffer(test_iid_dict[uid], dtype=np.int32)
        test_y = np.frombuffer(test_y_dict[uid], dtype=np.float64)

        if len(np.unique(test_y)) == 2 and len(np.unique(train_y)) == 2:
            user_model = model.fit_user(train_iids, train_y)
            train_yhat = model.predict_user(user_model, train_iids)
            test_yhat = model.predict_user(user_model, test_iids)
            
            train_aucs.append(roc_auc_score(train_y, train_yhat))
            test_aucs.append(roc_auc_score(test_y, test_yhat))

    return np.mean(test_aucs), np.mean(train_aucs)
