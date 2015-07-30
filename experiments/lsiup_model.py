"""
Fit the LSI-UP model.
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
from sklearn.preprocessing import normalize


logger = logging.getLogger(__name__)


class LsiUpModel(object):
    """
    The LSI-UP model.
    """

    def __init__(self, dim=64):

        self.dim = dim
        self.user_factors = None
        self.item_factors = None

    def fit(self, user_feature_matrix, product_feature_matrix):
        """
        Fit latent factors to the user-feature matrix through truncated SVD,
        then get item representations by projecting onto the latent feature
        space.
        """

        nrm = lambda x: normalize(x.astype(np.float64), norm='l2', axis=1)

        svd = TruncatedSVD(n_components=self.dim)
        svd.fit(nrm(user_feature_matrix))

        self.user_factors = svd.transform(nrm(user_feature_matrix))
        self.item_factors = svd.transform(nrm(product_feature_matrix))

    def predict(self, user_ids, product_ids, incremental=False):
        """
        Predict scores.
        """

        if not incremental:
            return np.inner(self.user_factors[user_ids],
                            self.item_factors[product_ids])
        else:
            result = array.array('f')
            
            for i in range(len(user_ids)):
                uid = user_ids[i]
                pid = product_ids[i]

                result.append(np.dot(self.user_factors[uid],
                                     self.item_factors[pid]))

            return np.frombuffer(result, dtype=np.float32)
