import array
import collections
import numpy as np
import os
import re
import scipy.sparse as sp


class IncrementalCOOMatrix(object):

    def __init__(self, dtype):

        if dtype is np.int32:
            type_flag = 'i'
        elif dtype is np.int64:
            type_flag = 'l'
        elif dtype is np.float32:
            type_flag = 'f'
        elif dtype is np.float64:
            type_flag = 'd'
        else:
            raise Exception('Dtype not supported.')

        self.dtype = dtype
        self.shape = None

        self.rows = array.array('i')
        self.cols = array.array('i')
        self.data = array.array(type_flag)

    def append(self, i, j, v):

        self.rows.append(i)
        self.cols.append(j)
        self.data.append(v)

    def tocoo(self):

        rows = np.frombuffer(self.rows, dtype=np.int32)
        cols = np.frombuffer(self.cols, dtype=np.int32)
        data = np.frombuffer(self.data, dtype=self.dtype)

        self.shape = self.shape or (np.max(rows) + 1, np.max(cols) + 1)

        return sp.coo_matrix((data, (rows, cols)),
                             shape=self.shape)

    def __len__(self):

        return len(self.data)


class Features(object):

    def __init__(self):

        self.feature_ids = {}
        self.item_ids = {}
        self.title_mapping = {}

        self.mat = IncrementalCOOMatrix(np.int32)

    def add_item(self, item_id):

        iid = self.item_ids.setdefault(item_id, len(self.item_ids))
        
    def add_feature(self, item_id, feature):

        iid = self.item_ids.setdefault(item_id, len(self.item_ids))

        feature_id = self.feature_ids.setdefault(feature, len(self.feature_ids))

        self.mat.append(iid, feature_id, 1)

    def add_title(self, item_id, title):

        iid = self.item_ids.setdefault(item_id, len(self.item_ids))
        self.title_mapping[iid] = title

    def set_shape(self):

        self.mat.shape = len(self.item_ids), len(self.feature_ids)

    def add_latent_representations(self, latent_representations):

        dim = latent_representations.shape[1]
        lrepr = np.zeros((len(self.title_mapping), dim),
                         dtype=np.float32)

        for i, row in enumerate(self.mat.tocoo().tocsr()):
            lrepr[i] = np.sum(latent_representations[row.indices], axis=0)

        self.lrepr = lrepr
        self.inverse_title_mapping = {v: k for k, v in self.title_mapping.iteritems()}

    def most_similar_movie(self, title, number=5):

        iid = self.inverse_title_mapping[title]

        vector = self.lrepr[iid]

        dst = (np.dot(self.lrepr, vector)
               / np.linalg.norm(self.lrepr, axis=1) / np.linalg.norm(vector))
        movie_ids = np.argsort(-dst)
        
        return [(self.title_mapping[x], dst[x]) for x in movie_ids[:number]
                if x in self.title_mapping]


class Interactions(object):

    def __init__(self, item_ids):

        self.item_ids = item_ids
        self.user_ids = {}

        self.user_data = collections.defaultdict(lambda: {1: array.array('i'),
                                                          0: array.array('i')})

        self.iids_sample_pool = np.array(item_ids.values())

        self._user_id = array.array('i')
        self._item_id = array.array('i')
        self._data = array.array('i')

    def add(self, user_id, item_id, value):

        iid = self.item_ids[item_id]
        user_id = self.user_ids.setdefault(user_id, len(self.user_ids))

        self.user_data[user_id][value].append(iid)

    def fit(self, min_positives=1, sampled_negatives_ratio=0, use_observed_negatives=True):
        """
        Constructs the training data set from raw interaction data.

        Parameters:
        - min_positives: users with fewer than min_positives interactions are excluded
                         from the training set
        - sampled_negatives_ratio: a ratio of 3 means that at most three negative examples
                         randomly sampled for the pids_sample_pool will be included.
        """

        for user_id, user_data in self.user_data.iteritems():

            positives = user_data.get(1, [])
            raw_negatives = user_data.get(0, [])

            if len(positives) < min_positives:
                continue

            if use_observed_negatives:
                observed_negatives = list(set(raw_negatives) - set(positives))
            else:
                observed_negatives = []

            if sampled_negatives_ratio:
                sampled_negatives = np.random.choice(self.iids_sample_pool,
                                                     size=len(positives) * sampled_negatives_ratio)
                sampled_negatives = list(set(sampled_negatives) - set(positives))
            else:
                sampled_negatives = []

            for value, pids in zip((1, 0, 0), (positives, observed_negatives, sampled_negatives)):
                for pid in pids:
                    self._user_id.append(user_id)
                    self._item_id.append(pid)
                    self._data.append(value)

        self.user_id = np.frombuffer(self._user_id, dtype=np.int32)
        self.item_id = np.frombuffer(self._item_id, dtype=np.int32)
        self.data = np.frombuffer(self._data, dtype=np.int32)
