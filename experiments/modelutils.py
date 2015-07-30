"""
Utilities for fitting models.
"""

import array
import collections
import logging

import numpy as np
import scipy.sparse as sp
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import roc_auc_score

from experiments.cf_model import CFModel, evaluate_cf_model
from experiments.lsiup_model import LsiUpModel


logger = logging.getLogger(__name__)


class StratifiedSplit(object):
    """
    Class responsible for producing train-test splits.
    """

    def __init__(self, user_ids, item_ids, n_iter=10, 
                 test_size=0.2, cold_start=False, random_seed=None):
        """
        Options:
        - test_size: the fraction of the dataset to be used as the test set.
        - cold_start: if True, test_size of items will be randomly selected to
                      be in the test set and removed from the training set. When
                      False, test_size of all training pairs are moved to the
                      test set.
        """

        self.user_ids = user_ids
        self.item_ids = item_ids
        self.no_interactions = len(self.user_ids)
        self.n_iter = n_iter
        self.test_size = test_size
        self.cold_start = cold_start

        self.shuffle_split = ShuffleSplit(self.no_interactions,
                                          n_iter=self.n_iter,
                                          test_size=self.test_size)

    def _cold_start_iterations(self):
        """
        Performs the cold-start splits.
        """

        for _ in range(self.n_iter):
            unique_item_ids = np.unique(self.item_ids)
            no_in_test = int(self.test_size * len(unique_item_ids))

            item_ids_in_test = set(np.random.choice(unique_item_ids, size=no_in_test))

            test_indices = array.array('i')
            train_indices = array.array('i')

            for i, item_id in enumerate(self.item_ids):
                if item_id in item_ids_in_test:
                    test_indices.append(i)
                else:
                    train_indices.append(i)

            train = np.frombuffer(train_indices, dtype=np.int32)
            test = np.frombuffer(test_indices, dtype=np.int32)

            # Shuffle data.
            np.random.shuffle(train)
            np.random.shuffle(test)

            yield train, test

    def __iter__(self):

        if self.cold_start:
            splits = self._cold_start_iterations()           
        else:
            splits = self.shuffle_split

        for train, test in splits:

            # Make sure that all the users in test
            # are represented in train.
            user_ids_in_train = collections.defaultdict(lambda: 0)
            item_ids_in_train = collections.defaultdict(lambda: 0)

            for uid in self.user_ids[train]:
                user_ids_in_train[uid] += 1

            for iid in self.item_ids[train]:
                item_ids_in_train[iid] += 1

            if self.cold_start:
                test = [x for x in test if self.user_ids[x] in user_ids_in_train]
            else:
                # For the non-cold start scenario, make sure that both users
                # and items are represented in the train set.
                test = [x for x in test if (self.user_ids[x] in user_ids_in_train
                                            and self.item_ids[x] in item_ids_in_train)]

            test = np.array(test)

            yield train, test


def stratified_roc_auc_score(y, yhat, user_indices):
    """
    Compute ROC AUC for each user individually, then average.
    """

    aucs = []

    y_dict = collections.defaultdict(lambda: array.array('d'))
    yhat_dict = collections.defaultdict(lambda: array.array('d'))

    for i, uid in enumerate(user_indices):
        y_dict[uid].append(y[i])
        yhat_dict[uid].append(yhat[i])

    for uid in y_dict:

        user_y = np.frombuffer(y_dict[uid], dtype=np.float64)
        user_yhat = np.frombuffer(yhat_dict[uid], dtype=np.float64)

        if len(user_y) and len(user_yhat) and len(np.unique(user_y)) == 2:
            aucs.append(roc_auc_score(user_y, user_yhat))

    logger.debug('%s users in stratified ROC AUC evaluation.', len(aucs))
    
    return np.mean(aucs)


def build_user_feature_matrix(user_ids):

    n = len(user_ids)

    return sp.coo_matrix((np.ones(n, dtype=np.int32), (np.arange(n), user_ids))).tocsr()


def fit_model(interactions, item_features_matrix,
              n_iter, epochs, modelfnc, test_size,
              cold_start, user_features_matrix=None):
    """
    Fits the model provided by modelfnc.
    """

    kf = StratifiedSplit(interactions.user_id, interactions.item_id,
                         n_iter=n_iter, test_size=test_size, cold_start=cold_start)

    logger.debug('Interaction density across all data: %s',
                 (float(len(interactions.data)) / (len(interactions.user_ids)
                                                   * len(interactions.item_ids))))
    logger.debug('Training model')

    # Store ROC AUC scores for all iterations.
    aucs = []

    # Iterate over train-test splits.
    for i, (train, test) in enumerate(kf):

        logger.debug('Split no %s', i)
        logger.debug('%s examples in training set, %s in test set. Interaction density: %s',
                    len(train), len(test), float(len(train)) / (len(interactions.user_ids)
                                                                * len(interactions.item_ids)))

        # For every split, get a new model instance.
        model = modelfnc()

        if isinstance(model, CFModel):
            logger.debug('Evaluating a CF model')
            test_auc, train_auc = evaluate_cf_model(model,
                                                    item_features_matrix,
                                                    interactions.user_id[train],
                                                    interactions.item_id[train],
                                                    interactions.data[train],
                                                    interactions.user_id[test],
                                                    interactions.item_id[test],
                                                    interactions.data[test])
            logger.debug('CF model test AUC %s, train AUC %s', test_auc, train_auc)
            aucs.append(test_auc)

        elif isinstance(model, LsiUpModel):
            logger.debug('Evaluating a LSI-UP model')

            # Prepare data.
            y = interactions.data
            no_users = np.max(interactions.user_id) + 1
            no_items = item_features_matrix.shape[0]

            train_user_ids = interactions.user_id[train]
            train_item_ids = interactions.item_id[train]

            user_features = sp.coo_matrix((interactions.data[train],
                                           (train_user_ids, train_item_ids)),
                                           shape=(no_users, no_items)).tocsr()
            user_feature_matrix = user_features * item_features_matrix

            # Fit model.
            model.fit(user_feature_matrix, item_features_matrix)
            
            # For larger datasets use incremental prediction. Slower, but
            # fits in far less memory.
            if len(train) or len(test) > 200000:
                train_predictions = model.predict(interactions.user_id[train],
                                                  interactions.item_id[train],
                                                  incremental=True)
                test_predictions = model.predict(interactions.user_id[test],
                                                 interactions.item_id[test],
                                                 incremental=True)
            else:
                train_predictions = model.predict(interactions.user_id[train],
                                                  interactions.item_id[train])
                test_predictions = model.predict(interactions.user_id[test],
                                                 interactions.item_id[test])

            # Compute mean ROC AUC scores on both test and train data.
            train_auc = stratified_roc_auc_score(y[train],
                                                 train_predictions,
                                                 interactions.user_id[train])
            test_auc = stratified_roc_auc_score(y[test],
                                                test_predictions,
                                                interactions.user_id[test])

            logger.debug('Test AUC %s, train AUC %s', test_auc, train_auc)

            aucs.append(test_auc)

        else:
            # LightFM and MF models using the LightFM implementation.
            if user_features_matrix is not None:
                user_features = user_features_matrix
            else:
                user_features = build_user_feature_matrix(interactions.user_id)

            item_features = item_features_matrix

            previous_auc = 0.0

            interactions.data[interactions.data == 0] = -1

            train_interactions = sp.coo_matrix((interactions.data[train],
                                                (interactions.user_id[train],
                                                 interactions.item_id[train])))

            # Run for a maximum of epochs epochs.
            # Stop if the test score starts falling, take the best result.
            for x in range(epochs):
                model.fit_partial(train_interactions,
                                  item_features=item_features,
                                  user_features=user_features,
                                  epochs=1, num_threads=1)

                train_predictions = model.predict(interactions.user_id[train],
                                                  interactions.item_id[train],
                                                  user_features=user_features,
                                                  item_features=item_features,
                                                  num_threads=4)
                test_predictions = model.predict(interactions.user_id[test],
                                                 interactions.item_id[test],
                                                 user_features=user_features,
                                                 item_features=item_features,
                                                 num_threads=4)

                train_auc = stratified_roc_auc_score(interactions.data[train],
                                                     train_predictions,
                                                     interactions.user_id[train])
                test_auc = stratified_roc_auc_score(interactions.data[test],
                                                    test_predictions,
                                                    interactions.user_id[test])

                logger.debug('Epoch %s, test AUC %s, train AUC %s', x, test_auc, train_auc)

                if previous_auc > test_auc:
                    break

                previous_auc = test_auc

            aucs.append(previous_auc)

    return model, np.mean(aucs)
