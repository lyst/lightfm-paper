import argparse
import json
import logging
import numpy as np
from pprint import pformat
import scipy.sparse as sp
import sys

from lightfm import LightFM

import experiments
from experiments.cf_model import CFModel
from experiments.lsiup_model import LsiUpModel
from experiments.modelutils import fit_model
from experiments.movielens.data import read_movie_features, read_interaction_data


logger = experiments.getLogger('experiments.movielens.model')


def read_data(titles, genres,
              genome_tag_threshold,
              positive_threshold):

    logger.debug('Reading features')
    features = read_movie_features(titles=titles, genres=genres, genome_tag_threshold=genome_tag_threshold)
    item_features_matrix = features.mat.tocoo().tocsr()

    logger.debug('Reading interactions')
    interactions = read_interaction_data(features.item_ids,
                                         positive_threshold=positive_threshold)
    interactions.fit(min_positives=1, sampled_negatives_ratio=0, use_observed_negatives=True)

    logger.debug('%s users, %s items, %s interactions, %s item features in the dataset',
                len(interactions.user_ids), len(features.item_ids),
                len(interactions.data), len(features.feature_ids))

    return features, item_features_matrix, interactions


def run(features,
        item_features_matrix,
        interactions,
        cf_model,
        lsiup_model,
        n_iter,
        test_size,
        cold_start,
        learning_rate,
        no_components,
        a_alpha,
        b_alpha,
        epochs):

    logger.debug('Fitting the model with %s', locals())

    no_interactions = len(interactions.data)

    if cf_model:
        logger.info('Fitting the CF model')
        modelfnc = lambda: CFModel(dim=no_components)
    elif lsiup_model:
        logger.info('Fitting the LSI-UP model')
        modelfnc = lambda: LsiUpModel(dim=no_components)
    else:
        modelfnc = lambda: LightFM(learning_rate=learning_rate,
                                    no_components=no_components,
                                    item_alpha=a_alpha,
                                    user_alpha=b_alpha)

    model, auc = fit_model(interactions=interactions,
                           item_features_matrix=item_features_matrix, 
                           n_iter=n_iter,
                           epochs=epochs,
                           modelfnc=modelfnc,
                           test_size=test_size,
                           cold_start=cold_start)
    logger.debug('Average AUC: %s', auc)

    if not cf_model and not lsiup_model:
        model.add_item_feature_dictionary(features.feature_ids, check=False)
        features.add_latent_representations(model.item_features)

        titles = ('Lord of the Rings: The Two Towers, The (2002)',
                  'Toy Story (1995)',
                  'Terminator, The (1984)',
                  'Europa Europa (Hitlerjunge Salomon) (1990)')

        for title in titles:
            logger.debug('Most similar movies to %s: %s', title,
                        features.most_similar_movie(title, number=20))

            # Can only get similar tags if we have tag features
        test_features = ('genome:art house',
                         'genome:dystopia',
                         'genome:bond')

        for test_feature in test_features:
            try:
                logger.debug('Features most similar to %s: %s',
                             test_feature,
                             model.most_similar(test_feature, 'item', number=10))
            except KeyError:
                pass

    return auc



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run the MovieLens experiment')
    parser.add_argument('-i', '--ids', action='store_true',
                      help='Use item ids as features.')
    parser.add_argument('-t', '--tags', action='store_true',
                        help='Use tags as features.')
    parser.add_argument('-s', '--split', action='store', required=True, type=float,
                        help='Fraction (eg, 0.2) of data to use as the test set')
    parser.add_argument('-c', '--cold', action='store_true',
                        help='Use the cold start split.')
    parser.add_argument('-l', '--lsi', action='store_true',
                        help='Use the LSI-LR model')
    parser.add_argument('-u', '--up', action='store_true',
                        help='Use the LSI-UP model')
    parser.add_argument('-d', '--dim', action='store',
                        type=int, default=(64,),
                        nargs='+',
                        help='Latent dimensionality of the model.')
    parser.add_argument('-n', '--niter', action='store',
                        type=int, default=5,
                        help='Number of train/test splits')
    args = parser.parse_args()

    logger.info('Running the MovieLens experiment.')
    logger.info('Configuration: %s', pformat(args))

    # A large tag threshold excludes all tags.
    tag_threshold = 0.8 if args.tags else 100.0
    features, item_features_matrix, interactions = read_data(titles=args.ids,
                                                             genres=False,
                                                             genome_tag_threshold=tag_threshold,
                                                             positive_threshold=4.0)

    results = {}
    
    for dim in args.dim:
        auc = run(features,
                  item_features_matrix,
                  interactions,
                  cf_model=args.lsi,
                  lsiup_model=args.up,
                  n_iter=args.niter,
                  test_size=args.split,
                  cold_start=args.cold,
                  learning_rate=0.05,
                  no_components=int(dim),
                  a_alpha=0.0,
                  b_alpha=0.0,
                  epochs=30)

        results[int(dim)] = auc
        logger.info('AUC %s for configuration %s', auc, pformat(args))

    sys.stdout.write(json.dumps(results))
