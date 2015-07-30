import collections
import os
import re

from experiments.data import Features, Interactions


SEPARATOR = '::'
DATA_DIR = os.path.join(os.path.dirname(__file__), 'ml-10M100K')
GENOME_DIR = os.path.join(os.path.dirname(__file__), 'tag-genome')


def read_genome_tags(min_popularity=20):

    tag_dict = {}

    with open(os.path.join(GENOME_DIR, 'tags.dat'), 'r') as tagfile:
        for line in tagfile:

            tag_id, tag, popularity = line.split('\t')

            if int(popularity) >= min_popularity:
                tag_dict[int(tag_id)] = tag

    with open(os.path.join(GENOME_DIR, 'tag_relevance.dat'), 'r') as tagfile:
        for line in tagfile:

            iid, tag_id, relevance = line.split('\t')

            if int(tag_id) in tag_dict:
                yield iid, tag_dict[int(tag_id)], float(relevance)


def _process_raw_tag(tag):

    tag = re.sub('[^a-zA-Z]+', ' ', tag.lower()).strip()

    return tag


def read_tags():

    tag_dict = collections.defaultdict(lambda: 0)

    with open(os.path.join(DATA_DIR, 'tags.dat'), 'r') as tagfile:
        for line in tagfile:

            uid, iid, tag, timestamp = line.split(SEPARATOR)
            processed_tag = _process_raw_tag(tag)
            tag_dict[tag] += 1

    with open(os.path.join(DATA_DIR, 'tags.dat'), 'r') as tagfile:
        for line in tagfile:

            uid, iid, tag, timestamp = line.split(SEPARATOR)
            processed_tag = _process_raw_tag(tag)
            tag_count = tag_dict[processed_tag]

            yield iid, processed_tag, tag_count


def read_movie_features(titles=False, genres=False, genome_tag_threshold=1.0, tag_popularity_threshold=30):

    features = Features()

    with open(os.path.join(DATA_DIR, 'movies.dat'), 'r') as moviefile:
        for line in moviefile:
            (iid, title, genre_list) = line.split(SEPARATOR)
            genres_list = genre_list.split('|')

            features.add_item(iid)

            if genres:
                for genre in genres_list:
                    features.add_feature(iid, 'genre:' + genre.lower().replace('\n', ''))

            if titles:
                features.add_feature(iid, 'title:' + title.lower())

            features.add_title(iid, title)

    for iid, tag, relevance in read_genome_tags():
        # Do not include any tags for movies not in the 10M dataset
        if relevance >= genome_tag_threshold and iid in features.item_ids:
            features.add_feature(iid, 'genome:' + tag.lower())

    # Tags applied by users
    ## for iid, tag, count in read_tags():
    ##     if count >= tag_popularity_threshold and iid in features.item_ids:
    ##         features.add_feature(iid, 'tag:' + tag)

    features.set_shape()

    return features


def read_interaction_data(item_id_mapping, positive_threshold=4.0):

    interactions = Interactions(item_id_mapping)

    with open(os.path.join(DATA_DIR, 'ratings.dat'), 'r') as ratingfile:
        for line in ratingfile:

            (uid, iid, rating, timestamp) = line.split(SEPARATOR)

            value = 1.0 if float(rating) >= positive_threshold else 0.0

            interactions.add(uid, iid, value)

    return interactions
