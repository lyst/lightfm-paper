import codecs
from HTMLParser import HTMLParser
from lxml import etree
import os
import re

import experiments
from experiments.data import Features, Interactions

# Data description here: http://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede
logger = experiments.getLogger('experiments.stackexchange.model')

DATA_DIR = os.path.dirname(__file__)

QUESTION_POST_TYPE_ID = 1
QUESTION_POST_TYPE_ANSWER = 2


class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)


def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


def _process_post_tags(tags_string):

    return [x for x in tags_string.replace('<', ' ').replace('>', ' ').split(' ') if x]


def _process_post_body(body):

    body = re.sub('[^a-zA-Z]+', ' ', body.lower())

    return [x for x in body.split(' ') if len(x) > 2]


def read_tag_data():

    # XML, ugh
    with codecs.open(os.path.join(DATA_DIR, 'Tags.xml'), 'r', encoding='utf-8') as datafile:
        for i, line in enumerate(datafile):

            # Skip over all the invalid lines
            try:
                datum = dict(etree.fromstring(line).items())
                tag_id = datum['Id']
                tag_name = datum['TagName']
                tag_count = datum['Count']
            except etree.XMLSyntaxError:
                pass


def read_post_data():

    # XML, ugh
    with codecs.open(os.path.join(DATA_DIR, 'Posts.xml'), 'r', encoding='utf-8') as datafile:
        for i, line in enumerate(datafile):

            try:
                datum = dict(etree.fromstring(line).items())

                post_id = datum['Id']
                post_type = datum['PostTypeId']
                user_id = datum.get('OwnerUserId', None)
                parent_post_id = datum.get('ParentId', None)
                body = _process_post_body(datum['Body'])

                tags = _process_post_tags(datum.get('Tags', ''))

                if None in (post_id, user_id):
                    continue

                yield post_id, user_id, tags, parent_post_id, body

            except etree.XMLSyntaxError:
                pass


def read_user_data():

    with codecs.open(os.path.join(DATA_DIR, 'Users.xml'), 'r', encoding='utf-8') as datafile:
        for i, line in enumerate(datafile):

            try:
                datum = dict(etree.fromstring(line).items())

                user_id = datum['AccountId']
                about_me = datum.get('AboutMe', '')

                yield user_id, about_me

            except etree.XMLSyntaxError:
                pass


def read_post_features(tags, post_ids, post_text):

    features = Features()

    for post_id, user_id, post_tags, parent_post_id, body in read_post_data():
        # Only get features for questions, not answers.
        if parent_post_id is None:

            features.add_item(post_id)

            if post_ids:
                features.add_feature(post_id, 'post_id:' + post_id)

            if tags:
                for tag in post_tags:
                    features.add_feature(post_id, 'tag:' + tag)

            if post_text:
                for token in body:
                    features.add_feature(post_id, 'body:' + token)

    features.set_shape()

    return features


def read_user_features(about, user_ids, user_id_mapping):

    features = Features()
    features.item_ids = user_id_mapping

    if user_ids:
        for uid in user_id_mapping:
            features.add_feature(uid, 'user_id:' + uid)
        
    if about:
        # Add intercepts
        for uid in user_id_mapping:
            features.add_feature(uid, 'intercept')

        for i, (user_id, about_me) in enumerate(read_user_data()):

            clean_about = (strip_tags(about_me)
                           .replace('\n', ' ')
                           .lower())
            clean_about = _process_post_body(clean_about)

            for token in clean_about:
                features.add_feature(user_id, 'about:' + token)

    features.set_shape()
                       
    return features


def read_interactions(item_id_mapping):

    interactions = Interactions(item_id_mapping)

    interactions_dropped = 0

    for post_id, user_id, tags, parent_post_id, body in read_post_data():

        assert user_id is not None
        assert post_id is not None
        
        # Only answers count as interactions
        if parent_post_id is not None:
            try:
                interactions.add(user_id, parent_post_id, 1.0)
            except KeyError:
                interactions_dropped += 1

    logger.info('Dropped %s interactions due to None user issues', interactions_dropped)

    return interactions
