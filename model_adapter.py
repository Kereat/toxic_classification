import dill as pickle
import pandas as pd
import numpy as np
import keras
from keras.models import model_from_json
from scipy.sparse import hstack
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
import logging

logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s %(name)s %(funcName)s %(message)s')

class SklearnAdapter(object):
    def __init__(self, project_path):
        self.project_path = project_path
        logger.info('Project path: {}'.format(project_path))

        self.word_vectorizer = pickle.load(open(self.project_path + "word_vectorizer.pickle", "rb"))
        self.char_vectorizer = pickle.load(open(self.project_path + "char_vectorizer.pickle", "rb"))

        self.group_model = pickle.load(open(self.project_path + "logreg_group.pkl", "rb"))
        self.group_mapping = pd.read_pickle(self.project_path + "group_mapping.df")

        self.service_model = pickle.load(open(self.project_path + "logreg_service.pkl", "rb"))
        self.service_mapping = pd.read_pickle(self.project_path + "service_mapping.df")

        self.service_part_model = pickle.load(open(self.project_path + "logreg_service_part.pkl", "rb"))
        self.service_part_mapping = pd.read_pickle(self.project_path + "service_part_mapping.df")

        self.priority_model = pickle.load(open(self.project_path + "logreg_priority.pkl", "rb"))
        self.priority_mapping = pd.read_pickle(self.project_path + "priority_mapping.df")

        self.type_model = pickle.load(open(self.project_path + "logreg_type.pkl", "rb"))
        self.type_mapping = pd.read_pickle(self.project_path + "type_mapping.df")


    def get_redictions(self, preprocessed_text: str):
        word_features = self.word_vectorizer.transform([preprocessed_text])
        char_features = self.char_vectorizer.transform([preprocessed_text])
        features = hstack([word_features, char_features])

        group_id, group_uid_b64 = self.predict_group(features)
        service_id, service_uid_b64 =  self.predict_service(features)
        type_id, type_uid_b64 = self.predict_type(features)
        priority_id, priority_uid_b64 = self.predict_priority(features)

        features_2 = hstack([
            features,
            group_id,
            service_id,
            type_id,
            priority_id
        ])
        logger.info("Shape: {}".format(features_2.shape))

        service_part_id, service_part_uid_b64 = self.predict_service_part(features_2)

        result = {
            "group": group_uid_b64,
            "service": service_uid_b64,
            "type": type_uid_b64,
            "priority": priority_uid_b64,
            "service_part": service_part_uid_b64
        }
        logger.info("Result: {}".format(result))
        return result

    def predict_group(self, features):
        group_id = self.group_model.predict(features)
        logger.info("Group_id: {}".format(group_id))
        name, uid = self.get_group_date(group_id[0])
        logger.info("Group_name: {}".format(name))
        return group_id, uid

    def get_group_date(self, group_id):
        name = self.group_mapping[self.group_mapping['group_id'] == group_id]["group_name"].values[0]
        uid = self.group_mapping[self.group_mapping['group_id'] == group_id]["group_uid_b64"].values[0]
        return name, uid

    def predict_service(self, features):
        service_id = self.group_model.predict(features)
        logger.info("service_id: {}".format(service_id))
        name, uid = self.get_service_data(service_id[0])
        logger.info("service_name: {}".format(name))
        return service_id, uid

    def get_service_data(self, service_id):
        name = self.service_mapping[self.service_mapping['service_id'] == service_id]["service_name"].values[0]
        uid = self.service_mapping[self.service_mapping['service_id'] == service_id]["service_uid_b64"].values[0]
        return name, uid

    def predict_service_part(self, features):
        service_part_id = self.service_part_model.predict(features)
        logger.info("Service_part_id: {}".format(service_part_id))
        name, uid = self.get_service_part_data(service_part_id[0])
        logger.info("Service_part_name: {}".format(name))
        return service_part_id, uid

    def get_service_part_data(self, service_part_id):
        name = self.service_part_mapping[
            self.service_part_mapping['service_part_id'] == service_part_id]["service_part_name"].values[0]
        uid = self.service_part_mapping[
            self.service_part_mapping['service_part_id'] == service_part_id]["service_part_uid_b64"].values[0]
        return name, uid


    def predict_priority(self, features):
        service_part_id = self.priority_model.predict(features)
        logger.info("priority_id: {}".format(service_part_id))
        name, uid = self.get_priority_data(service_part_id[0])
        logger.info("priority_name: {}".format(name))
        return service_part_id, uid

    def get_priority_data(self, priority_id):
        name = self.priority_mapping[self.priority_mapping['priority_id'] == priority_id]["priority_name"].values[0]
        uid = self.priority_mapping[self.priority_mapping['priority_id'] == priority_id]["priority_uid_b64"].values[0]
        return name, uid

    def predict_type(self, features):
        type_id = self.type_model.predict(features)
        logger.info("Type_id: {}".format(type_id))
        name, uid = self.get_type_data(type_id[0])
        logger.info("Type_name: {}".format(name))
        return type_id, uid

    def get_type_data(self, type_id):
        name = self.type_mapping[self.type_mapping['type_id'] == type_id]["type_name"].values[0]
        uid = self.type_mapping[self.type_mapping['type_id'] == type_id]["type_uid_b64"].values[0]
        return name, uid

class KerasAdapter():
    def __init__(self):
        self.model = model_from_json(open('models/classification_model.json').read())  # if json
        self.model.load_weights('models/classification_model_weights.h5')

        self.group_mapping = pd.read_pickle("mappings/group_mapping.pkl")

    def predict_group(self, features):
        probas = self.model.predict_proba([features])
        predictions_dict = dict(zip(self.group_mapping.group_name.values, probas[0]))
        return predictions_dict

    def get_predictions(self, features):
        response_dict = {
            "group_predictions": self.predict_group(features)
        }
        return response_dict

class FeatureExtractor():
    def __init__(self):
        self.word_vectorizer = joblib.load('models/word_vectorizer.pkl')
        self.char_vectorizer = joblib.load('models/char_vectorizer.pkl')

    def extract_features(self, raw_text: str):
        features = np.hstack([
            self.word_vectorizer.transform(raw_text),
            self.char_vectorizer.transform(raw_text),
        ])
        return features