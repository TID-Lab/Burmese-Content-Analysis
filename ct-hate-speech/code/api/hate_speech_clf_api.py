import sys
import pickle
from flask import Flask, jsonify, abort, request
from flask_classful import FlaskView, route
import data_processing.mm_segmenter as mm
from data_processing.mm_converter import zawgyi_to_unicode
import re
import json
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

class HateSpeechClassifier(FlaskView):

    def __init__(self):
        self.model = "None"
        self.burmese_segmenter = mm.Segmenter()
        self.clf_mapping = {
            "RF" : "clfs/oversampled-RFClf.pkl"
        }
        with open("clfs/oversampled-tfidf-vect.pkl", 'rb') as vect_b:
            self.vectorizer = pickle.load(vect_b)
        with open("clfs/oversampled-RFClf.pkl", 'rb') as clf_b:
            self.model = pickle.load(clf_b)
        

    def preprocess(self):
        try:
            input_data = request.args.get('input_data', default = [])
            input_data = json.loads(input_data)
        except ValueError as e:
            abort(400)
        non_burmese_pattern = re.compile("[^"u"\U00001000-\U0000109F"u"\U00000020""]+", flags=re.UNICODE)
        preprocessed_data = []
        for text_document in input_data:
            try:
                preprocessed_text_document = zawgyi_to_unicode(text_document)
                preprocessed_text_document = " ".join(self.burmese_segmenter.segment(preprocessed_text_document))
                preprocessed_text_document = re.sub("\s+", ' ', non_burmese_pattern.sub('', preprocessed_text_document)) 
                preprocessed_data.append(preprocessed_text_document)
            except ValueError as e:
                abort(400)
        return jsonify(preprocessed_data) 

    def fit(self, X_train, y_train):
        raise NotImplementedError

    def predict(self):
        try:
            input_data = request.args.get('input_data', default = "")
        except ValueError as e:
            abort(400)
        input_data_vect = self.vectorizer.transform([input_data])
        predicted_values = map(lambda x: x[1],self.model.predict_proba(input_data_vect))[0]
        return jsonify({"Response" : {"Input" : input_data, "Predicted Hate %": predicted_values}})

    def batch_predict(self):
        try:
            input_data = request.args.get('input_data', default = [""])
            input_data = json.loads(input_data)
        except ValueError as e:
            abort(400)
        input_data_vect = self.vectorizer.transform(input_data)
        predicted_values = map(lambda x: x[1],self.model.predict_proba(input_data_vect))
        return jsonify({"Response" : list(zip(list(input_data), list(predicted_values)))})

    def dump_model(self, filename):
        joblib.dump(self.model, filename)

    def load_model(self):
        raise NotImplementedError

HateSpeechClassifier.register(app, route_base = '/')

if __name__ == '__main__':
    app.run(debug=True)

