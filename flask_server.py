import logging
import logging.handlers
import sys
import base64
from flask import Flask, jsonify, request

import preprocessing_methods
import model_adapter
import request_handler


logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s %(funcName)s %(message)s')

app = Flask(__name__)

@app.before_request
def log_request():
    app.logger.debug(""" Request log: "
    headers: {}
    body: {}
    remote addr: {}
    method: {}
    scheme: {}
    full path: {}
    """.format(
        request.headers,
        request.get_data(),
        request.remote_addr,
        request.method,
        request.scheme,
        request.full_path
        )
    )

@app.after_request
def after_request(response):
    app.logger.debug("Response status: {}".format(response.status))
    return response

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    try:
        req = request_handler.ClassifyRequestHandler(request)
        subject = req.json_dict["subject"]
        description =  req.json_dict["description"]

        preprocessed_text = pp.apply_inference_pipeline(subject, description)
        features = feature_extractor.extract_features(preprocessed_text)
        response = keras_adapter.get_redictions(features)
        return jsonify(response), 200
    except BaseException as e:
        logger.error(str(e))
        return jsonify({"error": str(e)}), 500

"""
@app.route('/update', methods=['POST'])
def update():
    try:
        req = request_handler.ClassifyRequestHandler(request)
        query_interface.update_ticket(req.json_dict)
        return jsonify("Updated successfully"), 201
    except BaseException as e:
        logger.error(str(e))
        return jsonify({"error": str(e)}), 500
"""

if __name__ == "__main__":
    # Model interface
    pp = preprocessing_methods.PreprocessingInterface()
    feature_extractor = model_adapter.FeatureExtractor()
    keras_adapter = model_adapter.KerasAdapter()
    # query_interface = db_interface.QueryInterface()
    app.run(host='192.168.112.1', port=5012, debug=False, use_reloader=False)
