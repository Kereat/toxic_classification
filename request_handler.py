import json
import jsonschema
from utils import LazyProperty
import logging

# Logging config
logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(funcName)s %(message)s')

class ClassifyRequestHandler(object):
    """
     * Request validation
     * JSON validation
    """
    schema = {
        "type": "object",
        "properties": {
            "subject": {"type": "string"},
            "description": {"type": "string"},
            "type": {"type": "string"},
            "priority": {"type": "string"}
        }
    }

    def __init__(self, req):
        self.request = req
        self.validate_json()
        self.log_request()

    @LazyProperty
    def json_dict(self):
        request_json = self.request.get_json()
        if isinstance(request_json, str):
            request_json = json.loads(request_json)
            return request_json
        return request_json

    def validate_json(self):
        assert self.json_dict is not None, "Json is empty"
        jsonschema.validate(self.json_dict, self.schema)

    def log_request(self):
        logger.debug("Request handler: {}".format(vars(self)))


class UpdateRequestHandler(object):
    """
     * Request validation
     * JSON validation
    """
    schema = {
        "type": "object",
        "properties": {
            "task_uid_b64": {"type": "string"},
            "group_uid_b64": {"type": "string"},
            "service_uid_b64": {"type": "string"},
            "service_part_uid_b64": {"type": "string"},
            "type_uid_b64": {"type": "string"},
            "priority_uid_b64": {"type": "string"}
        }
    }

    def __init__(self, req):
        self.request = req
        self.validate_json()
        self.log_request()
        self.json_dict["source_name"] = "avi"

    @LazyProperty
    def json_dict(self):
        request_json = self.request.get_json()
        if isinstance(request_json, str):
            request_json = json.loads(request_json)
            return request_json
        return request_json

    def validate_json(self):
        assert self.json_dict is not None, "Json is empty"
        jsonschema.validate(self.json_dict, self.schema)

    def log_request(self):
        logger.debug("Request handler: {}".format(vars(self)))

class RequestHandler(object):
    schema = {
        "type": "object",
        "properties": {
            "default": {"type": "string"}
        }
    }
    def __init__(self, req):
        self.request = req
        # JSON handling
        self.validate_json()
        self.log_request()

    @LazyProperty
    def json_dict(self):
        request_json = self.request.get_json()
        logger.info("Type of request.get_json(): {}".format(request_json))
        if isinstance(request_json, str):
            msg = "Json is str"
            logger.info(msg)
            request_json = json.loads(request_json)
            return request_json
        return request_json

    @LazyProperty
    def request_meta(self):
        # Request handling
        self.environ = self.request.environ
        self.headers = str(self.request.headers)

    def validate_json(self):
        assert self.json_dict is not None, "Json is empty"
        jsonschema.validate(self.json_dict, self.schema)

    def log_request(self):
        logger.debug("Parsed request attributes: {}".format(vars(self)))