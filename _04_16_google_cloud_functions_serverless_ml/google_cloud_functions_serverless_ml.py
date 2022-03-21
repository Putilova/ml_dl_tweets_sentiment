import requests
import pickle
from google.cloud import storage
import numpy as np
import logging

def hello_world(request):
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """
    try:
        # convert json to dictionary
        request_json = request.get_json()
    except Exception as err:
        print(f"Cannot convert json to dictionary. Error message: {err}")
        logging.error(err)
        raise

    try:
        # extract features values from request
        age = request_json['age']
        salary = request_json['salary']
    except Exception as err:
        print(f"Cannot extract features values from request. Error message: {err}")
        logging.error(err)
        raise
        
    try:
        # initiate google storage client
        storage_client = storage.Client()
    except Exception as err:
        print(f"Cannot initiate google storage client. Error message: {err}")
        logging.error(err)
        raise
        
    try:
        # access bucket
        bucket = storage_client.get_bucket('ml_dl_course_bucket')
    except Exception as err:
        print(f"Cannot access bucket. Error message: {err}")
        logging.error(err)
        raise
        
    try:
        # access mclassifier and scaler files
        blob_classifier = bucket.blob('models/classifier.pickle')
        blob_scaler = bucket.blob('models/sc.pickle')
    except Exception as err:
        print(f"Cannot access mclassifier and scaler files. Error message: {err}")
        logging.error(err)
        raise
        
    try:
        # download files to temporary directory
        blob_classifier.download_to_filename('/tmp/classifier.pickle')
        blob_scaler.download_to_filename('/tmp/sc.pickle')
    except Exception as err:
        print(f"Cannot download files to temporary directory. Error message: {err}")
        logging.error(err)
        raise
        
    try:
        # unpickle classifier and scaler
        serverless_classifier = pickle.load(open('/tmp/classifier.pickle','rb'))
        serverless_scaler = pickle.load(open('/tmp/sc.pickle','rb'))
    except Exception as err:
        print(f"Cannot unpickle classifier and scaler. Error message: {err}")
        logging.error(err)
        raise
        
    try:
        # make prediction
        pred_proba = serverless_classifier.predict_proba(serverless_scaler.transform(np.array([[age,salary]])))[:,1]
    except Exception as err:
        print(f"Cannot make prediction. Error message: {err}")
        logging.error(err)
        raise
        
    try:
        # print prediction output
        pred_proba=0.2
        age=40
        salary=4000
        print(f'Prediction: {pred_proba}')
        print(f'Age: {age}, salary: {salary}')
    except Exception as err:
        print(f"Cannot print prediction output. Error message: {err}")
        logging.error(err)
        raise
        
    return "The prediction is {}".format(pred_proba)
    
    # request_json = request.get_json()
    # storage_client = storage.Client()
    # bucket = storage_client.get_bucket('ml_dl_course_bucket')
    # blob_classifier = bucket.blob('models/classifier.pickle')
    # blob_scaler = bucket.blob('models/sc.pickle')
    # blob_classifier.download_to_filename('/tmp/classifier.pickle')
    # blob_scaler.download_to_filename('/tmp/sc.pickle')
    # serverless_classifier = pickle.load(open('/tmp/classifier.pickle','rb'))
    # serverless_scaler = pickle.load(open('/tmp/sc.pickle','rb'))
    # age = request_json['age']
    # salary = request_json['salary']
    # pred_proba = serverless_classifier.predict_proba(serverless_scaler.transform(np.array([[age,salary]])))[:,1]
    # print(pred_proba)
    # print(age)
    # print(salary)
    # return "The prediction is {}".format(pred_proba)
    
# requests==2.24.0
# scikit-learn==1.0.2
# google-cloud-storage==1.25.0
# numpy==1.19.2

# {
#     "age":40,
#     "salary":20000
# }
