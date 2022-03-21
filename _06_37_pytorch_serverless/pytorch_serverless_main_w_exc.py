import requests
import pickle
from google.cloud import storage

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

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
        text = request_json['sentence']
        print(f"printing the sentence: \n{text}")
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
        # access classifier file
        blob_dictionary = bucket.blob('models/text_classifier_pytorch_1')
    except Exception as err:
        print(f"Cannot access classifier model file. Error message: {err}")
        logging.error(err)
        raise
    print("loaded classifier file from bucket")
    
    try:
        # access model file
        blob_tfidf = bucket.blob('models/tfidfmodel.pickle')
    except Exception as err:
        print(f"Cannot access model file. Error message: {err}")
        logging.error(err)
        raise
    print("loaded tfidf model")
     
    try:
        # download classifier file to temporary directory
        blob_dictionary.download_to_filename('/tmp/text_classifier_pytorch_1')
    except Exception as err:
        print(f"Cannot download classifier file to temporary directory. Error message: {err}")
        logging.error(err)
        raise
    print("downloaded classifier file  ")
    
    try:
        # download model file to temporary directory
        blob_tfidf.download_to_filename('/tmp/tfidfmodel.pickle')
    except Exception as err:
        print(f"Cannot download model file to temporary directory. Error message: {err}")
        logging.error(err)
        raise
    print("downloaded tfidf")

    try:
        # loading model
        serverless_tfidf = pickle.load(open('/tmp/tfidfmodel.pickle','rb'))
    except Exception as err:
        print(f"Cannot load model. Error message: {err}")
        logging.error(err)
        raise
    print("Loaded tfidf")

    input_size=467
    output_size=2
    hidden_size=500
    
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = torch.nn.Linear(input_size, hidden_size)
            self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
            self.fc3 = torch.nn.Linear(hidden_size, output_size)
        def forward(self, X):
            X = torch.relu((self.fc1(X)))
            X = torch.relu((self.fc2(X)))
            X = self.fc3(X)

            return F.log_softmax(X,dim=1)

    model = Net()
    
    try:
        # load PyTorch dictionary
        model.load_state_dict(torch.load('/tmp/text_classifier_pytorch_1'))
        #model.load_state_dict(torch.jit.load('/tmp/text_classifier_pytorch_1'))
    except Exception as err:
        print(f"Cannot load PyTorch dictionary. Error message: {err}")
        logging.error(err)
        raise
    print("Loaded PyTorch Dictionary  ")
    
    try:
        # transform sentence to vector
        print(text)
        text_list=[]
        text_list.append(text)
        print(text_list)
        numeric_text = serverless_tfidf.transform(text_list).toarray()
    except Exception as err:
        print(f"Cannot transform sentence to vector. Error message: {err}")
        logging.error(err)
        raise

    try:
        # make prediction
        output = model(torch.from_numpy(numeric_text).float())
    except Exception as err:
        print(f"Cannot make prediction. Error message: {err}")
        logging.error(err)
        raise
        
    try:
        # print prediction output
        print("Printing prediction")
        print(output[:,0][0])
        print(output[:,1][0])    
        sentiment="unknown"
        
        sentiment="unknown"
        if torch.gt(output[:,0][0],output[:,1][0]):
            print("negative prediction")
            sentiment="negative from pytorch"
        else:
            print("positive prediction")
            sentiment="positive from pytorch"
        print("Printing prediction")     
        print(sentiment)
    except Exception as err:
        print(f"Cannot print prediction output. Error message: {err}")
        logging.error(err)
        raise
        
    return "The sentiment is {}".format(sentiment)

# # Function dependencies, for example:
# # package>=version
# tensorflow==2.8.0
# google-cloud-storage==1.16.1
# scikit-learn==1.0.2
# requests==2.24.0

# {
#     "sentence": "mediocre by India in the match"
# }

