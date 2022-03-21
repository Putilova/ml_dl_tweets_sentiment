import requests
import pickle
from google.cloud import storage
import tensorflow as tf
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
        # access variables files
        blob_weights1 = bucket.blob('models/variables.index')
        blob_weights2 = bucket.blob('models/variables.data-00000-of-00001')
    except Exception as err:
        print(f"Cannot access variables files. Error message: {err}")
        logging.error(err)
        raise
    print("loaded models from bucket")
    
    try:
        # access model file
        blob_tfidf = bucket.blob('models/tfidfmodel.pickle')
    except Exception as err:
        print(f"Cannot access model file. Error message: {err}")
        logging.error(err)
        raise
    print("loaded tfidf model")
     
    try:
        # download variable files to temporary directory
        blob_weights1.download_to_filename('/tmp/variables.index')
        blob_weights2.download_to_filename('/tmp/variables.data-00000-of-00001')
    except Exception as err:
        print(f"Cannot download variable files to temporary directory. Error message: {err}")
        logging.error(err)
        raise
    print("downloaded model weights  ")
    
    try: ###########
        # generate layers
        serverless_model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(500, activation='relu'),
            tf.keras.layers.Dense(500, activation='relu'),
            tf.keras.layers.Dense(2, activation='softmax')
            ])
    except Exception as err:
        print(f"Cannot generate layers. Error message: {err}")
        logging.error(err)
        raise
    
    try:
        # loading weights
        serverless_model.load_weights('/tmp/variables')
    except Exception as err:
        print(f"Cannot load wieghts. Error message: {err}")
        logging.error(err)
        raise
    print("Loaded weights  ")

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
        output = serverless_model.predict(numeric_text)[:,1]
    except Exception as err:
        print(f"Cannot make prediction. Error message: {err}")
        logging.error(err)
        raise
        
    try:
        # print prediction output
        print("Printing prediction")
        print(output)
        sentiment="unknown"
        if output[0] > 0.5 :
            print("positive prediction")
            sentiment="postive"
        else:
            print("negative prediction")
            sentiment="negative"
        print("Printing sentiment")     
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
