# ml_dl_tweets_sentiment

This repo contains excercises from the "Machine Learning Deep Learning model deployment" course on Udemy: https://www.udemy.com/course/machine-learning-deep-learning-model-deployment/. 

Some skills from the course: 

Classification models with Scikit-learn.
Sentiment analysis with PyTorch and TensorFlow Keras. 
Flask applications.

Environment and tools:
Spyder / Windows Power Shell.
Google Colab, Google Drive, Google Cloud Platform (Compute Engine - VM instance, Cloud Storage, Cloud Function).
Ngrok account.
Twitter developer account.



## To start: Creating python environment

###  in Windows PS

Create virtual environment in user's folder:
```
C:\Users\admin> 

pip install virtualenv

virtualenv nlp_course

.\.nlp_course\Scripts\Activate.ps1

pip install scipy numpy pandas sklearn matplotlib pickle requests
```
Deactivate or delete:

```
deactivate

rm -r .nlp_course
```

### in Anaconda Powershell Prompt

```
conda create -n nlp_course

conda env list

conda activate nlp_course

conda install scipy numpy pandas sklearn matplotlib pickle requests
```

Deactivate or delete:

```
conda deactivate

env remove -n nlp_course
```

Change environment in Spyder:

Spyder > Preferences > Python interpreter > Use the following Python interpreter > C:\Users\admin\Anaconda3\envs\nlp_course\python.exe

## Section 2: Building, evaluating and saving a Model

### 6. Python NumPy Pandas Matplotlib crash course

Folder: _02_06_python_np_pd_plt

python_np_pd_plt.py

Input files: storepurchasedata.csv

### 7. Building and evaluating a Classification Model & 8. Saving the Model and the Scaler

Folder: _02_07_ml_classification

Notebook: ml_pipeline.ipynb

Input files: _02_06_python_np_pd_plt/storepurchasedata.csv

Output: classifier.pickle, sc.pickle

Folder: _02_07_ml_classification

ml_pipeline.py

Input files: _02_06_python_np_pd_plt\storepurchasedata.csv

Output: classifier.pickle, sc.pickle

## Section 3: Deploying the Model in other environments

### 9. Predicting locally with deserialized Pickle objects

Folder: _03_09_use_model

use_model.py

Input files: _02_07_ml_classification\classifier.pickle, _02_07_ml_classification\sc.pickle
    
### 10. Using the Model in Google Colab environment

Folder: _03_10_use_model_colab

use_model_colab.ipynb

Input files: _02_07_ml_classification/classifier.pickle, _02_07_ml_classification/sc.pickle

## Section 4: Creating a REST API for the Machine Learning Model

### 11. Flask REST API Hello World

Folder: _04_11_flask_hello_world

flask_hello_world.py

Run in Windows Powershell instance:

```
cd .\_04_11_flask_hello_world
py .\rest_client.py
```

### 12. Creating a REST API for the Model

Folder: _04_12_classifier_rest_service

classifier_rest_service.py

Input files: _02_07_ml_classification\classifier.pickle, _02_07_ml_classification\sc.pickle

Run in Windows Powershell instance:

```
cd .\_04_12_classifier_rest_service
py .\ml_rest_client.py
```

url = 'http://127.0.0.1:8005/model'

### 14. Hosting the Machine Learning REST API on the Cloud

Create VM instance:

e2-medium (2 vCPU, 4 GB memory)

Allow HTTP traffic

Allow HTTPS traffic

After creating check settings: More actions > View network details > FIREWALL RULES > default-allow-http > Protocols and ports > Allow all > SAVE

Folder: _04_14_classifier_rest_service_on_GCP

Instructions in _04_14_GCP_VM.txt

classifier_rest_service_on_GCP.py

Input files: https://github.com/Putilova/ml_dl_tweets_sentiment/raw/main/_02_07_ml_classification/classifier.pickle , 

https://github.com/Putilova/ml_dl_tweets_sentiment/raw/main/_02_07_ml_classification/sc.pickle

After running classifier_rest_service_on_GCP.py on CGP

Folder: _04_14_classifier_rest_service_on_GCP

ml_rest_client_from_GCP.py

url = 'http://{External_IP}:8005/model'

External_IP example: 34.125.173.28

### 16. Serverless Machine Learning API using Cloud Functions

Create cloud function (Python 3.7)

Folder: _04_16_google_cloud_functions_serverless_ml

google_cloud_functions_serverless_ml.py

upload to cloud storage bucket (ml_dl_course_bucket):

_02_07_ml_classification/classifier.pickle, 

_02_07_ml_classification/sc.pickle

Postman collection: ml_dl_course.postman_collection.json

request: use_sklearn_model_on_google_cloud_functions_serverless

### 17. Creating a REST API on Google Colab

Folder: _04_17_colab_rest_api

colab_rest_api_pyngrok.ipynb

Input files: _02_07_ml_classification/classifier.pickle, _02_07_ml_classification/sc.pickle 

Postman request: use_sklearn_model_on_google_colab_ipynb_w_pyngrok

## Section 5: Deploying Deep Learning Models

### 20. Building and deploying PyTorch models

Folder: _05_20_save_export_reload_pytorch_models

pytorch_create_save.ipynb

Input files: storepurchasedata_large.csv

Output: customer_buy.pt, customer_buy_state_dict, customer_buy_state_dict.zip

use_pytorch_dictionary.ipynb

Input files: customer_buy_state_dict, _02_07_ml_classification/sc.pickle

### 21. Creating a REST API for the PyTorch Model

Create VM instance:

e2-standard-16 (16 vCPU, 64 GB memory)

Folder: _05_21_pytorch_flask

Instructions in _05_21_GCP_VM.txt

pytorch_flask.py

Input files: https://github.com/Putilova/ml_dl_tweets_sentiment/raw/main/_02_07_ml_classification/sc.pickle

https://github.com/Putilova/ml_dl_tweets_sentiment/raw/main/_05_20_save_export_reload_pytorch_models/customer_buy_state_dict.zip

Folder: _05_21_pytorch_flask

ml_rest_client_pytorch.py

Postman request: use_pytorch_model_on_google_cloud

### 22. Saving & loading TensorFlow Keras models

Folder: _05_22_tf_serving_save_export

tf_customer_buy.ipynb

Input files: _05_20_save_export_reload_pytorch_models/storepurchasedata_large.csv

Output: customer_behavior_model/1/ , customermodel.zip

### 24. Creating a REST API using TensorFlow Model Server

Create VM instance:

e2-standard-8 (8 vCPU, 32 GB memory)

Operating system: Ubuntu

Version: Ubuntu 18.04 LTS

Size (GB): 100

Folder: _05_24_tf_model_serving

Instructions in _05_24_GCP_VM.txt

use_tf_model_serving.ipynb

Input files: _02_07_ml_classification/sc.pickle

### 25. Converting a PyTorch model to TensorFlow format using ONNX

Folder: _05_25_pytorch_create_save_onnx

pytorch_create_save_onnx.ipynb

Input files: _05_20_save_export_reload_pytorch_models/storepurchasedata_large.csv

Output: customer.onnx

## Section 6: Deploying NLP models for Twitter sentiment analysis

### 28. Creating and saving text classifier and tf-idf models

Folder: _06_28_text_classifier

text_classifier.ipynb

Input files: Restaurant_Reviews.tsv.txt

Output: textclassifier.pickle, tfidfmodel.pickle

### 30. Deploying tf-idf and text classifier models for Twitter sentiment analysis

Folder: _06_30_twitter_sentiment_analysis

twitter_sentiment_analysis.ipynb

Input files: _06_28_text_classifier/textclassifier.pickle,  _06_28_text_classifier/tfidfmodel.pickle
    
Output: twitter_sentiment_analysis_output_log.txt

### 31. Creating a text classifier using PyTorch

Folder: _06_31_text_classifier_pytorch

text_classifier_pytorch.ipynb

Input files: _06_28_text_classifier/Restaurant_Reviews.tsv.txt

Output: text_classifier_pytorch, tfidfmodel.pickle

### 32. Creating a REST API for the PyTorch NLP model

Folder: _06_32_pytorch_nlp_rest

pytorch_nlp_rest.ipynb
Input files: _06_31_text_classifier_pytorch/text_classifier_pytorch

Postman request: use_pytorch_nlp_model_on_google_colab_ipynb

### 33. Twitter sentiment analysis with PyTorch REST API

Folder: _06_33_twitter_pytorch_rest

Run app _06_32_pytorch_nlp_rest/pytorch_nlp_rest.ipynb

twitter_sentiment_analysis_rest.ipynb

### 34. Creating a text classifier using TensorFlow

Folder: _06_34_text_classifier_tensorflow

text_classifier_tensorflow.ipynb

Input files: _06_28_text_classifier/Restaurant_Reviews.tsv.txt

Output: text_classifier_model/1 , text_classifier_model.zip

### 35. Creating a REST API for TensforFlow models using Flask

Folder: _06_35_tf_nlp_flask_rest

tf_nlp_rest.ipynb

Input files: _06_34_text_classifier_tensorflow/text_classifier_model/1/ , _06_31_text_classifier_pytorch/tfidfmodel.pickle

Postman request: use_tensorflow_nlp_model_on_google_colab_ipynb

### 36. Serving TensorFlow models serverless

Create cloud function (Python 3.7)

Folder: _06_36_tf_serverless

tfserverless_main_w_exc.py

upload to cloud storage bucket (ml_dl_course_bucket):

_06_31_text_classifier_pytorch/tfidfmodel.pickle,

_06_34_text_classifier_tensorflow/text_classifier_model/1/variables/ . 

in variables: variables.index, variables.data-00000-of-00001

Postman request: use_tensorflow_nlp_model_on_google_cloud_functions_serverless

### 37. Serving PyTorch models serverless

Create cloud function (Python 3.7)

Folder: _06_37_pytorch_serverless

text_classifier_pytorch_101.ipynb

Input files: Restaurant_Reviews.tsv.txt

Output: text_classifier_pytorch_1, tfidfmodel.pickle

pytorch_serverless_main_w_exc.py

upload to cloud storage bucket (ml_dl_course_bucket):

_06_37_pytorch_serverlesstext_classifier_pytorch_1, _06_37_pytorch_serverless/tfidfmodel.pickle

Postman request: use_pytorch_nlp_model_on_google_cloud_functions_serverless

## Section 7: Deploying models on browser using JavaScript and TensorFlow.js

### 41. Adding TensforFlow.js to a web page

Folder: _07_41_tfjs1

### 42. Basic tensor operations using TensorFlow.js

Folder: _07_42_tfjs2

### 43. Deploying Keras model on a web page using TensorFlow.js

Folder: _07_43_tfjs4

tf3_keras_age_salary_tfjs.ipynb

Input files: _05_20_save_export_reload_pytorch_models/storepurchasedata_large.csv
Output: customer_behavior_model/1 , customermodel.zip, model.json

## Section 8: Model as a mathematical formula & Model as code

### 44. Deriving formula from a Linear Regression Model

Folder: _08_44_linear_regression

linear_regression.py

Input files: houseprice.csv

Output: Figure 1

### 45. Model as code

Folder: _08_45_model_as_code

another_python_app.py

Input files: house_price_predictor.py

house_price_predictor_2.py

## Section 9: Model in Database

### 46. Storing and retrieving models from a database using Colab, Postgres and psycopg2

Folder: _09_46_models_in_db



### 47. Creating a local model store with PostgreSQL

Folder: _09_47_local_model_store

Execute in database workbench:

create_model_table_local.sql

train_and_store_in_db.py

Input files: storepurchasedata.csv

use_models_from_db.py

## Section 10: MLOps and MLflow

### 50. Tracking Model training experiments with MLfLow

Folder: _10_50_ml_pipeline_mlfow



### 52. Running MLflow on Colab

Folder: _10_52_running_mlflow_on_colab



### 53. Tracking PyTorch experiments with MLflow

Folder: _10_53_mlflow_pytorch



### 54. Deploying Models with MLflow

Folder: _10_54_mlflow-deploy


