import json
import requests

url = 'http://127.0.0.1:8005/model'
#url = 'http://34.125.132.249:8005/model'

request_data = json.dumps({'age':40,'salary':40000})
response = requests.post(url,request_data)
print (response.text)



