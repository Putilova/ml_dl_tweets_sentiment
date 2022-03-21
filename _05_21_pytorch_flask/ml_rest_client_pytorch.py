import json
import requests

url =  'http://34.125.181.30:8005/model'

request_data = json.dumps({'age':40,'salary':50000})
response = requests.post(url,request_data)
print (response.text)



