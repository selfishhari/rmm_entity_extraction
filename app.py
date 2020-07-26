from flask import Flask, request
import json
import os,sys
import sys
sys.path.append("/home/ec2-user/rmm_entity_extraction")

from src.extractor.entity_extractor import MatchUser
from conf.public import catalog, credentials


app = Flask(__name__)

@app.route('/match')
def hello_world():
   return 'Hello World'

@app.route('/extract',methods = ['POST'])
def match_user():
   if request.method == "POST":

      input_json = request.get_json()

      match_user = MatchUser(catalog, credentials, input_json)

      users = match_user.matched_users()

      return users


if __name__ == '__main__':
   app.run(host='0.0.0.0')
