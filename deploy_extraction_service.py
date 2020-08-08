from flask import Flask, request
import json
import os,sys
import sys
from src.extractor.entity_extractor import MatchUser
from conf.public import catalog, credentials


match_user = MatchUser(catalog, credentials)

app = Flask(__name__)

@app.route('/match')
def hello_world():
   return 'Hello World'

@app.route('/extract',methods = ['POST'])
def match_message():
   if request.method == "POST":

      input_json = request.get_json()

      match_user.initialize_text(input_json)

      users = match_user.get_matched_messages()

      return users


if __name__ == '__main__':
   app.run(host='0.0.0.0')

