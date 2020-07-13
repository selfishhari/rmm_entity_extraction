from flask import Flask, request
import json
from created import MatchUser
app = Flask(__name__)

@app.route('/match')
def hello_world():
   return 'Hello World'

@app.route('/',methods = ['POST','GET'])
def match_user():
   if request.method == "POST":
      input_json = request.get_json()
      print(type(input_json))
      match_user = MatchUser(input_json)
      users = match_user.matched_users()
      return users
   else:
      return "USERS"

if __name__ == '__main__':
   app.run()
