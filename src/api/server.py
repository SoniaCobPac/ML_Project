import os
from flask import Flask, request
import sys
import pandas as pd 
import argparse
import json 

def get_route(steps):
    route = os.path.abspath(__file__)
    for i in range(steps):
        route = os.path.dirname(route)
    sys.path.append(route)
    return route 

get_route(2)
from utils.folders_tb import read_json

app = Flask(__name__)   

@app.route("/") 
def home():
    return "To access the machine learning project 'Generating Music with Machine Learning' please add the endpoint '/give_me_id?password=' and enter the correct password."
   
# json - cleaned and treated data
@app.route("/give_me_id", methods=['GET']) # "localhost:6060/give_me_id?password=H71525533"
def give_id():
    x = request.args['password']    
    if x == "H71525533":    
        df = pd.read_csv(get_route(3) + os.sep + "data" + os.sep + "output" + os.sep + "df_music.csv", encoding="latin-1") 
        return json.dumps(json.loads(df.to_json())) 
    else:
        return "Wrong password"

# json - model output
@app.route("/give_me_id", methods=['GET']) # "localhost:6060/give_me_id?password=H71525533"
def give_id():
    pass

def main():
    print("---------STARTING PROCESS---------")    
    json_readed = read_json(get_route(1), "settings")
    # Json variables are loaded:
    DEBUG = json_readed["debug"]
    HOST = json_readed["host"]
    PORT_NUM = json_readed["port"]
    app.run(debug=DEBUG, host=HOST, port=PORT_NUM)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-x", type=str, help="Required password", required=True)
    args = vars(parser.parse_args())
    x = args["x"] 
    if x == "sonia":
        main()  
    else:
        print("Wrong password") 