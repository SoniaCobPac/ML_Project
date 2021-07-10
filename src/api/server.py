import os
from flask import Flask, request
import sys
import pandas as pd 
import argparse
import json 
from sqlalchemy import create_engine

def get_route(steps):
    """
    This function appends the route of the file to the sys path
    to be able to import files from/to other foders.

    Param: Steps (int) to go up to the required folder
    """
    route = os.path.abspath(__file__)
    for i in range(steps):
        route = os.path.dirname(route)
    sys.path.append(route)
    return route 

get_route(2)
from utils.folders_tb import read_json
import utils.mining_data_tb as md
from utils.sql_tb import MySQL
 
app = Flask(__name__)   

@app.route("/") 
def home():
    return "To access the machine learning project 'Generating Music with Machine Learning' please add the endpoint '/give_me_id?password=' and enter the correct password."
   
# json - cleaned and treated data
@app.route("/give_me_id", methods=['GET']) # "localhost:6060/give_me_id?password=H71525533"
def give_id():
    x = request.args['password']    
    if x == "H71525533":    
        df_music = pd.read_csv(get_route(3) + os.sep + "data" + os.sep + "output" + os.sep + "df_music.csv", encoding="latin-1") 
        return json.dumps(json.loads(df_music.to_json()), indent=4) 
    else:
        return "Wrong password"


# json - dataframe in SQL 
@app.route("/upload_form", methods = ['POST', 'GET'])
def upload_dataframe():
    if request.method == 'POST':
    # Import the password and all sensitive data from 'sql_setting.json'
        json_readed = read_json(get_route(1) + os.sep , "sql_setting.json")
        mysql_db = MySQL(json_readed["IP_DNS"], json_readed["USER"], json_readed["PASSWORD"], json_readed["BD_NAME"], json_readed["PORT"])
        # Connection to database
        db_connection_str = mysql_db.SQL_ALCHEMY
        db_connection = create_engine(db_connection_str)
        # read dataframe
        df_music = md.read_dataframe(get_route(3) + os.sep + "data" + os.sep + "output" + os.sep, "df_music")
        df_music.reset_index(inplace=True)
        df_music.rename(columns={"index": "ID"}, inplace=True)
        # Insert dataframe in SQL
        df_music.to_sql("sonia_cobo_pacios", db_connection, index=False, if_exists="replace")
    # Message in Flask
    elif request.method == 'GET':
        # SELECT it is possible to check if the table has been inserted correctly
        json_readed = read_json(get_route(1) + os.sep , "sql_setting.json")
        mysql_db = MySQL(json_readed["IP_DNS"], json_readed["USER"], json_readed["PASSWORD"], json_readed["BD_NAME"], json_readed["PORT"])
        try:
            select_sql = """SELECT * FROM sonia_cobo_pacios"""
            mysql_db.execute_get_sql(sql=select_sql)
            return "The dataframe has been inserted successfully in SQL"
        except Exception as error:
            return "Error: The dataframe couldn't be inserted in SQL", error
            

def main():
    print("---------STARTING PROCESS---------")    
    json_readed = read_json(get_route(1) + os.sep , "settings.json")
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