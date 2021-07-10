import os
from flask import Flask, request
import sys
import pandas as pd 
import json
from sqlalchemy import create_engine

def get_route(steps):
    route = os.path.abspath(__file__)
    for i in range(steps):
        route = os.path.dirname(route)
    sys.path.append(route)
    return route 

get_route(2)
from utils.folders_tb import read_json
import utils.mining_data_tb as md
from utils.sql_tb import MySQL

print(get_route(2))
def upload_comparison_dataframe(path, dataframe):
    # Import the password and all sensitive data from 'sql_setting.json'
    json_readed = read_json(get_route(2) + os.sep + "api" + os.sep, "sql_setting.json")
    mysql_db = MySQL(json_readed["IP_DNS"], json_readed["USER"], json_readed["PASSWORD"], json_readed["BD_NAME"], json_readed["PORT"])
    # Connection to database
    db_connection_str = mysql_db.SQL_ALCHEMY
    db_connection = create_engine(db_connection_str)
    # read dataframe
    df_results = md.read_dataframe(path, dataframe)
    df_results.reset_index(inplace=True)
    df_results.rename(columns={"index": "ID"}, inplace=True)
    # Insert dataframe in SQL
    df_results.to_sql("model_comparison", db_connection, index=False, if_exists="replace")
 