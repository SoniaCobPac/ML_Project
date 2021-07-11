import streamlit as st
import os
import sys
import pandas as pd 
import webbrowser
from PIL import Image
import requests
import json 
from sqlalchemy import create_engine


sep = os.sep
def route(steps):
    route = os.path.abspath(__file__)
    for i in range(steps):
        route = os.path.dirname(route)
    sys.path.append(route)
    return route

route(2)

import utils.visualization_tb as vis
import utils.mining_data_tb as md
from utils.folders_tb import read_json
from utils.sql_tb import MySQL

sep = os.sep

def menu_welcome(path, imag):
    st.title("Generating Music with Artificial Intelligence")
    st.write("Author: Sonia Cobo")
    st.write("Date: July 2021")
    st.write("This project has been done while studying a Data Science course in The Bridge (Madrid).")
    st.write(" ")
    image = Image.open(path + sep + imag)
    st.image (image)

    st.write("Music is associated with emotions, experiences and creativity, all o them considered human's qualities.")
    st.write("Though this project doesn't have a hypothesis per se it was done to prove that technology has advanced to the \
        point where a machine, that cannot experience these feelings, can generate music.")
    st.write("This project will explain and show how music has been generated using a differente types of neural networks in Python.")

    # clean garbage
    import gc
    gc.collect()

def menu_visualization(path1, filename, path3):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    df = pd.read_csv(path1 + sep + filename)
    pieces = [elem for elem in df["Piece"]]
   
    song = st.selectbox("Choose a piece", pieces)

    data_piece = df["Piece"]
    data_notes = df["Notes"]

    st.pyplot(vis.plot_song_streamlit(data_notes, data_piece, song+".mid", path3))


def menu_dataframe(path1, filename):
    st.write("The below dataframe includes a list of pieces used to train a neural network to generate music and all their notes.")
    df_writed = pd.read_csv(path1 + sep + filename)
    print(st.dataframe(df_writed))


def menu_prediction(file1, file11, file5, file6, file7, file8, file9, file10):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    tests = st.sidebar.selectbox(" ",
        options=["First trial", "LSTM Model", "Play Time"])

    if tests == "First trial":
        audio_file1 = open(file1, 'rb')
        audio_bytes1 = audio_file1.read()
        st.title("Generated melodies")
        st.write("Some tests have been done with different MIDI files and neural networks.")
        st.write(" ")
        st.write("The first test was done with the piece Prelude (1890) by Albeniz:")
        st.audio(audio_bytes1, format='wav')
        st.write("The generated sound from this piece was the following (Please bear in mind this was the first test, enjoy!):")
        audio_file11 = open(file11, 'rb')
        audio_bytes11 = audio_file11.read()
        st.audio(audio_bytes11, format='wav')

    elif tests == "LSTM Model":
        st.title("Generated melodies")
        st.write("The following audio was predicted with a LSTM model, 10 epoch, 50 songs")
        audio_file5 = open(file5, 'rb')
        audio_bytes5 = audio_file5.read()
        st.audio(audio_bytes5, format='wav') 
        st.write("The following audio was predicted with a LSTM model, 1 epoch, 50 songs")
        audio_file6 = open(file6, 'rb')
        audio_bytes6 = audio_file6.read()
        st.audio(audio_bytes6, format='wav')
        st.write("The following audio was predicted with a LSTM model, 100 epoch, 1 song")
        audio_file7 = open(file7, 'rb')
        audio_bytes7 = audio_file7.read()
        st.audio(audio_bytes7, format='wav')

    elif tests == "Play Time":
        st.title("Generated melodies")
        st.write("Though the models have been trained to predict classical music from one instrument, they were also tested with more variety of songs.")
        st.write("The following audio was predicted using Never Gonna Give you Up")
        audio_file8 = open(file8, 'rb')
        audio_bytes8 = audio_file8.read()
        st.audio(audio_bytes8, format='wav')
        st.write("The following audio was predicted using Guns and Roses: Sweet Child of Mine")
        audio_file9 = open(file9, 'rb')
        audio_bytes9 = audio_file9.read()
        st.audio(audio_bytes9, format='wav')
        st.write("The following audio was predicted using the theme song Naruto")
        audio_file10 = open(file10, 'rb')
        audio_bytes10 = audio_file10.read()
        st.audio(audio_bytes10, format='wav')


        
def menu_model_from_sql():
    json_readed = read_json(route(2) + os.sep + "api" + os.sep , "sql_setting.json")
    mysql_db = MySQL(json_readed["IP_DNS"], json_readed["USER"], json_readed["PASSWORD"], json_readed["BD_NAME"], json_readed["PORT"])
    # Connection to database
    mysql_db.connect()
    db_connection_str = mysql_db.SQL_ALCHEMY
    db_connection = create_engine(db_connection_str)
    
    select_sql = """SELECT * FROM model_comparison"""
    select_result = mysql_db.execute_get_sql(sql=select_sql)
    col = ["INDEX", "MODEL", "PARAMETERS", "LOSS", "RMSE", "ACCURACY"]
    df = pd.DataFrame(select_result, columns=[col])
    df = df.iloc[: , 1:]
    st.write("Accurary and evaluation comparison of neural networks used in this project and their parameters.")
    st.write(df)

