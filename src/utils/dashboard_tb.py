import streamlit as st
import os
import sys
import pandas as pd 
import webbrowser
from PIL import Image
import requests


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


def menu_visualization(path1, filename):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    df = pd.read_csv(path1 + sep + filename)
    pieces = [elem for elem in df["Piece"]]
   
    song = st.selectbox("Choose a piece", pieces)

    #df = df[df["Piece"] == song]
    data_piece = df["Piece"]
    data_notes = df["Notes"]
    d = dict((note, number) for number, note in enumerate(pieces))  

    st.pyplot(vis.plot_one_song(data_notes, data_piece, d[song]))


def menu_dataframe(path1, filename):
    st.write("The below dataframe includes a list of pieces used to train a neural network to generate music and all their notes.")
    df_writed = pd.read_csv(path1 + sep + filename)
    print(st.dataframe(df_writed))


def menu_prediction(path1, path2):

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.write("If you wish to make a prediction from a music file, please upload it here.\nThe file must be a MIDI format")
    st.sidebar.subheader("Upload a MIDI file")

    
    if gases == "All":
        all_pol = pd.read_csv(path1 + sep + filename)
        st.pyplot(vis.pollutant_evolution_all(all_pol, "YEAR", "VALUE", "POLLUTANT", "POLLUTANT", "GASES"))

    elif gases == "PM10":
        pm10 = pd.read_csv(path1 + sep + filename5)
        st.pyplot(vis.pollutant_evolution_one(pm10, "PM10"))
        image5 = Image.open(path2 + sep + imag5)
        st.image (image5,use_column_width=True)

def menu_temperature_graphs(path, imag1, imag2, imag3, imag4, imag5, imag6, imag7):

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.sidebar.subheader("Years:")
    years = st.sidebar.selectbox("Choose a year:", options=["All", "2014", "2015", "2016", "2017", "2018", "2019"])
    
    if years == "All": 
        image1 = Image.open(path + sep + imag1)
        st.image (image1,use_column_width=True)

    elif years == "2014":
        image2 = Image.open(path + sep + imag2)
        st.image (image2,use_column_width=True)

    elif years == "2015":
        image3 = Image.open(path + sep + imag3)
        st.image (image3,use_column_width=True)

    elif years == "2016":
        image4 = Image.open(path + sep + imag4)
        st.image (image4,use_column_width=True)

    elif years == "2017":
        image5 = Image.open(path + sep + imag5)
        st.image (image5,use_column_width=True)

    elif years == "2018":
        image6 = Image.open(path + sep + imag6)
        st.image (image6,use_column_width=True)

    elif years == "2019":
        image7 = Image.open(path + sep + imag7)
        st.image (image7,use_column_width=True)

        
def menu_correlation(path, imag):
    image = Image.open(path + sep + imag)
    st.image(image,use_column_width=True)  
    st.write("Correlation between pollutants (represented as value in the above figure) \
        and temperature does not exist. The only correlation shown in the graph is between the \
        temperature of different years. However, these aren't two variables we can relate.")

def menu_datajson(url): 
    resp = requests.get(url).json()
    st.write(resp)
