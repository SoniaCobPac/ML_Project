import streamlit as st
import os
import sys


sep = os.sep
def route(steps):
    route = os.path.abspath(__file__)
    for i in range(steps):
        route = os.path.dirname(route)
    sys.path.append(route)
    return route

route(2)
import utils.dashboard_tb as dash
import utils.mining_data_tb as md


menu = st.sidebar.selectbox('Menu:',    
            options=["Welcome", "Visualization", "Json API-Flask", "Model Prediction", "Models From SQL Database"])

path1 = route(3) + sep + "data" + sep + "output" 
path2 = route(3) + sep + "resources" 
path3 = route(3) + sep + "data" + sep + "converted_data"
path4 = route(3) + sep + "reports" + sep 
path5 = route(3) + sep + "data" + sep + "wav_files" + sep

if menu == "Welcome":
    dash.menu_welcome(path2, "title.jpeg")

elif menu == "Visualization":
    filename = "df_music.csv"
    dash.menu_visualization(path1, filename, path3)

elif menu == "Json API-Flask":
    dash.menu_dataframe(path1, "df_music.csv")

elif menu == "Model Prediction":
    file11 = path5 + "1st_baseline_lstm_10epoch_1song_1.wav"
    file1 = path5 + "alb_esp1.wav"
    file5 = path5 + "baseline_lstm_10epoch_chopin_1.wav"
    file6 = path5 + "baseline_lstm_1epoch_chopin_1.wav" 
    file7 = path5 + "baseline_lstm_100epoch_1song_1.wav"
    file8 = path5 + "GiveUp.wav"
    file9 = path5 + "GunsnRoses.wav"
    file10 = path5 + "Naruto.wav"
    dash.menu_prediction(file1, file11, file5, file6, file7, file8, file9, file10)

elif menu == "Models From SQL Database":
    dash.menu_model_from_sql()
