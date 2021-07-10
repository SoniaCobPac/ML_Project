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
            options=["Welcome", "Visualization", "Json API-Flask", "Model Prediction", "Models From SQL Database", "Data Json"])

path1 = route(3) + sep + "data" + sep + "output" 
path2 = route(3) + sep + "resources" 
path3 = route(3) + sep + "data" + sep + "converted_data"

if menu == "Welcome":
    dash.menu_welcome(path2, "title.jpeg")
elif menu == "Visualization":
    filename = "df_music.csv"
    dash.menu_visualization(path1, filename)

elif menu == "Json API-Flask":
    dash.menu_dataframe(path1, "df_music.csv")
elif menu == "Model Prediction":
    dash.menu_prediction(path1, path2)

elif menu == "Models From SQL Database":
    path = route(3) + sep + "reports"
    dash.menu_temperature_graphs(path, "all_temp.png", "2014_temp_plot.png", "2015_temp_plot.png", "2016_temp_plot.png",\
         "2017_temp_plot.png", "2018_temp_plot.png", "2019_temp_plot.png")

elif menu == "Data Json": 
    dash.menu_datajson("http://localhost:6060/give_me_id?password=H71525533")
