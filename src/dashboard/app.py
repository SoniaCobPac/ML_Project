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


menu = st.sidebar.selectbox('Menu:',    
            options=["Welcome", "Introduction", "Json API-Flask", "Visualization", "Model Prediction", "Models From SQL Database", "Data Json"])

path1 = route(3) + sep + "data" + sep + "output"
path2 = route(3) + sep + "resources"

if menu == "Welcome":
    dash.menu_welcome(path2, "title.jpeg")
elif menu == "Introduction":
    dash.menu_introduction(path2, "o3.png", "so2.png", "no2.png", "co.png", "pm10.png")
elif menu == "Json API-Flask":
    dash.menu_dataframe(path1, "df_music.csv")
elif menu == "Gas Graphs":
    dash.menu_gas_graphs(path1, path2, "air.csv", "o3_mean.csv", "so2_mean.csv", "no2_mean.csv", "co_mean.csv", "pm10_mean.csv",\
        "ozone_limits.PNG", "so2_limits.PNG", "no2_limits.PNG", "co_limits.PNG", "pm10_limits.PNG")
elif menu == "Temperature Graphs":
    path = route(3) + sep + "reports"
    dash.menu_temperature_graphs(path, "all_temp.png", "2014_temp_plot.png", "2015_temp_plot.png", "2016_temp_plot.png",\
         "2017_temp_plot.png", "2018_temp_plot.png", "2019_temp_plot.png")
elif menu == "Correlation":
    path = route(3) + sep + "reports"
    dash.menu_correlation(path, "heatmap_corr.png")
elif menu == "Data Json": 
    dash.menu_datajson("http://localhost:6060/give_me_id?password=H71525533")
