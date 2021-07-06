import streamlit as st
import os
import sys
import pandas as pd 
import webbrowser
from PIL import Image
import requests


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utils.visualization_tb as vis

sep = os.sep

def menu_welcome(path, imag):
    st.title("Generating Music with Machine Learning")
    st.write("Author: Sonia Cobo")
    image = Image.open(path + sep + imag)
    st.image (image)

    st.write("This project has been done while studying a Data Science course in The Bridge (Madrid).")
    st.write("Its objective is to generate music using LSTM models.")
    st.write(" ")
    st.write("Please use the sidebar menu to go to more dashboards for information.")

def menu_introduction(path, imag1, imag2, imag3, imag4, imag5):
    st.title("EDA project: Relationship between temperature and air quality")    
    
    
    st.write("This project studies the relationship between air quality and temperature.\
        It is well-known that the increase in average temperature due to climate change has a negative impact to air quality.")

    st.write("The project's aim is to prove this relationship, particularly when comparing temperature to ozone concentration, \
        as the hypothesis is that ozone is the main pollutant affecting air quality due to climate change.")

    st.write("To prove this hypothesis the project considers the behaviour of several harmful pollutant, \
        such as ozone (O3), sulphur dioxide (SO2), nitrogen dioxide (NO2), carbon monoxide (CO) and Particles Matter 10 (PM10) \
            over the past six years in relation to temperature changes.")
    
    st.write("Select a pollutant below for more information about it:")
    options = st.selectbox("",["None", "O3", "SO2", "NO2", "CO", "PM10"])
    
    if options == "O3":
        col1, col2 = st.beta_columns([1, 2])
        with col1:
            image1 = Image.open(path + sep + imag1)
            st.image (image1,use_column_width=True)
        with col2:   
            if st.button("Link to the European Environment Agency"):
                webbrowser.open_new_tab("https://www.eea.europa.eu/publications/2-9167-057-X/page022.html")
            st.write("Ozone is a molecule composed of three atoms of oxygen.\
                    Ozone in the upper atmosphere helps filter out damaging ultraviolet radiation from the sun.\
                    However, ozone in the atmosphere, which is the air we breathe, is harmful to the respiratory system.\
                    When inhaled, ozone can damage the lungs.\
                    Relatively low amounts can cause chest pain, coughing, shortness of breath and throat irritation.")

    elif options == "SO2":
        col1, col2 = st.beta_columns([1, 2])
        with col1:
            image2 = Image.open(path + sep + imag2)
            st.image (image2,use_column_width=True)
        with col2:   
            if st.button("Link to the European Environment Agency"):
                webbrowser.open_new_tab("https://www.eea.europa.eu/data-and-maps/indicators/eea-32-sulphur-dioxide-so2-emissions-1")
            st.write("Sulfur dioxide is a colorless gas at ambient temperature and pressure.\
            The main sources of sulfur dioxide emissions are from fossil fuel combustion and natural volcanic activity.\
            Sulphur dioxide contributes to respiratory illness by making breathing more difficult.")
        
    elif options == "NO2":
        col1, col2 = st.beta_columns([1, 2])
        with col1:
            image3 = Image.open(path + sep + imag3)
            st.image (image3,use_column_width=True)
        with col2:   
            if st.button("Link to the European Environment Agency"):
                webbrowser.open_new_tab("https://www.eea.europa.eu/themes/air/air-quality/resources/glossary/nitrogen-oxides")
            st.write("Nitrogen Dioxide (NO2) is one of a group of highly reactive gases known as oxides of nitrogen or nitrogen oxides (NOx).\
                Road traffic is the principal outdoor source of nitrogen dioxide.\
                Breathing air with a high concentration of NO2 can irritate airways in the human respiratory system.")
    
    elif options == "CO":
        col1, col2 = st.beta_columns([1, 2])
        with col1:
            image4 = Image.open(path + sep + imag4)
            st.image (image4,use_column_width=True)
        with col2:   
            if st.button("Link to the European Environment Agency"):
                webbrowser.open_new_tab("https://www.eea.europa.eu/publications/2-9167-057-X/page024.html")
            st.write("Carbon Monoxide is an odorless, tasteless, poisonous gas, CO, that results from the incomplete combustion of carbon.\
                Common sources of CO in cases of poisoning include house fire, motor-vehicle exhaust and faulty domestic heating systems.\
                It is very flammable in air over a wide range of concentrations.")

    elif options == "PM10":
        col1, col2 = st.beta_columns([1, 2])
        with col1:
            image5 = Image.open(path + sep + imag5)
            st.image (image5,use_column_width=True)
        with col2:   
            if st.button("Link to the European Environment Agency"):
                webbrowser.open_new_tab("https://ec.europa.eu/environment/air/quality/standards.htm")
            st.write("Particulate Matter (PM10) describes inhalable particles, with diameters that are generally 10 micrometers and smaller.\
                Natural dust is the main source of PM10 in the Middle-East and Northern African countries.\
                Sea salt is the most important natural source of PM10 in north-western Europe.\
                Short-term exposures to PM10 have been associated primarily with worsening of respiratory diseases, \
                including asthma and chronic obstructive pulmonary ")

def menu_dataframe(path, filename):
    df_writed = pd.read_csv(path + sep + filename)

    options = st.selectbox("Filter by pollutants", ["All", "O3", "SO2", "NO2", "CO", "PM10"])
    options1 = st.selectbox("Filter by region", ["All", 'MADRID', 'OURENSE', 'A CORUÑA', \
        'LUGO', 'PONTEVEDRA', 'BARCELONA', 'GIRONA', 'TARRAGONA', 'LLEIDA', 'ASTURIAS', \
        'LA RIOJA', 'CÁCERES', 'BADAJOZ', 'NAVARRA', 'ÁVILA', 'BURGOS', 'PALENCIA', 'SEGOVIA',\
        'VALLADOLID', 'ZAMORA', 'ALBACETE', 'CIUDAD REAL', 'GUADALAJARA', 'TOLEDO', 'ISLAS BALEARES',\
        'HUELVA', 'GRANADA', 'MURCIA', 'CANTABRIA', 'CUENCA', 'ZARAGOZA', 'LEÓN', 'SORIA', 'SALAMANCA',\
        'LAS PALMAS', 'SANTA CRUZ DE TENERIFE', 'ALMERÍA', 'CÁDIZ', 'CÓRDOBA', 'JAÉN', 'MÁLAGA', 'SEVILLA',\
        'HUESCA', 'ALICANTE', 'TERUEL'])
    if options == "All" and options1 == "All":
        print(st.dataframe(df_writed))
    elif options == "All" or options1 == "All":
        df_writed = df_writed[(df_writed["POLLUTANT"] == options) | (df_writed["REGION"] == options1)]
        print(st.dataframe(df_writed))
    elif options != "All" and options1 != "All":
        df_writed = df_writed[(df_writed["POLLUTANT"] == options) & (df_writed["REGION"] == options1)]
        print(st.dataframe(df_writed))

    st.write("Pollutant concentration is represented in micrograms per cubic meter air (µg/m3)")
    st.write("Temperature is represented in degrees celsius (ºC)")

def menu_gas_graphs(path1, path2, filename, filename1, filename2, filename3, filename4, filename5,\
     imag1, imag2, imag3, imag4, imag5):

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.sidebar.subheader("Pollutant:")
    gases = st.sidebar.selectbox("Choose a pollutant:", options=["All","O3", "SO2", "NO2", "CO", "PM10"])
    
    if gases == "All":
        all_pol = pd.read_csv(path1 + sep + filename)
        st.pyplot(vis.pollutant_evolution_all(all_pol, "YEAR", "VALUE", "POLLUTANT", "POLLUTANT", "GASES"))
    
    elif gases == "O3":
        o3 = pd.read_csv(path1 + sep + filename1)
        st.pyplot(vis.pollutant_evolution_one(o3, "O3"))  
        st.write("The drop of O3 concentration in 2016 wasn't consistent along time, so it cannot be considered as an improvement of air quality.")
        image1 = Image.open(path2 + sep + imag1)
        st.image (image1,use_column_width=True)
    
    elif gases == "SO2":
        so2 = pd.read_csv(path1 + sep + filename2)
        st.pyplot(vis.pollutant_evolution_one(so2, "SO2"))
        st.write("SO2 reduction has been achieved as a result of a combination of measures, including fuel-switching \
            in energy-related sectors away from high-sulphur solid and liquid fuels to low-sulphur fuels such as natural gas")
        image2 = Image.open(path2 + sep + imag2)
        st.image (image2,use_column_width=True)
  
    elif gases == "NO2":
        no2 = pd.read_csv(path1 + sep + filename3)
        st.pyplot(vis.pollutant_evolution_one(no2, "NO2")) 
        st.write("Average NO2 concentrations have decreased over the years due to tougher environment regulations.") 
        image3 = Image.open(path2 + sep + imag3)
        st.image (image3,use_column_width=True)
    
    elif gases == "CO":
        co = pd.read_csv(path1 + sep + filename4)
        st.pyplot(vis.pollutant_evolution_one(co, "CO"))
        st.write("")
        image4 = Image.open(path2 + sep + imag4)
        st.image (image4,use_column_width=True)
    
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
