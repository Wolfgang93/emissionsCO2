import streamlit as st
import pathlib
import os
from datetime import date
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import chart_studio.plotly as py
import plotly.offline as po
import plotly.graph_objs as pg
import matplotlib.pyplot as plt
#librerías ML
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
def load_model():
    # Unauthenticated client only works with public data sets. Note 'None'
    # in place of application token, and no username or password:
    
    url1 = 'rawDataset/'
    url2 = 'Tablas/'

    data_dir_1 = pathlib.Path(url1)
    data_dir_2 = pathlib.Path(url2)

    class_names1 = np.array(sorted([item.name for item in data_dir_1.glob('*')]))
    class_names2 = np.array(sorted([item.name for item in data_dir_2.glob('*')]))
    energyco2 = pd.read_csv(os.path.join(url1, 'energyco2.csv'))
    energyconsumption = pd.read_csv(os.path.join(url1, 'owid-energy-consumption-source.csv'))
    owidco2 = pd.read_csv(os.path.join(url1, 'owid-co2-data.csv'))
    countries = pd.read_csv(os.path.join(url2, 'country.csv'))
    country = countries.country.to_list()   
    energyco2.drop('Unnamed: 0', axis = 1, inplace = True)
    energyco2['Energy_consumption'] = energyco2['Energy_consumption'] * 293.07
    energyco2['Energy_production'] = energyco2['Energy_production'] * 293.07
    energyco2 = energyco2.round(3)
    energyco2 = energyco2.replace('Cabo Verde','Cape Verde')
    energyco2 = energyco2.replace('U.S. Virgin Islands','United States Virgin Islands')
    energyco2 = energyco2.replace('Côte d’Ivoire',"Cote d'Ivoire")
    energyco2 = energyco2.replace('Guinea-Bissau','Gambia')
    energyco2 = energyco2.replace('Czech Republic','Czechia')
    energyco2 = energyco2.replace('The Bahamas','Bahamas')
    energyco2 = energyco2.replace('Burma','Myanmar')
    energyco2 = energyco2.replace('Macau','Macao')
    energyco2 = energyco2.replace('Gambia, The','Gambia')
    energyco2 = energyco2[energyco2.Country.isin(country + ['World'])]
    energyco2 = energyco2[energyco2['Year'] >= 1980][energyco2['Energy_type'] == 'all_energy_types']
    energyco2 = energyco2[['Country', 'Year', 'Energy_consumption',
        'Energy_production']]
    energyco2.columns = ['country', 'year', 'energyConsumption',
        'energyProduction']
    energyconsumption = energyconsumption[['country','year','population','gdp']]
    energyconsumption.drop_duplicates(inplace=True)
    energyconsumption = energyconsumption[energyconsumption['year'] >= 1980]
    energy_consumption = energyconsumption.replace(',', ' ', regex=True)
    energyconsumption = energyconsumption.replace('Micronesia (country)','Micronesia')
    energyconsumption = energyconsumption.replace('Central America (BP)','Central America')
    energyconsumption = energyconsumption.replace('Guinea-Bissau','Guinea Bissau')
    energyconsumption = energyconsumption.replace('Timor','Timor-Leste')
    energyconsumption= energyconsumption[energyconsumption.country.isin(country)]
    energyconsumption.drop_duplicates(subset = ['country', 'year'],  inplace=True)
    owidco2 = owidco2[['country', 'co2','year', 'co2_per_capita', 'gdp', 'population']]
    owidco2 = owidco2[owidco2['year'] >= 1980]
    owidco2 = owidco2.replace('Micronesia (country)','Micronesia')
    owidco2 = owidco2.replace('Timor','Timor-Leste')
    owidco2 = owidco2.replace('Guinea-Bissau','Guinea Bissau')
    owidco2 = owidco2[owidco2['country'].isin(country + ['World'])]
    owidco2.columns = ['country', 'co2emissions','year', 'co2PerCapita', 'gdp', 'population']
    merge = owidco2.merge(energyco2, how="left", on = ["country", "year"])
    datos = merge[~merge.country.isin(['Antarctica', 'Eritrea', 'Micronesia', 'Namibia', 'Puerto Rico',
       'Saint Helena', 'Timor-Leste', 'Antigua and Barbuda', 'Aruba', 'Bahamas', 'Belize', 'Bermuda',
       'Bhutan', 'British Virgin Islands', 'Brunei', 'Cook Islands',
       'Faeroe Islands', 'Fiji', 'French Guiana', 'French Polynesia',
       'Greenland', 'Grenada', 'Guadeloupe', 'Guyana', 'Kiribati',
       'Kosovo', 'Macao', 'Maldives', 'Martinique', 'Montserrat', 'Nauru',
       'New Caledonia', 'Niue', 'Papua New Guinea',
       'Reunion', 'Saint Kitts and Nevis', 'Saint Pierre and Miquelon',
       'Saint Vincent and the Grenadines', 'Samoa', 'Slovakia',
       'Solomon Islands', 'Somalia', 'South Sudan', 'Sudan', 'Suriname',
       'Tonga', 'Turks and Caicos Islands', 'Tuvalu'
       ,'Vanuatu', 'Armenia', 'Azerbaijan', 'Belarus', 'Bosnia and Herzegovina',
       'Congo','Democratic Republic of Congo',
       'Estonia', 'Georgia', 'Guinea Bissau', 'Kazakhstan',
       'Kyrgyzstan', 'Latvia', 'Lithuania', 'Moldova', 'Montenegro',
       'North Macedonia', 'Palestine', 'Slovenia',
       'Tajikistan', 'Turkmenistan', 'Uzbekistan', 'Central African Republic', 'Equatorial Guinea', 'Mauritania'])]
    datos['co2emissions'] = datos['co2emissions']*(10**6)
    df_Model1 = datos[["country","year", "gdp", "population", "energyConsumption", "co2emissions"]]
    df_Model1 = df_Model1.dropna()
    return df_Model1


def prediccion_poblacion_dependiente_tiempo(df_Model1, anho_prediction):

    X = df_Model1.iloc[:,1:2].values  
    y_Population = df_Model1.iloc[:,3].values
    pre_process = PolynomialFeatures(degree=3)
    X_poly = pre_process.fit_transform(X)
    pr_model = LinearRegression()
    pr_model.fit(X_poly, y_Population)
    y_pred = pr_model.predict(X_poly)
    y_new_poblacion = pr_model.predict(pre_process.fit_transform([[anho_prediction]]))
    return y_new_poblacion

def prediccion_PBI_dependiente_tiempo(df_Model1, anho_prediction):
    #PBI
    X = df_Model1.iloc[:,1:2].values  
    y_PBI = df_Model1.iloc[:,2:3].values
    pr_model_2 = LinearRegression()
    pr_model_2.fit(X, y_PBI)
    y_pred2 = pr_model_2.predict(X)
    y_new_PBI = pr_model_2.predict([[anho_prediction]])
    return y_new_PBI

def prediccion_emisiones_de_consumo_energético_dependiente_tiempo_poblacion_PBI(df_Model1, anho_prediction):
    X = df_Model1.iloc[:,1:4].values  
    y_Consumption = df_Model1.iloc[:,4].values
    pre_process = PolynomialFeatures(degree=3)
    X_poly = pre_process.fit_transform(X)
    pr_model = LinearRegression()
    pr_model.fit(X_poly, y_Consumption)
    y_pred = pr_model.predict(X_poly)
    y_new_poblacion = prediccion_poblacion_dependiente_tiempo(df_Model1, anho_prediction)
    y_new_PBI = prediccion_PBI_dependiente_tiempo(df_Model1, anho_prediction)
    y_new_consumo = pr_model.predict(pre_process.fit_transform([[anho_prediction, y_new_PBI, y_new_poblacion]]))
    RMS_grado3 = mean_squared_error(y_Consumption, y_pred)
    y_lista = []
    anho_ultimo2 = 2018
    anho_ultimo = 2019
    X_year = df_Model1['year']
    while (anho_ultimo<=2030):
        y_lista.append(pr_model.predict(pre_process.fit_transform([[anho_ultimo, prediccion_PBI_dependiente_tiempo(df_Model1, anho_ultimo), prediccion_poblacion_dependiente_tiempo(df_Model1, anho_ultimo)]]))[0])
        anho_ultimo += 1
    rango = anho_prediction - anho_ultimo2+1
    lista = list(range(2018,anho_prediction +1))
    y_lista.insert(0, list(y_pred).pop())
    fig = plt.figure(figsize=(10,5))
    plt.scatter(X_year , y_Consumption, c = "black")
    plt.xlabel("Año")
    plt.ylabel("Consumo energético TWh")
    plt.plot(X_year, y_pred)
    plt.plot(lista, y_lista[:rango],c = "red", linestyle ="--")
    plt.scatter(anho_prediction, y_new_consumo, c = "red")
    st.pyplot(fig)
    return y_Consumption, y_pred, y_new_consumo, RMS_grado3


def prediccion_emisiones_de_CO2_dependiente_tiempo_poblacion_PBI(df_Model1, anho_prediction, pais):
    X = df_Model1.iloc[:,1:4].values  
    y_Emisiones = df_Model1.iloc[:,5].values
    grado = 3
    if pais == "United States":
        grado = 4
    
    pre_process = PolynomialFeatures(degree=grado)
    X_poly = pre_process.fit_transform(X)
    pr_model = LinearRegression()
    pr_model.fit(X_poly, y_Emisiones)
    y_pred = pr_model.predict(X_poly)
    y_new_poblacion = prediccion_poblacion_dependiente_tiempo(df_Model1, anho_prediction)
    y_new_PBI = prediccion_PBI_dependiente_tiempo(df_Model1, anho_prediction)
    y_new_consumo = pr_model.predict(pre_process.fit_transform([[anho_prediction, y_new_PBI, y_new_poblacion]]))
    RMS_grado3 = mean_squared_error(y_Emisiones, y_pred)
    y_lista = []
    anho_ultimo2 = 2018
    anho_ultimo = 2019
    X_year = df_Model1['year']
    while (anho_ultimo<=2030):
        y_lista.append(pr_model.predict(pre_process.fit_transform([[anho_ultimo, prediccion_PBI_dependiente_tiempo(df_Model1, anho_ultimo), prediccion_poblacion_dependiente_tiempo(df_Model1, anho_ultimo)]]))[0])
        anho_ultimo += 1
    rango = anho_prediction - anho_ultimo2+1
    lista = list(range(2018,anho_prediction +1))
    y_lista.insert(0, list(y_pred).pop())
    fig = plt.figure(figsize=(10,5))
    plt.scatter(X_year , y_Emisiones, c = "black")
    plt.xlabel("Año")
    plt.ylabel("Emisiones TnCO2")
    plt.plot(X_year, y_pred)
    plt.plot(lista, y_lista[:rango],c = "red", linestyle ="--")
    plt.scatter(anho_prediction, y_new_consumo, c = "red")
    st.pyplot(fig)
    return y_Emisiones, y_pred, y_new_consumo, RMS_grado3

def modelo_polinomica_grado_3(df_Model1, anho_prediction):
    X = df_Model1.iloc[:,1:2].values  
    y_Consumption = df_Model1.iloc[:,4].values 
    pre_process = PolynomialFeatures(degree=3)
    X_poly = pre_process.fit_transform(X)
    pr_model = LinearRegression()
    pr_model.fit(X_poly, y_Consumption)
    y_pred = pr_model.predict(X_poly)
    theta0 = pr_model.intercept_
    _, theta1, theta2, theta3 = pr_model.coef_
    theta0, theta1, theta2, theta3
    y_new = pr_model.predict(pre_process.fit_transform([[anho_prediction]]))    
    RMS_grado3 = mean_squared_error(y_Consumption, y_pred)
    y_lista = []
    anho_ultimo2 = 2018
    anho_ultimo = 2019
    while (anho_ultimo<=2030):
        y_lista.append(pr_model.predict(pre_process.fit_transform([[anho_ultimo]]))[0])
        anho_ultimo += 1
    rango = anho_prediction - anho_ultimo2+1
    lista = list(range(2018,anho_prediction +1))
    y_lista.insert(0, list(y_pred).pop())
    fig = plt.figure(figsize=(10,5))
    plt.scatter(X, y_Consumption, c = "black")
    plt.xlabel("Año")
    plt.ylabel("Consumo energético TWh")
    plt.plot(X, y_pred)
    plt.plot(lista, y_lista[:rango],c = "red", linestyle ="--")
    plt.scatter(anho_prediction, y_new, c = "red")
    st.pyplot(fig)
    
    return X, y_Consumption, y_pred, y_new, RMS_grado3
def prediccion_emisiones_tiempo(df_Model1, anho_prediction):
    X = df_Model1.iloc[:,1:2].values  
    y_Emisiones = df_Model1.iloc[:,5].values 
    pre_process = PolynomialFeatures(degree=3)
    X_poly = pre_process.fit_transform(X)
    pr_model = LinearRegression()
    pr_model.fit(X_poly, y_Emisiones)
    y_pred = pr_model.predict(X_poly)
    theta0 = pr_model.intercept_
    _, theta1, theta2, theta3 = pr_model.coef_
    theta0, theta1, theta2, theta3
    y_new = pr_model.predict(pre_process.fit_transform([[anho_prediction]]))    
    RMS_grado3 = mean_squared_error(y_Emisiones, y_pred)
    y_lista = []
    anho_ultimo2 = 2018
    anho_ultimo = 2019
    while (anho_ultimo<=2030):
        y_lista.append(pr_model.predict(pre_process.fit_transform([[anho_ultimo]]))[0])
        anho_ultimo += 1
    rango = anho_prediction - anho_ultimo2+1
    lista = list(range(2018,anho_prediction +1))
    y_lista.insert(0, list(y_pred).pop())
    fig = plt.figure(figsize=(10,5))
    plt.scatter(X, y_Emisiones, c = "black")
    plt.xlabel("Año")
    plt.ylabel("Emisiones TnCO2")
    plt.plot(X, y_pred)
    plt.plot(lista, y_lista[:rango],c = "red", linestyle ="--")
    plt.scatter(anho_prediction, y_new, c = "red")
    st.pyplot(fig)
    
    return X, y_Emisiones, y_pred, y_new, RMS_grado3
anhos_prediction = (2020, 2021,2022,2023,2024,2025,2026,2027,2028,2029,2030)   
df_Model1 = load_model()
paises = df_Model1["country"].unique()

def show_Modelo_Regresión_Polinómica():
    st.markdown("<h1 style='text-align: center; color: navy;'>Pronóstico de consumo energético</h1>", unsafe_allow_html=True)
    df_Model1 = load_model()
    
    pais = st.selectbox("Elija un país", paises)
    df_Model1 = df_Model1[df_Model1["country"]== pais]
        
    
    anho_prediction = st.selectbox("Elija año a predecir", anhos_prediction)    
    st.subheader("&#8594; Predicción de consumo energético")
    st.write("")
    if st.button('Calcular consumo energético'):
        X, y_Consumption, y_pred, prediccion, error = modelo_polinomica_grado_3(df_Model1,anho_prediction)
        
        st.write("Con un modelo de regresión polinómica dependiente del año, el consumo energético para el año " + str(anho_prediction) + " es " + str(round(prediccion[0],2)) + " TWh con un error de " + str(round(error,2)))
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                col1.metric("consumo energético", f"{str(round(prediccion[0],2))} TWh")
            with col2:
                col2.metric("Error (MSE)", str(round(error,2)))
        y_Consumption2, y_pred2, prediccion2, error2  = prediccion_emisiones_de_consumo_energético_dependiente_tiempo_poblacion_PBI(df_Model1, anho_prediction)

        st.write("Con un modelo de regresión polinómica dependiente del PBI y de la problación, el consumo energético para el año " + str(anho_prediction) + " es " + str(round(prediccion2[0],2)) + " TWh con un error de " + str(round(error2,2)))
        
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                col1.metric("consumo energético", f"{str(round(prediccion2[0],2))} TWh")
            with col2:
                col2.metric("Error (MSE)", str(round(error2,2)))

    st.subheader("&#8594; Predicción de consumo energético")
    st.write("")

    if st.button('Calcular emisiones'):    
        
        X3, y_Consumption3, y_pred3, prediccion3, error3 = prediccion_emisiones_tiempo(df_Model1, anho_prediction)
        st.write("Con un modelo de regresión polinómica dependiente del año, la emisión de CO2 para el año " + str(anho_prediction) + " es " + str(round(prediccion3[0],2)) + " TnCO2 con un error de " + str(round(error3,2)))
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                col1.metric("Emisiones ", f"{str(round(prediccion3[0],2))} TnCO2")
            with col2:
                col2.metric("Error (MSE)", str(round(error3,2)))
        y_Consumption4, y_pred4, prediccion4, error4 = prediccion_emisiones_de_CO2_dependiente_tiempo_poblacion_PBI(df_Model1, anho_prediction, pais)
        st.write("Con un modelo de regresión polinómica dependiente del PBI y de la problación, la emisión de CO2 para el año " + str(anho_prediction) + " es " + str(round(prediccion4[0],2)) + " TnCO2 con un error de " + str(round(error4,2)))
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                col1.metric("Emisiones ", f"{str(round(prediccion4[0],2))} TnCO2")
            with col2:
                col2.metric("Error (MSE)", str(round(error4,2)))

    
