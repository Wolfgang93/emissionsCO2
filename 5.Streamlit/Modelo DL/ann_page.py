import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cols = ['year', 'co2PerCapita', 'gdp', 'population', 'energyConsumption', 'co2emissions']
features = ['co2PerCapita', 'gdp', 'population', 'energyConsumption']

def load_datasets():
    countries = pd.read_csv('files/countries_dl.csv')
    logs = pd.read_csv('files/logs_dl.csv')
    forecast = pd.read_csv('files/forecast_emissions.csv')

    return logs, countries, forecast



def pronosticar(proyeccion, pais, horizonte):
    year_future = np.arange(2021, horizonte + 1).tolist()
    future = proyeccion[proyeccion['country'] == pais]
    future = future[future['year'] <= horizonte]['co2emissions'].values.tolist()
    return future, year_future


def plot_forecast(pais_plot, pred_real, year_future, pais):
    year_init = pais_plot.year.values.tolist()
    emission_init = pais_plot.co2emissions.values.tolist()
    
    fig, ax = plt.subplots()
    ax.plot(year_init, emission_init)
    ax.plot(year_init[-1:] + year_future, emission_init[-1:] + pred_real, c = 'g', linestyle = '--')
    ax.scatter(year_future, pred_real, c = 'r')
    ax.set_xlabel('Año')
    ax.set_ylabel('Emisiones CO2 (Tn)')
    ax.set_title(f"Pronóstico hasta el 2030 de {pais}")
    st.pyplot(fig)
   
   


def show_forecast_page(logs, paises, proyeccion):
    st.markdown("<h1 style='text-align: center; color: navy;'>Pronóstico de emisiones CO2</h1>", unsafe_allow_html=True)
    countries = paises['country'].values.tolist()
    pais = st.selectbox("País", [''] + countries)
    horizonte = st.slider("Año del horizonte a pronosticar", 2021, 2030, 2022)

    ok = st.button("Calcular emisión CO2")

    if ok:
        pronostico, years = pronosticar(proyeccion, pais, horizonte)
        pais_plot = logs[logs['country'] == pais][cols]
        plot_forecast(pais_plot, pronostico, years, pais)

