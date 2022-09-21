import streamlit as st
import ann_page as dl
import regpol_page as reg
# Cargar archivos
logs, countries, forecast = dl.load_datasets()

# Bara lateral izq. (conteniendo el selectbox)
st.sidebar.title("**PROYECTO GRUPAL G9**")
st.sidebar.markdown('*Abel Alejandro Yepez Giraldo*')
st.sidebar.markdown('*Edgar Alejandro Hernández León*')
st.sidebar.markdown('*Esteffany Huamanraime Maquin*')
st.sidebar.markdown('*Sandro Palacios Antivo*')
page = st.sidebar.selectbox("Seleccione el objetivo", ("Modelo de regresión polinómica", "Modelo de Red Neuronal"))

if page == "Modelo de regresión polinómica":
    reg.show_Modelo_Regresión_Polinómica()
    
elif page == "Modelo de Red Neuronal":
    dl.show_forecast_page(logs, countries, forecast)

