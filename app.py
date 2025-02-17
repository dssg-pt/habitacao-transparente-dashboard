import plotly.express as px
import pymongo
import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output
import datetime
import geojson
import geopandas as gpd
import streamlit as st
import ast


#load data

df = pd.read_csv('data.csv')
# Feature Engineering
df['distrito'] = df['distrito'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8').map(str.title)
#df['tipologia'] = df['tipologia'].apply(lambda x: ','.join(map(str, x)))
#df['situacao-habitacional'] = df['situacao-habitacional'].apply(lambda x: ','.join(map(str, x)))
#df['area-util'] = df['area-util'].astype(str).str.replace("['", "", regex=True).str.replace("']", "", regex=True)
df['area-util']=df['area-util']=df['area-util'].astype(str).str.replace("['","").str.replace("']","")
df['area_left'] = df['area-util'].str.extract('(\\d+)', expand=False).astype('Int32')
df['area_right'] = df['area-util'].replace('<20', '-20').replace('>400', '-400').str.extract('-(\\d+)', expand=False).astype('Int32')
df['area_mean'] = df[['area_left', 'area_right']].mean(axis=1)
df['valorcompraarea'] = df['valor-compra'] / df['area_mean']
#df['rendimento-anual'] = df['rendimento-anual'].astype(str).str.replace("['", "", regex=True).str.replace("']", "", regex=True)
df['rendimento-anual']=df['rendimento-anual'].astype(str).str.replace("['","").str.replace("']","")
df['income_sort'] = df['rendimento-anual'].str.extract('(\\d+)', expand=False).astype('Int32')
df['income_right'] = df['rendimento-anual'].replace('<7001', '-7001').replace('>80001', '-80001').str.extract('-(\\d+)', expand=False).astype('Int32')
df['income_mean'] = df[['income_sort', 'income_right']].mean(axis=1)
df['percentagem-renda-paga'] = df['percentagem-renda-paga'].replace("NR", None).astype(float)

# Load GeoJSON
gj = gpd.read_file('distrito_all_s.geojson')

# Streamlit UI
st.set_page_config(page_title="Habitacao Transparente", layout="wide")

st.title("🏠 Habitação Transparente Dashboard")

# Sidebar filters
st.sidebar.header("Filtros")

# District filter
selected_districts = st.sidebar.multiselect(
    "Selecionar Distrito",
    options=df['distrito'].unique(),
    default=df['distrito'].unique()
)

# Typology filter
selected_typologies = st.sidebar.multiselect(
    "Selecionar Tipologia",
    options=df['tipologia'].apply(ast.literal_eval).explode().unique(),
    default=df['tipologia'].apply(ast.literal_eval).explode().unique()
)

# Housing Situation filter
selected_habitation = st.sidebar.radio(
    "Situação Habitacional",
    options=np.append(df['situacao-habitacional'].apply(ast.literal_eval).explode().unique(), ['Todos']),
    index=len(df['situacao-habitacional'].unique())
)

# Year of Birth Filter
min_year = int(df['ano-nascimento'].min())
max_year = int(df['ano-nascimento'].max())
selected_year_range = st.sidebar.slider("Ano de Nascimento", min_value=min_year, max_value=max_year, value=(min_year, max_year))

# Metric Selection
selected_metric = st.sidebar.radio("Escolher Métrica", ["median", "mean"], index=0)

# Data Filtering
filtered_df = df[(df['distrito'].isin(selected_districts)) & (df['tipologia'].apply(ast.literal_eval).explode().isin(selected_typologies))]

if selected_habitation != "Todos":
    filtered_df = filtered_df[filtered_df['situacao-habitacional'].apply(ast.literal_eval).explode() == selected_habitation]

filtered_df = filtered_df[(filtered_df['ano-nascimento'] >= selected_year_range[0]) & (filtered_df['ano-nascimento'] <= selected_year_range[1])]

# --- Graphs ---
col1, col2 = st.columns(2)

# Price per Square Meter Over Time
with col1:
    st.subheader("📊 Preço/m² Mediano por Ano de Compra")
    median_price_area_year = filtered_df.groupby('ano-compra').agg({'valorcompraarea': 'median'}).reset_index()
    fig_price = px.bar(median_price_area_year, x='ano-compra', y='valorcompraarea', title="Preço/m² mediano por ano de compra")
    st.plotly_chart(fig_price, use_container_width=True)

# Rent vs Income
with col2:
    st.subheader("💰 Renda/Rendimento Mensal Médio")
    median_rent_income = filtered_df.groupby('rendimento-anual').agg({'percentagem-renda-paga': 'median'}).reset_index()
    fig_rent = px.bar(median_rent_income, x='rendimento-anual', y='percentagem-renda-paga', title="Percentagem da renda paga")
    fig_rent.update_yaxes(ticksuffix="%")
    st.plotly_chart(fig_rent, use_container_width=True)

# Choropleth Map
st.subheader("🗺 Distribuição de Respostas por Distrito")
count_district = filtered_df.groupby('distrito').agg({'email': 'count'}).reset_index().rename(columns={'email': 'Number of responses'})
count_district[['distrito']] = count_district[['distrito']].map(str.title)

fig_map = px.choropleth(
    count_district,
    geojson=gj,
    featureidkey='properties.Distrito',
    locations='distrito',
    color='Number of responses',
    color_continuous_scale="algae",
    title="Distribuição de respostas por distrito"
)

fig_map.update_geos(fitbounds="locations", visible=False, bgcolor="white")
st.plotly_chart(fig_map, use_container_width=True)

