import plotly.express as px
import pymongo
import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output
from io import StringIO
import datetime
import geojson
import geopandas as gpd
from plotly.subplots import make_subplots
import streamlit as st
import ast
import requests


# define parameters for a request
token = st.secrets.token
owner = st.secrets.owner
repo = st.secrets.repo
path = st.secrets.path

# send a request
r = requests.get(
    'https://api.github.com/repos/{owner}/{repo}/contents/{path}'.format(
    owner=owner, repo=repo, path=path),
    headers={
        'accept': 'application/vnd.github.v3.raw',
        'authorization': 'token {}'.format(token)
            }
    )

# convert string to StringIO object
string_io_obj = StringIO(r.text)

# Load data to df
df = pd.read_csv(string_io_obj, sep=",", index_col=0)
# Feature Engineering
df['distrito_og']=df['distrito'].map(str.title)
df['distrito'] = df['distrito'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8').map(str.title)
df['area-util']=df['area-util']=df['area-util'].astype(str).str.replace("['","").str.replace("']","")
df['area_left'] = df['area-util'].str.extract('(\\d+)', expand=False).astype('Int32')
df['area_right'] = df['area-util'].replace('<20', '-20').replace('>400', '-400').str.extract('-(\\d+)', expand=False).astype('Int32')
df['area_mean'] = df[['area_left', 'area_right']].mean(axis=1)
df['valorcompraarea'] = df['valor-compra'] / df['area_mean']
df['rendimento-anual']=df['rendimento-anual'].astype(str).str.replace("['","").str.replace("']","")
df['income_left']=df['rendimento-anual'].astype(str).str.extract('(\d+)', expand=False).astype('Int32')
df['income_right']=df['rendimento-anual'].astype(str).replace('<7001', '-7001').replace('<7001', '-7001').replace('>80001', '-80001').str.extract('-(\d+)', expand=False).astype('Int32')
df['income_mean'] = df[['income_left', 'income_right']].mean(axis=1)
df['educacao']=df['educacao'].astype(str).str.replace("['","").str.replace("']","")
df['percentagem-renda-paga']=df['percentagem-renda-paga'].replace("NR", None).astype(float)
df['valor_renda_total'] = df['valor-mensal-renda']/(df['percentagem-renda-paga']/100)
df['valorrendaarea']=(df['valor_renda_total']/df['area_mean']).replace(np.inf, np.nan)
df['valorrendaarea_left']=df['valor_renda_total']/df['area_left'].replace(np.inf, np.nan)
df['valorrendaarea_right']=df['valor_renda_total']/df['area_right'].replace(np.inf, np.nan)
df['valorcompraarea']=df['valor-compra']/df['area_mean'].replace(np.inf, np.nan)
df['valorcompraarea_left']=df['valor-compra']/df['area_left'].replace(np.inf, np.nan)
df['valorcompraarea_right']=df['valor-compra']/df['area_right'].replace(np.inf, np.nan)
df['valorcompraarea_int']='[' + df['valorcompraarea_right'].round(2).astype(str) + ' - ' + df['valorcompraarea_left'].round(2).astype(str) + ']'
#df['valorcompraarea_int']=pd.arrays.IntervalArray.from_arrays(df['valorcompraarea_right'],df['valorcompraarea_left'])
df['rendimento-arrendamento']=df['rendimento-arrendamento'].astype(str).str.replace("['","").str.replace("']","")
df['rendarr_sort']=df['rendimento-arrendamento'].astype(str).str.extract('(\d+)', expand=False).astype('Int32')
df['estrategia-arrendamento']=df['estrategia-arrendamento'].astype(str).str.replace("['","").str.replace("']","")
df['insatisfacao-motivos']=df['insatisfacao-motivos'].astype(str).str.replace("['","").str.replace("']","")
df['estado-conservacao']=df['estado-conservacao'].astype(str).str.replace("['","").str.replace("']","")
df['estrategia-compra']=df['estrategia-compra'].astype(str).str.replace("['","").str.replace("']","")
df['rendimento-liquido-anual-individual-na-compra']=df['rendimento-liquido-anual-individual-na-compra'].astype(str).str.replace("['","").str.replace("']","")
df['rendimento-liquido-anual-conjunto-na-compra']=df['rendimento-liquido-anual-conjunto-na-compra'].astype(str).str.replace("['","").str.replace("']","")
#df['ano-nascimento']=df['ano-nascimento'].astype('Int64')
df['ano-compra']=df['ano-compra'].astype('Int64')
df['valorcompraarea_eleft']=df['valorcompraarea']-df['valorcompraarea_right']
df['valorcompraarea_eright']=df['valorcompraarea_left']-df['valorcompraarea']
df['valorrendaarea_eleft']=df['valorrendaarea']-df['valorrendaarea_right']
df['valorrendaarea_eright']=df['valorrendaarea_left']-df['valorrendaarea']
df['valorrendaincome']=(df['valor-mensal-renda']/(df['income_mean']/14)*100).replace(np.inf, np.nan) 
df['valorrendaincome_left']=(df['valor-mensal-renda']/(df['income_left']/14)*100).replace(np.inf, np.nan)
df['valorrendaincome_right']=(df['valor-mensal-renda']/(df['income_right']/14)*100).replace(np.inf, np.nan)
df['valorrendaincome_eleft']=df['valorrendaincome']-df['valorrendaincome_right']
df['valorrendaincome_eright']=df['valorrendaincome_left']-df['valorrendaincome']

#remove outliers
df['outlier_count']=0
#remove ages <18
df['age_submission']=pd.to_datetime(df['insertedAt']).dt.year - df['ano-nascimento'] + 1
df.loc[df['age_submission']<18, 'outlier_count']+=1
df.loc[df['age_submission']<18, ['ano-nascimento','age_submission']]=None
#remove dependants + non dependants >20
df.loc[(df['num-pessoas-nao-dependentes']+df['num-pessoas-dependentes'])>20,'outlier_count']+=1
df.loc[(df['num-pessoas-nao-dependentes']+df['num-pessoas-dependentes'])>20,['num-pessoas-nao-dependentes','num-pessoas-dependentes']]=None
#remove high rents
df.loc[df['valorrendaarea']>50, 'outlier_count']+=1
df.loc[df['valorrendaarea']>50, ['valor_renda_total','valor-mensal-renda','valorrendaarea']]=None
#remove high prices
df.loc[df['valorcompraarea']>7000, 'outlier_count']+=1
df.loc[df['valorcompraarea']>7000, ['valor-compra','valorcompraarea']]=None
#remove low prices
df.loc[df['valorcompraarea']<100, 'outlier_count']+=1
df.loc[df['valorcompraarea']<100, 'valor-compra']=df['valor-compra']*100
#df['valorcompraarea']=df['valor-compra']/df['area_right']
df.loc[df['valorcompraarea']<100, ['valor-compra','valorcompraarea', 'valorcompraarea_int']]=None

#remove rows with over 3 outlier columns
df = df[df['outlier_count'] <= 3]

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

# Data Filtering
filtered_df = df[(df['distrito'].isin(selected_districts)) & (df['tipologia'].apply(ast.literal_eval).explode().isin(selected_typologies))]

if selected_habitation != "Todos":
    filtered_df = filtered_df[filtered_df['situacao-habitacional'].apply(ast.literal_eval).explode() == selected_habitation]

filtered_df = filtered_df[(filtered_df['ano-nascimento'] >= selected_year_range[0]) & (filtered_df['ano-nascimento'] <= selected_year_range[1])]

# --- Graphs ---
# --- Graphs ---

# Choropleth Map
st.subheader("🗺 Distribuição de Respostas por Distrito")
count_district = filtered_df.groupby('distrito').agg({'email': 'count'}).reset_index().rename(columns={'email': 'Number of responses'})
count_district = count_district.merge(gj[['Distrito', 'zona']], left_on='distrito', right_on='Distrito', how='outer').fillna(0)
distrito_mapping = df[['distrito', 'distrito_og']].drop_duplicates().set_index('distrito')['distrito_og'].to_dict()
count_district['distrito_og'] = count_district['distrito'].map(distrito_mapping)

mainland_df = count_district[count_district['zona'] == 'continente']
madeira_df = count_district[count_district['zona'] == 'madeira']
acores_df = count_district[count_district['zona'] == 'acores']

map_count_continente = px.choropleth(
    mainland_df, 
    geojson=gj, 
    featureidkey='properties.Distrito', 
    locations='distrito', 
    color='Number of responses', 
    hover_name='distrito_og',
    hover_data={'distrito': False},
    color_continuous_scale="algae", 
    range_color=(count_district['Number of responses'].min(), count_district['Number of responses'].max())
)
map_count_madeira = px.choropleth(
    madeira_df, 
    geojson=gj, 
    featureidkey='properties.Distrito', 
    locations='distrito', 
    color='Number of responses', 
    hover_name='distrito_og',
    hover_data={'distrito': False},
    color_continuous_scale="algae", 
    range_color=(count_district['Number of responses'].min(), count_district['Number of responses'].max()))
map_count_acores = px.choropleth(
    acores_df, 
    geojson=gj, 
    featureidkey='properties.Distrito', 
    locations='distrito', 
    color='Number of responses', 
    hover_name='distrito_og',
    hover_data={'distrito': False},
    color_continuous_scale="algae", 
    range_color=(count_district['Number of responses'].min(), count_district['Number of responses'].max()))

fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "choropleth"}, {"type": "choropleth", "rowspan": 2}],
               [{"type": "choropleth"}, None]],
        subplot_titles=("Açores", "Continente", "Madeira")
    )
for trace in map_count_continente.data:
    fig.add_trace(trace, row=1, col=2)
for trace in map_count_madeira.data:
    fig.add_trace(trace, row=2, col=1)
for trace in map_count_acores.data:
    fig.add_trace(trace, row=1, col=1)

fig.update_layout(showlegend=False, coloraxis_colorbar=dict(title="Nº de respostas"), coloraxis_colorscale="algae")
fig.update_geos(fitbounds="locations", visible=False, bgcolor='white')
fig.update_traces(marker_line_width=0.5, selector=dict(type='choropleth'), marker_line_color="White")
st.plotly_chart(fig, use_container_width=True)


# Demographics
st.subheader("📖 Algumas características adicionais")
count_situacao=df.groupby('educacao').agg({'email':'count'}).reset_index().rename(columns={'email':'Nº de submissões'})
pie_educ = px.pie(count_situacao, values='Nº de submissões', names='educacao'
                      , title="Educação")
pie_educ.update_traces(textposition='inside', textinfo='percent+label', insidetextorientation='radial')
pie_educ.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', showlegend=False)

hist_age = px.histogram(df, x='age_submission', color_discrete_sequence=['#50C878'], title="Distribuição de respostas por idade à data da submissão")
hist_age.update_xaxes(title_text="Idade à data da submissão")
hist_age.update_yaxes(title_text="Nº de submissões")

agg_satisf_situacao = df.groupby(['situacao-habitacional', 'satisfacao']).agg({'email':'count'}).reset_index().rename(columns={'email':'Nº de submissões'})
agg_satisf_situacao['situacao-habitacional'] = agg_satisf_situacao['situacao-habitacional'].apply(ast.literal_eval).explode()
agg_satisf_situacao['satisfacao'] = agg_satisf_situacao['satisfacao'].apply(ast.literal_eval).explode()
agg_satisf_situacao['percentage'] = agg_satisf_situacao.groupby('situacao-habitacional')['Nº de submissões'].apply(lambda x: x / x.sum()).reset_index()['Nº de submissões']
agg_satisf_situacao['percentage'] = agg_satisf_situacao['percentage'].map("{:.2%}".format)
agg_satisf_situacao['satisfacao'] = pd.Categorical(agg_satisf_situacao['satisfacao'], categories=['muito-insatisfeito', 'insatisfeito', 'indiferente', 'satisfeito', 'muito-satisfeito'], ordered=True)
agg_satisf_situacao = agg_satisf_situacao.sort_values(by=['situacao-habitacional', 'satisfacao'])
bar_satisf = px.bar(agg_satisf_situacao, x='situacao-habitacional', y='Nº de submissões'
                    , color='satisfacao', barmode='group', text='percentage'
                    , title="Nível de satisfação por situação habitacional")
bar_satisf.update_xaxes(title_text="Situação Habitacional")

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(pie_educ, use_container_width=True)
with col2:
    st.plotly_chart(hist_age, use_container_width=True)

st.plotly_chart(bar_satisf, use_container_width=True)


median_price_area_year = filtered_df.groupby(['ano-compra']).agg({'valorcompraarea': [np.mean, np.std], '_id': 'count'}).reset_index()#.dropna(subset=['valorcompraarea'])
median_price_area_year.columns = ['ano-compra', 'valorcompraarea', 'valorcompraarea_std', '_id']
median_price_area_year['valorcompraarea_eright'] = median_price_area_year['valorcompraarea_std'] / np.sqrt(median_price_area_year['_id']) * 1.96
median_price_area_year['valorcompraarea_eleft'] = median_price_area_year['valorcompraarea_std'] / np.sqrt(median_price_area_year['_id']) * 1.96
median_price_area_year = median_price_area_year.dropna(subset=['valorcompraarea_std'])
price_area_year = px.bar(median_price_area_year, x='ano-compra', y='valorcompraarea'
                            , error_y='valorcompraarea_eright', error_y_minus='valorcompraarea_eleft'
                            , title="Preço/m² médio por ano de compra")
price_area_year.update_xaxes(title_text="Ano de Compra")
price_area_year.update_yaxes(title_text="Preço/m² Médio")

median_price_area_year = filtered_df.groupby(['ano-inicio-arrendamento']).agg({'valorrendaarea': [np.mean, np.std], '_id': 'count'}).reset_index()
median_price_area_year.columns = ['ano-inicio-arrendamento', 'valorrendaarea', 'valorrendaarea_std', '_id']
median_price_area_year['valorrendaarea_eright'] = median_price_area_year['valorrendaarea_std'] / np.sqrt(median_price_area_year['_id']) * 1.96
median_price_area_year['valorrendaarea_eleft'] = median_price_area_year['valorrendaarea_std'] / np.sqrt(median_price_area_year['_id']) * 1.96
median_price_area_year = median_price_area_year.dropna(subset=['valorrendaarea_std'])
rent_area_year = px.bar(median_price_area_year, x='ano-inicio-arrendamento', y='valorrendaarea'
                            , error_y='valorrendaarea_eright', error_y_minus='valorrendaarea_eleft'
                            , title="Renda/m² médio por ano de início de arrendamento")
rent_area_year.update_xaxes(title_text="Ano de Início de Arrendamento")
rent_area_year.update_yaxes(title_text="Renda/m² Média")

colprice, colrent = st.columns(2)

with colprice:
    st.subheader("🏠 Preço/m² médio por Ano de Compra")
    st.plotly_chart(price_area_year, use_container_width=True)
with colrent:
    st.subheader("💰 Renda/m² médio por Ano de Compra")
    st.plotly_chart(rent_area_year, use_container_width=True)

st.caption("Médias e intervalos de confiança a 95% calculados assumindo o valor médio do intervalo da área da habitação")

median_rent_income=filtered_df.groupby(['rendimento-anual']).agg({'valorrendaincome': [np.mean, np.std], '_id': 'count'}).reset_index()
median_rent_income['rendimento-anual'] = pd.Categorical(median_rent_income['rendimento-anual'], categories=[
    '<7001', '7001-12000', '12001-20000', '20001-35000', '35001-50000', '50001-80000', '>80001'
], ordered=True)
median_rent_income = median_rent_income.sort_values('rendimento-anual')
median_rent_income.columns = ['rendimento-anual', 'valorrendaincome', 'valorrendaincome_std', '_id']
median_rent_income['valorrendaincome_eright'] = median_rent_income['valorrendaincome_std'] / np.sqrt(median_rent_income['_id']) * 1.96
median_rent_income['valorrendaincome_eleft'] = median_rent_income['valorrendaincome_std'] / np.sqrt(median_rent_income['_id']) * 1.96
median_rent_income = median_rent_income.dropna(subset=['valorrendaincome_std'])
rent_income = px.bar(median_rent_income, x='rendimento-anual', y='valorrendaincome'
                        , error_y='valorrendaincome_eright', error_y_minus='valorrendaincome_eleft')
rent_income.update_xaxes(title_text="Rendimento Anual")
rent_income.update_yaxes(title_text=r"Renda/rendimento médio", ticksuffix="%")

st.subheader(r"💸 \% do rendimento alocado à renda")
st.plotly_chart(rent_income, use_container_width=True)
st.caption("Médias e intervalos de confiança a 95/% calculados assumindo o valor médio do intervalo de rendimento")
