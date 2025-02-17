import plotly.express as px
import pymongo
import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output
import datetime
import geojson
import geopandas as gpd


#load data
client = pymongo.MongoClient("mongodb+srv://dssgpt:4yLuKY7khdfOxHes@habitacao.y9hxkyt.mongodb.net/")
database_names = client.list_database_names()
db = client["habitacao-transparente"]
collection = db['user-data']
cursor = collection.find()
documents = list(cursor)
df = pd.DataFrame(documents)
df['rendimento-anual'].drop_duplicates()
# feature engineering
df['distrito']=df['distrito'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8').map(str.title)
#df['age'] = datetime.datetime.now().year - df['ano-nascimento']
df['tipologia']=df['tipologia'].apply(lambda x: ','.join(map(str, x)))
df['situacao-habitacional']=df['situacao-habitacional'].apply(lambda x: ','.join(map(str, x)))
df['tipo-casa']=df['tipo-casa'].apply(lambda x: ','.join(map(str, x)))
df['area-util']=df['area-util']=df['area-util'].astype(str).str.replace("['","").str.replace("']","")
df['area_left']=df['area-util'].astype(str).str.extract('(\\d+)', expand=False).astype('Int32')
df['area_right']=df['area-util'].astype(str).replace('<20', '-20').replace('>400', '-400').str.extract('-(\\d+)', expand=False).astype('Int32')
df['area_mean'] = df[['area_left', 'area_right']].mean(axis=1)
df['rendimento-anual']=df['rendimento-anual'].astype(str).str.replace("['","").str.replace("']","")
df['income_sort']=df['rendimento-anual'].astype(str).str.extract('(\\d+)', expand=False).astype('Int32')
df['income_right']=df['rendimento-anual'].astype(str).replace('<7001', '-7001').replace('<7001', '-7001').replace('>80001', '-80001').str.extract('-(\\d+)', expand=False).astype('Int32')
df['income_mean'] = df[['income_sort', 'income_right']].mean(axis=1)
df['situacao-profissional']=df['situacao-profissional'].apply(lambda x: ','.join(map(str, x)))
df['educacao']=df['educacao'].astype(str).str.replace("['","").str.replace("']","")
df['satisfacao']=df['satisfacao'].apply(lambda x: ','.join(map(str, x)))
df['valorrendaarea']=(df['valor-mensal-renda']/df['area_mean']).replace(np.inf, np.nan)
df['valorcompraarea']=df['valor-compra']/df['area_mean'].replace(np.inf, np.nan)
df['percentagem-renda-paga']=df['percentagem-renda-paga'].replace("NR", None).astype(float)
df['rendimento-arrendamento']=df['rendimento-arrendamento'].astype(str).str.replace("['","").str.replace("']","")
df['rendarr_sort']=df['rendimento-arrendamento'].astype(str).str.extract('(\\d+)', expand=False).astype('Int32')
df['estrategia-arrendamento']=df['estrategia-arrendamento'].astype(str).str.replace("['","").str.replace("']","")
df['insatisfacao-motivos']=df['insatisfacao-motivos'].astype(str).str.replace("['","").str.replace("']","")
df['estado-conservacao']=df['estado-conservacao'].astype(str).str.replace("['","").str.replace("']","")
df['estrategia-compra']=df['estrategia-compra'].astype(str).str.replace("['","").str.replace("']","")
df['rendimento-liquido-anual-individual-na-compra']=df['rendimento-liquido-anual-individual-na-compra'].astype(str).str.replace("['","").str.replace("']","")
df['rendimento-liquido-anual-conjunto-na-compra']=df['rendimento-liquido-anual-conjunto-na-compra'].astype(str).str.replace("['","").str.replace("']","")
df['ano-nascimento']=df['ano-nascimento'].astype('Int64')
df['valor_renda_pago'] = df['percentagem-renda-paga']*df['valor-mensal-renda']/100  
df['valorrendaincome']=(df['valor_renda_pago']/(df['income_mean']/14)*100).replace(np.inf, np.nan)

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
df.loc[df['valorrendaarea']>50, ['valor-mensal-renda','valorrendaarea']]=None
#remove high prices
df.loc[df['valorcompraarea']>7000, 'outlier_count']+=1
df.loc[df['valorcompraarea']>7000, ['valor-compra','valorcompraarea']]=None
#remove low prices
df.loc[df['valorcompraarea']<100, 'outlier_count']+=1
df.loc[df['valorcompraarea']<100, 'valor-compra']=df['valor-compra']*100
df['valorcompraarea']=df['valor-compra']/df['area_right']
df.loc[df['valorcompraarea']<100, ['valor-compra','valorcompraarea']]=None

#with open('distrito_all.geojson', 'rb') as f:
#        gj = geojson.load(f)
gj = gpd.read_file(r'distrito_all_s.geojson')

green_hex='#32a852'
app = Dash()

app.layout = html.Div([
    html.Div([
        dcc.Checklist(
            df['distrito'].unique(),
            id='crossfilter-district1',
            labelStyle={'display': 'inline-block', 'marginTop': '5px'},
            value=df['distrito'].unique(),
            style={'accent-color': green_hex},
        ),
        dcc.Checklist(
            df['tipologia'].unique(),
            id='crossfilter-tipologia1',
            labelStyle={'display': 'inline-block', 'marginTop': '5px'},
            value=df['tipologia'].unique()
        ),
    ]),
    html.Div([
        dcc.Graph(id='interactive_price_area_year')#figure=price_area_year),
    ]),
    html.Div([
        dcc.Graph(id='interactive_rent_income')
    ]),
    html.Div([
        html.Div(children='Tipologia'),
        dcc.Dropdown(
            np.append(df['tipologia'].unique(), ['Todos']),
            'Todos',
            id='crossfilter-tipologia',
        ),
        html.Div(children='Situacao habitacional'),
        dcc.RadioItems(
            np.append(df['situacao-habitacional'].unique(), ['Todos']),
            'Todos',
            id='crossfilter-situahab',
            labelStyle={'display': 'inline-block', 'marginTop': '5px'}
        ),
        html.Div(children='Ano nascimento'),
        dcc.RangeSlider(
            min=df['ano-nascimento'].min(),
            max=df['ano-nascimento'].max(),
            value=[df['ano-nascimento'].min(),df['ano-nascimento'].max()], #default
            step=1,
            marks = {i: str(i) for i in range(df['ano-nascimento'].min(),df['ano-nascimento'].max(),5)},
            id='crossfilter-birthyear'
        ),
        html.Div(children='Métrica'),
        dcc.RadioItems(
            ['median', 'mean'],
            'median',
            id='crossfilter-xaxis-type',
            labelStyle={'display': 'inline-block', 'marginTop': '5px'}
        )
    ]),
    html.Div([
        dcc.Graph(id='interactive_map')
    ])
], style={'font-family': "Sans-serif",'title-font-family':'Helvetica', 'font-size' : 14, 'color':'#202020', 'accent-color': green_hex})


@app.callback(Output("interactive_price_area_year", "figure"), Input("crossfilter-district1", "value"),Input("crossfilter-tipologia1", "value"))
def display_color(distr, tipol):
    if distr==None:
        distr=['']
    if tipol==None:
        tipol=['']
    filtered_df=df[(df['distrito'].isin(distr))&(df['tipologia'].isin(tipol))]
    median_price_area_year=filtered_df.groupby(['ano-compra']).agg({'valorcompraarea':'median'}).reset_index().dropna(subset=['valorcompraarea'])
    price_area_year = px.bar(median_price_area_year, x='ano-compra', y='valorcompraarea', title="Preço/m² mediano por ano de compra")
    return price_area_year

@app.callback(Output("interactive_rent_income", "figure"), Input("crossfilter-district1", "value"),Input("crossfilter-tipologia1", "value"))
def display_color(distr, tipol):
    if distr==None:
        distr=['']
    if tipol==None:
        tipol=['']
    filtered_df=df[(df['distrito'].isin(distr))&(df['tipologia'].isin(tipol))]
    median_rent_income=filtered_df.groupby(['rendimento-anual']).agg({'valorrendaincome':'median'}).reset_index().dropna(subset=['valorrendaincome'])
    rent_income = px.bar(median_rent_income, x='rendimento-anual', y='valorrendaincome', title="Renda/rendimento mensal médio por intervalo de rendimento")
    rent_income.update_yaxes(ticksuffix="%")
    # make hist
    return rent_income


@app.callback(Output("interactive_map", "figure"), Input("crossfilter-tipologia", "value"),Input("crossfilter-situahab", "value"), Input("crossfilter-birthyear", "value"))
def display_color(tipol, hab, birthyear):
    if tipol=='Todos':
        if hab=='Todos':
            filtered_df=df[(df['ano-nascimento']>=birthyear[0])&(df['ano-nascimento']<=birthyear[1])]
        else:
            filtered_df=df[(df['situacao-habitacional']==hab)&(df['ano-nascimento']>=birthyear[0])&(df['ano-nascimento']<=birthyear[1])]
    else:
        if hab=='Todos':
            filtered_df=df[(df['tipologia']==tipol)&(df['ano-nascimento']>=birthyear[0])&(df['ano-nascimento']<=birthyear[1])]
        else:
            filtered_df=df[(df['situacao-habitacional']==hab)&(df['tipologia']==tipol)&(df['ano-nascimento']>=birthyear[0])&(df['ano-nascimento']<=birthyear[1])]
   
    count_district=filtered_df.groupby('distrito').agg({'email':'count'}).reset_index().rename(columns={'email':'Number of responses'})
    count_district[['distrito']]=count_district[['distrito']].map(str.title)
    map_count = px.choropleth(count_district, geojson=gj
                        , featureidkey='properties.Distrito',locations='distrito',fitbounds='locations'
                        , color='Number of responses', color_continuous_scale="algae"
                        , range_color=(count_district['Number of responses'].min(), count_district['Number of responses'].max())
                        , title="Distribuição de respostas por distrito")
    map_count=map_count.update_traces(marker_line_width=0.2)
    map_count=map_count.update_geos(fitbounds='locations', visible=False, bgcolor='white')
    #map_count.update_layout(font_family="Courier New", title_font_family="Times New Roman", font_size = 12)
    print("writing dashboard.html")
    map_count.write_html('dashboard.html')
    return map_count


app.run_server(debug=True, use_reloader=False)
