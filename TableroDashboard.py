# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 20:11:18 2023
#http://127.0.0.1:8050/ 
@author: juane
"""
import pandas as pd
import numpy as np
import tqdm
from ucimlrepo import fetch_ucirepo
import warnings
warnings.filterwarnings("ignore")

# fetch dataset
predict_students_dropout_and_academic_success = fetch_ucirepo(id=697)

# data (as pandas dataframes)
X = predict_students_dropout_and_academic_success.data.features
y = predict_students_dropout_and_academic_success.data.targets


#Discretizar variable objetivo
#Correlacion con variable objetivo
#Graduate = 2
#Enrolled = 1
#Dropout= 0
def ponerNumerico(row):
    if row["Target"] == 'Dropout':
        row["TargetN"] = 0

    elif row["Target"] == 'Graduate':
        row["TargetN"] = 2
    else:
        row["TargetN"] = 1
    return row

y = y.apply(ponerNumerico ,axis = 1)

#Discretizar porcentaje examenes aprobados
# 0- no presento examenes
#1- [0%-25%)
#2- 25%-50%
# 50%-75%
#75%-100%
def discretizar_porcentajeExamenesAprobados(row):
    if row['Curricular units 1st sem (evaluations)']==0:
        row["% of approved evaluations 1st sem"] = 0
    else:
        porcentaje = row['Curricular units 1st sem (approved)']/row['Curricular units 1st sem (evaluations)']
        if porcentaje >= 0.75:
            row["% of approved evaluations 1st sem"] = 4
        elif porcentaje >= 0.50:
            row["% of approved evaluations 1st sem"] = 3
        elif porcentaje >= 0.25:
            row["% of approved evaluations 1st sem"] = 2
        else:
            row["% of approved evaluations 1st sem"] = 1

    if row['Curricular units 2nd sem (evaluations)']==0:
        row["% of approved evaluations 2nd sem"] = 0
    else:
        porcentaje = row['Curricular units 2nd sem (approved)']/row['Curricular units 2nd sem (evaluations)']
        if porcentaje >= 0.75:
            row["% of approved evaluations 2nd sem"] = 4
        elif porcentaje >= 0.50:
            row["% of approved evaluations 2nd sem"] = 3
        elif porcentaje >= 0.25:
            row["% of approved evaluations 2nd sem"] = 2
        else:
            row["% of approved evaluations 2nd sem"] = 1

    return row

X = X.apply(discretizar_porcentajeExamenesAprobados ,axis = 1)

# 0- 95-120
# 1- 120-160
# 2- 160-200

def discretizar_resultados_examenes(row):
  if row["Previous qualification (grade)"]>= 160:
    row["Previous qualification (grade) performance"] = 2

  elif row["Previous qualification (grade)"]>= 120 :
    row["Previous qualification (grade) performance"] = 1

  else:
    row["Previous qualification (grade) performance"] = 0

  if row["Admission grade"]>= 160:
    row["Admission grade performance"] = 2

  elif row["Admission grade"]>= 120 :
    row["Admission grade performance"] = 1

  else:
    row["Admission grade performance"] = 0

  return row

X = X.apply(discretizar_resultados_examenes, axis = 1)

Variables = ['Marital Status', "Previous qualification", "Scholarship holder",  'Tuition fees up to date', 'Gender','% of approved evaluations 1st sem', '% of approved evaluations 2nd sem',
'Previous qualification (grade) performance', 'Admission grade performance',"Mother's occupation","Father's occupation" , 'Age at enrollment']

X_definitivo = X[Variables]
X_definitivo["Target"] = y["TargetN"]

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import scipy
from sklearn.model_selection import train_test_split
import pyparsing
import torch
import statsmodels
import tqdm
import joblib

model = BayesianNetwork ([("Age at enrollment", "Marital Status") , ("Gender", "Previous qualification"),
 ("Marital Status","Previous qualification"),("Father's occupation","Scholarship holder"),
                          ("Mother's occupation","Scholarship holder"),
                           ("Previous qualification","Tuition fees up to date"),
                          ("Tuition fees up to date","Target"),
                          ("Previous qualification","Previous qualification (grade) performance"),
                          ("Previous qualification (grade) performance","Scholarship holder"),
                           ("Previous qualification (grade) performance","Admission grade performance"),
                          ("Scholarship holder","Target"),("Admission grade performance","% of approved evaluations 1st sem"),
                           ("% of approved evaluations 1st sem","% of approved evaluations 2nd sem"),
                           ("% of approved evaluations 2nd sem","Target")])

from pgmpy.estimators import MaximumLikelihoodEstimator

emv = MaximumLikelihoodEstimator(model, data=X_definitivo)

# Estimar para nodos sin padres
cpdem_age = emv.estimate_cpd(node="Age at enrollment")

cpdem_gender = emv.estimate_cpd(node="Gender")

cpdem_father = emv.estimate_cpd(node="Father's occupation")

cpdem_mother = emv.estimate_cpd(node="Mother's occupation")


# Estimar para demas nodos
cpdem_PQ = emv.estimate_cpd(node="Previous qualification")


cpdem_scholar = emv.estimate_cpd(node = "Scholarship holder")

cpdem_TuitionDate = emv.estimate_cpd(node="Tuition fees up to date")

cpdem_gradeperformance = emv.estimate_cpd(node="Previous qualification (grade) performance")


cpdem_admision = emv.estimate_cpd(node="Admission grade performance")


cpdem_semester1 = emv.estimate_cpd(node="% of approved evaluations 1st sem")


cpdem_semester2 = emv.estimate_cpd(node="% of approved evaluations 2nd sem")

train,test = train_test_split(X_definitivo, test_size = 0.2, random_state= 42)

from pgmpy.inference import VariableElimination
model.fit( data = train , estimator = MaximumLikelihoodEstimator)
infer = VariableElimination(model)
resultados =[]

test.drop([2269,505,1532,949,3281],inplace=True,axis=0)

for index,row in test.iterrows():
    Variables = ['Gender','Age at enrollment', "Previous qualification", 'Marital Status',"Father's occupation","Mother's occupation", "Scholarship holder",  'Tuition fees up to date','Previous qualification (grade) performance','Admission grade performance', '% of approved evaluations 1st sem', '% of approved evaluations 2nd sem']
    evidencia={}
    for variable in Variables:
      evidencia[variable]= row[variable]
    inferencia = infer.query(["Target"], evidence=evidencia)
    valores = list(inferencia.values)
    target = valores.index(max(valores))
    resultados.append(target)

from sklearn.metrics import accuracy_score, confusion_matrix
accuracy = accuracy_score(test["Target"],resultados)

matriz = confusion_matrix(test["Target"],resultados)
accuracy

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px



Ocupaciones = ['0 - Student ',
 '1 - Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers ',
 '2 - Specialists in Intellectual and Scientific Activities ',
 '3 - Intermediate Level Technicians and Professions ',
 '4 - Administrative staff ',
 '5 - Personal Services, Security and Safety Workers and Sellers ',
 '6 - Farmers and Skilled Workers in Agriculture, Fisheries and Forestry ',
 '7 - Skilled Workers in Industry, Construction and Craftsmen',' 8 - Installation and Machine Operators and Assembly Workers ',
 '9 - Unskilled Workers ',
 '10 - Armed Forces Professions ',
 '90 - Other Situation 99 - (blank) ',
 '122 - Health professionals ',
 '123 - teachers ',
 '125 - Specialists in information and communication technologies (ICT) ',
 '131 - Intermediate level science and engineering technicians and professions ',
 '132 - Technicians and professionals, of intermediate level of health ',
 '134 - Intermediate level technicians from legal, social, sports, cultural and similar services ',
 '141 - Office workers, secretaries in general and data processing operators ',
 '143 - Data, accounting, statistical, financial services and registry-related operators ',
 '144 - Other administrative support staff ',
 '151 - personal service workers ',
 '152 - sellers ',
 '153 - Personal care workers and the like ',
 '171 - Skilled construction workers and the like, except electricians ',
 '173 - Skilled workers in printing, precision instrument manufacturing, jewelers, artisans and the like ',
 '175 - Workers in food processing, woodworking, clothing and other industries and crafts ',
 '191 - cleaning workers ',
 '192 - Unskilled workers in agriculture, animal production, fisheries and forestry ',
 '193 - Unskilled workers in extractive industry, construction, manufacturing and transport ',
 '194 - Meal preparation assistants']
Genero = ["1 – male"," 0 – female "]
edad =[str(i) for i in range(17,71)]
estadoCivil = ['1 – single ',
 '2 – married ',
 '3 – widower ',
 '4 – divorced ',
 '5 – facto union ',
 '6 – legally separated']
educacion = ['1 - Secondary Education - 12th Year of Schooling or Eq. ',
 "2 - Higher Education - Bachelor's Degree ",
 '3 - Higher Education - Degree ',
 "4 - Higher Education - Master's ",
 '5 - Higher Education - Doctorate ',
 '6 - Frequency of Higher Education',' 9 - 12th Year of Schooling - Not Completed ',
 '10 - 11th Year of Schooling - Not Completed ',
 '11 - 7th Year (Old) ',
 '12 - Other - 11th Year of Schooling ',
 '14 - 10th Year of Schooling ',
 '18 - General commerce course ',
 '19 - Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv. ',
 '22 - Technical-professional course ',
 '26 - 7th year of schooling ',
 '27 - 2nd cycle of the general high school course ',
 '29 - 9th Year of Schooling - Not Completed ',
 '30 - 8th year of schooling ',
 "34 - Unknown 35 - Can't read or write ",
 '36 - Can read without having a 4th year of schooling ',
 '37 - Basic education 1st cycle (4th/5th year) or equiv. ',
 '38 - Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv. ',
 '39 - Technological specialization course ',
 '40 - Higher education - degree (1st cycle) ',
 '41 - Specialized higher studies course ',
 '42 - Professional higher technical course ',
 '43 - Higher Education - Master (2nd cycle) ',
 '44 - Higher Education - Doctorate (3rd cycle)']
Beca = ["1-yes", "0-no"]
PagosMatricula = ["1-yes", "0-no"]
listaExamenes= ["0-menos de 120","1- entre 120 y 160", "2-entre 160 y 200"]
listaPorcentajeAprobados = ["0- no presento examanes","1-entre 0% y 25%", "2- entre 25% y 50%","3-entre 50% y 75%", "4-entre 75% y 100%"]
variablesFormulario = ['Gender','Age at enrollment', "Previous qualification", 'Marital Status',"Father's occupation","Mother's occupation", "Scholarship holder",  'Tuition fees up to date','Previous qualification (grade) performance','Admission grade performance', '% of approved evaluations 1st sem', '% of approved evaluations 2nd sem']
listasDesplegables= [Genero,edad,educacion,estadoCivil,Ocupaciones,Ocupaciones,Beca,PagosMatricula,listaExamenes,listaExamenes, listaPorcentajeAprobados, listaPorcentajeAprobados]
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

server

app.layout = html.Div([
    html.H1("Mida su riesgo de deserción"),
    html.H2("Por favor llene los campos solicitados a continuación para poder hacer una predicción"),

    html.Div("Seleccione su género"),
    dcc.Dropdown(
        id="gender-dropdown",
        options=[{'label': gender, 'value': gender} for gender in Genero],
        value="",
    ),

    html.Div("Seleccione su edad en el momento de su inscripción"),
    dcc.Dropdown(
        id="age-dropdown",
        options=[{'label': age, 'value': age} for age in edad],
        value="",
    ),

    html.Div("Seleccione su nivel educativo"),
    dcc.Dropdown(
        id="education-dropdown",
        options=[{'label': education, 'value': education} for education in educacion],
        value="",
    ),

    html.Div("Seleccione su estado civil"),
    dcc.Dropdown(
        id="status-dropdown",
        options=[{'label': status, 'value': status} for status in estadoCivil],
        value="",
    ),
    html.Div("Seleccione la ocupación de su padre"),
    dcc.Dropdown(
        id='occupationFather-dropdown',
        options=[{'label': occupation, 'value': occupation} for occupation in Ocupaciones],
        value="",
    ),
    html.Div("Seleccione la ocupación de su madre"),
    dcc.Dropdown(
        id='occupationMother-dropdown',
        options=[{'label': occupation, 'value': occupation} for occupation in Ocupaciones],
        value=""),
    
    html.Div("¿Se encuentra con algún apoyo económico?"),
    dcc.Dropdown(
        id="scholarship holder-dropdown",
        options=[{'label': Scholarship_holder, 'value': Scholarship_holder} for Scholarship_holder in Beca],
        value="",
    ),

    html.Div("¿Tiene registrado el pago de su matrícula?"),
    dcc.Dropdown(
        id="pagos-dropdown",
        options=[{'label': Pagos, 'value': Pagos} for Pagos in PagosMatricula],
        value="",
    ),

    html.Div("Ingrese su calificación obtenida en su institución educativa previa (Número entre 0 y 200):"),
    dcc.Input(
        id='prev-quali-grade',
        type='number',  # Configura el tipo de entrada como numérico
        value=0,        # Valor inicial
        step=1          # Incremento/decremento en cada cambio
    ),

    html.Div("Ingrese su calificación obtenida en el examen de admisión (Número entre 0 y 200):"),
    dcc.Input(
        id='admission-grade',
        type='number',  # Configura el tipo de entrada como numérico
        value=0,        # Valor inicial
        step=1          # Incremento/decremento en cada cambio
    ),

    html.Div("Ingrese el número de exámenes presentados en su primer semestre:"),
    dcc.Input(
        id='exams-taken-1st',
        type='number',  # Configura el tipo de entrada como numérico
        value=0,        # Valor inicial
        step=1          # Incremento/decremento en cada cambio
    ),

    html.Div("Ingrese el número de exámenes aprobados en su primer semestre:"),
    dcc.Input(
        id='exams-approved-1st',
        type='number',  # Configura el tipo de entrada como numérico
        value=0,        # Valor inicial
        step=1          # Incremento/decremento en cada cambio
    ),

    html.Div("Ingrese el número de exámenes presentados en su segundo semestre:"),
    dcc.Input(
        id='exams-taken-2nd',
        type='number',  # Configura el tipo de entrada como numérico
        value=0,        # Valor inicial
        step=1          # Incremento/decremento en cada cambio
    ),

    html.Div("Ingrese el número de exámenes aprobados en su segundo semestre:"),
    dcc.Input(
        id='exams-approved-2nd',
        type='number',  # Configura el tipo de entrada como numérico
        value=0,        # Valor inicial
        step=1          # Incremento/decremento en cada cambio
    ),
    html.H2("Segun sus datos, su prediccion es la siguiente: "),
    dcc.Textarea(
        id='prediction',
        value='La prediccion es: ',
        style={'width': '50%', 'height': 50, "fontsize":"40px"},
        disabled= True
    ),
    html.H2("Comparese con el resto de estudiantes segun una categoria de su eleccion: "),
    dcc.Dropdown(
        id="categoria-dropdown",
        options=[{'label': categoria, 'value': categoria} for categoria in variablesFormulario],
        value="",
    ),
    html.Div("Escoja algun rubro de la categoria seleccionada: "),
    dcc.Dropdown(
        id="rubrocategoria-dropdown",
        value="",
    ),
    dcc.Graph(id='graph')
    
],   style={'backgroundColor': '#f2f2f2'})

@app.callback(
     Output('prediction', 'value'),
     [Input('gender-dropdown', 'value'),
     Input('age-dropdown','value'),
     Input('education-dropdown', 'value'),
     Input('status-dropdown', 'value'),
     Input('occupationFather-dropdown', 'value'),
     Input('occupationMother-dropdown', 'value'),
     Input('scholarship holder-dropdown', 'value'),
     Input('pagos-dropdown','value'),
     Input('prev-quali-grade', 'value'),
     Input('admission-grade', 'value'),
     Input('exams-taken-1st', 'value'),
     Input('exams-approved-1st', 'value'),
     Input('exams-taken-2nd', 'value'),
     Input('exams-approved-2nd', 'value')
    ])




def prediccion(gender,age,education,status,occupationFather,occupationMother,scholarship,pagos,educationgrade,admissiongrade,examstaken1st,examsapproved1st,examstaken2nd,examsapproved2nd):
  
  if not all([gender, age, education, status, occupationFather, occupationMother, scholarship, pagos, educationgrade, admissiongrade, examstaken1st, examsapproved1st, examstaken2nd, examsapproved2nd]):
      return None
  else :
      
      discretas = [gender,age,education,status,occupationFather,occupationMother,scholarship,pagos]
      variables = ['Gender','Age at enrollment', "Previous qualification", 'Marital Status',"Father's occupation","Mother's occupation", "Scholarship holder",  'Tuition fees up to date','Previous qualification (grade) performance','Admission grade performance', '% of approved evaluations 1st sem', '% of approved evaluations 2nd sem']
      evidencia = {}
    
      for i in range(len(discretas)):
        if i != 1:
          if str(discretas[i][1]) in [" ","-"]:
            evidencia[variables[i]]=int(discretas[i][0])
          else:
            if str(discretas[i][2])in [" ","-"]:
              evidencia[variables[i]]=int(discretas[i][0:2])
            else:
              evidencia[variables[i]]=int(discretas[i][0:3])
        else:
          evidencia[variables[i]]=int(discretas[i])
    
      if int(educationgrade) >= 160:
          evidencia[variables[8]] = 2
      elif int(educationgrade) >= 120 :
          evidencia[variables[8]] = 1
      else:
          evidencia[variables[9]]= 0
    
      if int(admissiongrade) >= 160:
          evidencia[variables[9]] = 2
      elif int(admissiongrade) >= 120 :
          evidencia[variables[9]] = 1
      else:
          evidencia[variables[9]]= 0
    
      if int(examstaken1st) != 0:
          porcentaje1st= int(examsapproved1st)/int(examstaken1st)
          if porcentaje1st >= 0.75:
              evidencia[variables[10]] = 4
          elif porcentaje1st >= 0.50:
              evidencia[variables[10]] = 3
          elif porcentaje1st >= 0.25:
              evidencia[variables[10]] = 2
          else:
              evidencia[variables[10]]= 1
      else:
          evidencia[variables[10]]= 0
    
      if int(examstaken2nd) != 0:
          porcentaje2nd= int(examsapproved2nd)/int(examstaken2nd)
          if porcentaje2nd >= 0.75:
              evidencia[variables[11]] = 4
          elif porcentaje2nd >= 0.50:
              evidencia[variables[11]] = 3
          elif porcentaje2nd >= 0.25:
              evidencia[variables[11]] = 2
          else:
              evidencia[variables[11]]= 1
      else:
          evidencia[variables[11]]= 0
    
      inferencia = infer.query(["Target"], evidence=evidencia)
      valores = list(inferencia.values)
      target = valores.index(max(valores))
    
      if target == 0:
        value = "Desertor"
      elif target == 1:
        value = "Matriculado"
      else:
        value=  "Graduado"
      return value

@app.callback(
     Output("rubrocategoria-dropdown", 'options'),
     Input("categoria-dropdown","value")
    )

def generarListaDesplegable(categoria):
    if not categoria:
        return dash.no_update
    variables = ['Gender','Age at enrollment', "Previous qualification", 'Marital Status',"Father's occupation","Mother's occupation", "Scholarship holder",  'Tuition fees up to date','Previous qualification (grade) performance','Admission grade performance', '% of approved evaluations 1st sem', '% of approved evaluations 2nd sem']
    posicion = variables.index(categoria)
    lista = listasDesplegables[posicion]
    opciones_actualizadas = [{'label': opcion, 'value': opcion} for opcion in lista]
    return opciones_actualizadas 
    
        
@app.callback(Output("graph", 'figure'),
[Input("categoria-dropdown", "value"),
Input("rubrocategoria-dropdown","value")
    ])

def grafica(categoria,rubrocategoria):
  if categoria is None or rubrocategoria is None or categoria == "" or rubrocategoria == "":
        # Mostrar un mensaje de error en el gráfico
        empty_data = pd.DataFrame({'x': [0], 'y': [0], 'text': ['Seleccione valores válidos en los dropdowns']})
        fig = px.scatter(empty_data, x='x', y='y', text='text')
        return fig
    
  if categoria != "Age at enrollment":
      if str(rubrocategoria[1]) in [" ","-"]:
            rubro=float(rubrocategoria[0])
      else:
          if str(rubrocategoria[2])in [" ","-"]:
              rubro=float(rubrocategoria[0:2])
          else:
              rubro=float(rubrocategoria[0:3])
  else:
       rubro = float(rubrocategoria)
          
  X_filtrado = X_definitivo[X_definitivo[categoria]==rubro]["Target"]
        
  titulo = "Distribucion de graduados, matriculados y desertores según la categoria: " + str(categoria) + ": " + str(rubrocategoria)
  frecuencias =X_filtrado.value_counts().reset_index()
  frecuencias.columns = ['Categoria', 'Frecuencia']
  for index,row in frecuencias.iterrows():
      if row["Categoria"] == 0:
          frecuencias.at[index,"Categoria"] = "Desertor"
      elif row["Categoria"] == 1:
          frecuencias.at[index,"Categoria"] = "Matriculado"
      else:
          frecuencias.at[index,"Categoria"] = "Graduado"
          
        # Crea el gráfico de pastel con Plotly Express
  fig = px.pie(frecuencias, names='Categoria', values='Frecuencia', title=titulo)
        
        # Ajusta la duración de la transición (opcional)
  fig.update_layout(transition_duration=1200)
  return fig

if __name__ == '__main__':
    app.run_server(debug=True)
    
