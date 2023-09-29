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
