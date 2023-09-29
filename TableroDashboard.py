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
