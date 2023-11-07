import pandas as pd
import numpy as np
import pickle
import psycopg2
from ucimlrepo import fetch_ucirepo
import pickle

# fetch dataset
predict_students_dropout_and_academic_success = fetch_ucirepo(id=697)

# data (as pandas dataframes)
X = predict_students_dropout_and_academic_success.data.features
y = predict_students_dropout_and_academic_success.data.targets

# metadata
print(predict_students_dropout_and_academic_success.metadata)

# variable information
print(predict_students_dropout_and_academic_success.variables)

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

def discretizar_nota_semestre(row):
  if row['Curricular units 1st sem (grade)'] >= X['Curricular units 1st sem (grade)'].quantile(0.75):
    row["Grade sem1"] = 4
  elif row['Curricular units 1st sem (grade)'] >= X['Curricular units 1st sem (grade)'].quantile(0.50):
    row["Grade sem1"] = 3
  elif row['Curricular units 1st sem (grade)'] >= X['Curricular units 1st sem (grade)'].quantile(0.25):
    row["Grade sem1"] = 2
  else:
    row["Grade sem1"] = 1

  if row['Curricular units 2nd sem (grade)'] >= X['Curricular units 2nd sem (grade)'].quantile(0.75):
    row["Grade sem2"] = 4
  elif row['Curricular units 2nd sem (grade)'] >= X['Curricular units 2nd sem (grade)'].quantile(0.50):
    row["Grade sem2"] = 3
  elif row['Curricular units 2nd sem (grade)'] >= X['Curricular units 2nd sem (grade)'].quantile(0.25):
    row["Grade sem2"] = 2
  else:
    row["Grade sem2"] = 1

  return row

X = X.apply(discretizar_nota_semestre,axis = 1)

Variables = ['Marital Status', "Previous qualification", "Scholarship holder",  'Tuition fees up to date', 'Gender',
'Previous qualification (grade) performance', 'Admission grade performance',"Mother's occupation","Father's occupation" , 'Age at enrollment',"Grade sem1","Grade sem2"]

df = X[Variables]
df['Target'] = y['TargetN']


#Metodo Hill Clim por Bic
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import BicScore
from pgmpy.estimators import K2Score
from pgmpy.models import BayesianNetwork


#Metodo Hill Clim por Bic
from pgmpy.estimators import BicScore
from pgmpy.inference import VariableElimination

scoring_method = BicScore(data=df)
esth = HillClimbSearch(data=df)
estimated_modelHC2 = esth.estimate(
    scoring_method=scoring_method, max_indegree=4, max_iter=int(1e4)
)
print(estimated_modelHC2)
print(estimated_modelHC2.nodes())
print(estimated_modelHC2.edges())


from pgmpy.estimators import MaximumLikelihoodEstimator
estimated_modelHC2 = BayesianNetwork(estimated_modelHC2.edges())

estimated_modelHC2.fit( data = df , estimator = MaximumLikelihoodEstimator)
infer_HC2 = VariableElimination(estimated_modelHC2)


# Ruta completa o relativa al directorio donde deseas guardar el archivo de pickle
filename = "C:/Users/LENOVO/temp/Proyecto2/estudiantes.pkl"

with open(filename,'wb') as file:
    pickle.dump(estimated_modelHC2 , file)
    file.close()