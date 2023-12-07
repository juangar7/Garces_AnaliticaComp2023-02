# ANALISIS EXPLORATORIO 
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("C:/Users/juane/AnaliticaComputacional/Proyecto3/datosValle4000.csv")


#Limpiar base de datos--------------------------------------------------------------------------------------------------------------------
df.dropna(inplace = True)
df = df[df["periodo"] >= 20221]
df= df[df['fami_estratovivienda'].str.contains("Estrato")]
df.drop(['cole_depto_ubicacion','estu_depto_presentacion' ], axis = 1, inplace= True)
df.sort_values(by = ["fami_estratovivienda","desemp_ingles"], ascending = True, inplace = True)

cuantil1 = df["punt_global"].quantile(0.4)
cuantil2 = df["punt_global"].quantile(0.8)

#Grupo1 0-212
#Grupo2 212-
#Grupo4 287-337
#Grupo5 337-

dfN = pd.DataFrame()

etiquetas= []
for columna in df.columns[0:-1]:
    if columna == "periodo":
       dfN[columna]= df[columna]
    else:
        dfN[columna] = pd.factorize(df[columna])[0]
    
    valor = 0
    dicc = {}
    for categoria in df[columna].unique():
        dicc[categoria]= valor
        valor +=1 
    etiquetas.append(dicc)
        
def ponerNumerico(row):
    if row["punt_global"] >= cuantil2:
        row["TargetN"] = 2
    elif row["punt_global"] >= cuantil1:
        row["TargetN"] = 1
    else :
          row["TargetN"] = 0
    return row

df = df.apply(ponerNumerico,axis = 1)

dfN["Target"] = df["TargetN"]

#Aprendizaje de estructura------------------------------------------------------------------------------------------

from pgmpy.estimators import PC
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import K2Score
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.estimators import BicScore
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import scipy
from sklearn.model_selection import train_test_split
import pyparsing
import torch
import statsmodels
import tqdm
import joblib
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import ConfusionMatrixDisplay
from pgmpy.estimators import ParameterEstimator
import pickle

dfNdef = dfN[['cole_area_ubicacion', 'cole_bilingue', 'cole_caracter',
       'cole_calendario', 'cole_genero', 'cole_naturaleza',
       'estu_genero', 'fami_estratovivienda', 'fami_tienecomputador',
       'fami_tieneinternet', 'desemp_ingles','Target']]

train,test = train_test_split(dfNdef, test_size = 0.2, random_state= 42, stratify = dfNdef["Target"])


#Algortimo PC------------------------------------------------------------------------------------

est = PC(data=train)
estimated_modelPC = est.estimate(variant="stable", max_cond_vars=4)

print(estimated_modelPC)
print(estimated_modelPC.nodes())
print(estimated_modelPC.edges())



estimated_modelPC = BayesianNetwork(estimated_modelPC)
estimated_modelPC.fit( data = train , estimator = MaximumLikelihoodEstimator)

bic_score = BicScore(train)
bic_value = bic_score.score(estimated_modelPC)

infer_PC = VariableElimination(estimated_modelPC)


#Inferencia PC
resultadosPC = []
probasPC = []

for index,row in test.iterrows():
    variables = []
    for variable in  list(estimated_modelPC.nodes()):
        if variable != 'Target':
           variables.append(variable)
    evidencia={}
    for variable in variables:
      evidencia[variable]= row[variable]
    inferencia = infer_PC.query(["Target"], evidence=evidencia)
    valores = list(inferencia.values)
    target = valores.index(max(valores))
    proba= max(valores)
    resultadosPC.append(target)
    probasPC.append(proba)

# METRICAS PC
import matplotlib.pyplot as plt


accuracyPC = accuracy_score(test["Target"],resultadosPC)
matriz = confusion_matrix(test["Target"],resultadosPC)
# Calcular el recall para cada clase individualmente
recall_clase_0 = recall_score(test["Target"],resultadosPC ,labels=[0], average='micro')
recall_clase_1 = recall_score(test["Target"],resultadosPC , labels=[1], average='micro')
recall_clase_2 = recall_score(test["Target"],resultadosPC , labels=[2], average='micro')
disp = ConfusionMatrixDisplay(confusion_matrix=matriz)
disp.plot()
plt.title("Matriz de PC")
plt.show()

# Ruta completa o relativa al directorio donde deseas guardar el archivo de pickle
filename = r"C:\Users\juane\AnaliticaComputacional\Proyecto3\monty.pkl"

with open(filename,'wb') as file:
    pickle.dump(estimated_modelPC , file)
    file.close()
    
    
    