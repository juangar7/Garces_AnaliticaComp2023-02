#!/usr/bin/env python
# coding: utf-8

# In[15]:


pip install --upgrade numexpr


# In[16]:


pip install ucimlrepo


# In[17]:


pip install seaborn


# In[18]:


import pandas as pd
import numpy as np


# In[19]:


from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
predict_students_dropout_and_academic_success = fetch_ucirepo(id=697) 
  
# data (as pandas dataframes) 
X = predict_students_dropout_and_academic_success.data.features 
y = predict_students_dropout_and_academic_success.data.targets 
  
# metadata 
print(predict_students_dropout_and_academic_success.metadata) 
  
# variable information 
print(predict_students_dropout_and_academic_success.variables)


# In[20]:


X.head()


# In[21]:


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


# In[22]:


#Se revisan valores faltantes
X.isna().sum()


# In[23]:


#Se revisan typos
valoresPorColumna = {}
for columns in X.columns:
    valoresPorColumna[columns]= X[columns].unique()


# In[24]:


#Estadisticas descriptivas variables demograficas de interes
variables =['Age at enrollment',"Marital Status", 'International',"Gender","Mother's occupation", "Father's occupation" ,"Previous qualification","Educational special needs"]
X[variables].describe()


# In[25]:


#Diagrama de caja e histograma
import seaborn as sns
import matplotlib.pyplot as plt

variable = "Gender"
plt.subplot(2, 1, 1)  # 1 fila, 2 columnas, gráfico 1 (diagrama de caja)
sns.boxplot(X[variable])
plt.title('Diagrama de Caja')

# Crear un histograma
plt.subplot(2, 1, 2)  # 1 fila, 2 columnas, gráfico 2 (histograma)
sns.histplot(X[variable], kde=True, color='blue')
plt.title('Histograma')

plt.tight_layout()  # Para asegurarse de que los gráficos no se superpongan
plt.show()


# In[26]:


#Matriz de correlaciones
variables =['Age at enrollment',"Marital Status", 'International',"Gender","Mother's occupation", "Father's occupation" ,"Previous qualification","Educational special needs"]
X[variables].corr(method = "kendall")


# In[37]:


import scipy.stats as stats
#Matriz de p-valores
variables =['Age at enrollment',"Marital Status", 'International',"Gender","Mother's occupation", "Father's occupation" ,"Previous qualification","Educational special needs"]
matriz = pd.DataFrame(np.zeros((len(variables),len(variables))))
matriz.set_index(pd.Index(variables), inplace = True)
matriz.columns = variables

for index in matriz.index:
    for column in matriz.columns:
        coeficiente_tau, p_valor = stats.kendalltau( X[index],X[column])
        matriz.loc[index,column]= p_valor
matriz


# In[28]:


X.columns


# In[31]:


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


# In[33]:


X.head()


# In[ ]:


#Revisar correlaciones Pearson y Kendall
import scipy.stats as stats

# Ejemplo de dos listas de datos
# Calcular el coeficiente de correlación de tau de Kendall
coeficiente_tau, p_valor = stats.kendalltau( X["Previous qualification"],X["International"])
print(coeficiente_tau, p_valor)


# In[ ]:


correlaciones = {}
for column in X.columns:
    coeficiente_tau, p_valor = stats.kendalltau( X[column],y["TargetN"])
    correlaciones[column]= (coeficiente_tau, p_valor)


# In[ ]:





# In[ ]:




