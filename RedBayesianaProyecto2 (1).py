import pandas as pd
import numpy as np
import pickle
import psycopg2

engine = psycopg2.connect(
    dbname="postgres",
    user="postgres",
    password="1463jwJE2212",
    host="tablero.cpq1o3uew0fx.us-east-1.rds.amazonaws.com",
    port='5432'
)
# Crear un cursor
cur = engine.cursor()

# Ejecuta la consulta SQL
cur.execute("SELECT * FROM estudiantes2")

# Recupera todos los datos de la consulta
data = cur.fetchall()

# Obtiene los nombres de las columnas
columns = [desc[0] for desc in cur.description]

# Crea un DataFrame de pandas
df = pd.DataFrame(data, columns=columns)

# Cierra el cursor y la conexión
cur.close()
engine.close()


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
filename = "C:/Users/LENOVO/temp/Proyecto2/monty.pkl"

with open(filename,'wb') as file:
    pickle.dump(estimated_modelHC2 , file)
    file.close()