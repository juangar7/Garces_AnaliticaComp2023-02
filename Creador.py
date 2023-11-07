import pandas as pd
import numpy as np
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


table_name = 'estudiantes2'

# Crear la sentencia SQL para la creación de la tabla
create_table_sql = f"CREATE TABLE {table_name} ("

# Recorre las columnas del DataFrame y genera las columnas de la tabla
for column_name, data_type in zip(df.columns, df.dtypes):
    # Reemplaza caracteres no válidos en los nombres de columna
    formatted_column_name = column_name.replace('(', '').replace(')', '').replace(' ', '').replace('/', '').replace("'", '_')
    
    if data_type == 'object':
        data_type = 'TEXT'
    elif data_type == 'int64':
        data_type = 'INTEGER'
    elif data_type == 'float64':
        data_type = 'REAL'
    create_table_sql += f'"{formatted_column_name}" {data_type}, '  # Agrega comillas dobles a los nombres de columna

# Elimina la última coma y agrega el paréntesis de cierre
create_table_sql = create_table_sql.rstrip(', ') + ");"

# Ruta donde se creará el archivo SQL
sql_file_path = "C:/Users/LENOVO/temp/Proyecto2/estudiantes.sql"

# Guarda la sentencia SQL de creación de la tabla en un archivo
with open(sql_file_path, 'w') as sql_file:
    sql_file.write(create_table_sql)

# Iterar a través de las filas de datos en el DataFrame
for index, row in df.iterrows():
    # Generar un comando SQL INSERT para cada fila
    formatted_columns = ', '.join(map(lambda x: x.replace('(', '').replace(')', '').replace(' ', '').replace('/', '').replace("'", '_'), row.index))
    formatted_columns = ', '.join([f'"{col}"' for col in formatted_columns.split(', ')])
    values = ', '.join([f"'{value}'" if isinstance(value, str) else str(value) for value in row])
    insert_sql = f"INSERT INTO {table_name} ({formatted_columns}) VALUES ({values});\n"
    # Escribir el comando SQL INSERT en el archivo
    with open(sql_file_path, 'a') as sql_file:
        sql_file.write(insert_sql)