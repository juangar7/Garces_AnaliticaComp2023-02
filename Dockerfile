# syntax=docker/dockerfile:1
FROM ubuntu:22.04

# Instalar dependencias de la aplicación
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip install flask==2.2.*

# Copiar la aplicación y archivos necesarios al contenedor
COPY DashboarddefinitivoP3.py /
COPY datosValle4000.csv /
COPY resultados.pkl /

# Configuración final
ENV FLASK_APP=DashboarddefinitivoP3
EXPOSE 8050

# Comando a ejecutar al correr el contenedor
CMD flask run --host 0.0.0.0 --port 8050