# BanorteAPI
API utilizada para el proyecto del hack MTY 2023

## Si es la primera vez que descargas la api sigue estos pasos:

### 1. Crea un virtual environment:
#### Para windows
python -m venv venv (el segundo venv es el nombre del entorno virtual)
#### Para Mac
sudo pip install virtualenv
virtualenv venv

### 2. Abre el virtual environment: 
#### Para windows
venv\Scripts\activate
#### Para Mac
source venv/bin/activate

### 3. Instala las librerias necesarias para que funcione la api dentro del ambiente virtual:
pip install flask pymongo
pip install flask-pymongo
pip install flask-cors
pip install python-dotenv
pip install json
pip install scikit-learn

## Para correr la api:

### 1. Inicia el virtual environment:
Puedes ver los comandos en el punto 2 de la inicializacion del proyecto

### 2. Corre el archivo con la api:
#### Para windows
python api.py (dentro del venv)
#### Para Mac
flask --app api.py run


## Otras especificaciones

El archivo de train_clean_base.csv **no se debe modificar** ya que son los datos base que se obtuvieron de Kaggle. Se recomienda hacer una copia del archivo y nombrarla *train_clean.csv* ya que será la "base de datos" de la que el modelo se entrenará y a la que se le añadirán nuevos registros. Si se quiere "reiniciar" esta base de datos se debe borrar el archivo y hacer una nueva copia de train_clean_base.csv
