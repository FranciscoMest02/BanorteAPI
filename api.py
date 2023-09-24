'''
Estas son las librerias que instalamos para que funcione la api
pip install flask pymongo
pip install flask-pymongo
pip install flask cors
pip install python-dotenv


Para entrar al ambiente virtual:
venv\Scripts\activate

Este no es el script bueno de la api
'''

#-----------------Esta es la inicializacion de la base de datos-------------------------
#Importamos las librerias
import os
from pickle import TRUE
from dotenv import load_dotenv 
from flask import Flask, request
from bson.objectid import ObjectId
from flask_cors import CORS
import pandas as pd
import openai
import os

import json
import csv

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__) #Inicializamos Flask
CORS(app)

df = pd.read_csv('data.csv') # abrir el archivo de datos con los nombres dados para las columnas

# Use the pd.get_dummies() function to one-hot encode the 'Category' column
one_hot = pd.get_dummies(df['Establecimiento'], prefix='Establecimiento')
one_hot_columns = []
for df_name in one_hot.columns.tolist():
    one_hot_columns.append(df_name)

# Concatenate the one-hot encoded columns with the original DataFrame
df = pd.concat([df, one_hot], axis=1)

# Drop the original 'Category' column if needed
df = df.drop('Establecimiento', axis=1)

#dividimos en la variable dependiente y las independientes
X = df.drop("Etiqueta", axis=1)
y = df["Etiqueta"]

#Otenemos nuestros datos de prueba, validacion y de entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

model = RandomForestClassifier(criterion= 'gini',
 max_depth= None,
 max_leaf_nodes= None,
 min_samples_leaf= 1,
 min_samples_split= 2,
 n_estimators= 50)

model.fit(X_train, y_train)

def get_chatgpt_response(messages):
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=messages
    )

    return response['choices'][0]['message']['content']

#--------------------------Estas son las llamadas a la base de datos--------------------------------
#not yet used
@app.route('/data', methods=['GET'])
def get_data():
    data_json = df.to_json(orient = 'table')

    # Parse the JSON string
    data_dict = json.loads(data_json)

    # Extract the "data" part
    data = data_dict.get("data", [])

    return data

@app.route('/predict', methods=['POST'])
def predict():
    #Usamos el body para los parametros porque son sensibles y tambien porque queremos que sean case sensitive (mientras no los mandamos con la app en hash)
    establecimiento = request.json['Establecimiento'] 
    monto = request.json['Monto'] 
    repeticiones = request.json['Repeticiones'] 

    '''
    API call example
    {
    "Establecimiento": H&M,
    "Monto": 500,
    "Repeticiones": 1,
    }
    '''

    inputData = {
        "Establecimiento": establecimiento,
        "Monto": monto,
        "Repeticiones": repeticiones,
    }
    
    inputDf = pd.DataFrame([inputData])

    oneHotColumns = {}
    for column in one_hot_columns:
        if establecimiento in column:
            oneHotColumns[column] = [1]
        else:
            oneHotColumns[column] = [0]

    inputDf = inputDf.drop('Establecimiento', axis=1)

    dfOH = pd.DataFrame(oneHotColumns)

    inputDf = pd.concat([inputDf, dfOH], axis=1)

    y_pred = model.predict(inputDf)

    return {
        'prediction': y_pred[0]
    }

@app.route('/predict/user', methods=['GET'])
def predict_user():
    inputDf = pd.read_csv('data_real.csv')

    oneHotColumns = {}
    for column in one_hot_columns:
        list = []
        for establecimiento in inputDf['Establecimiento']:
            if establecimiento in column:
                list.append(1)
            else:
                list.append(0)

        oneHotColumns[column] = list

    inputDf = inputDf.drop('Establecimiento', axis=1)
    inputDf = inputDf.drop('Etiqueta', axis=1)

    dfOH = pd.DataFrame(oneHotColumns)

    inputDf = pd.concat([inputDf, dfOH], axis=1)

    y_pred = model.predict(inputDf)

    return {
        'cost': inputDf['Monto'].tolist(),
        'prediction': y_pred.tolist()
    }

@app.route('/spendings/user', methods=['POST'])
def spendings_users():
    ingreso = request.json['Ingreso']

    gastosFijos = ['Libros', 'Comida', 'Gimnasio', 'Cafetería', 'Farmacia', 'Transporte', 'Electrónicos', 'Alquiler', 
                     'Material de estudio', 'Médico', 'Gasolina', 'Teléfono', 'Impresiones', 'Taxis', 'Lavandería', 'Deportes',
                     'Regalos', 'Viaje', 'Ahorro', 'Seguro médico', 'Reparaciones', 'Transporte público', 'Internet']
    gastosHormiga = ['Ropa', 'Suscripciones', 'Pizza', 'Entretenimiento',  'Cine', 'Conciertos', 'Fiesta', 'Café',]

    fijos = []
    hormiga = []

    classes = predict_user()

    sumFijos = 0
    sumHormiga = 0
    sumTotal = 0
    for i in range(len(classes['prediction'])):
        sumTotal += classes['cost'][i] * 15
        if classes['prediction'][i] in gastosFijos:
            sumFijos += classes['cost'][i] * 15
            fijos.append({'category': classes['prediction'][i], 
                          'cost': classes['cost'][i] * 15})
        elif classes['prediction'][i] in gastosHormiga:
            sumHormiga += classes['cost'][i] * 15
            hormiga.append({'category': classes['prediction'][i], 
                            'cost': classes['cost'][i] * 15})


    return {
        'spendings':{
            'fijos': {
                'list': fijos,
                'total': sumFijos,
                'percentage': round(sumFijos * 100 / ingreso)
            },
            'hormiga': {
                'list': hormiga,
                'total': sumHormiga,
                'percentage': round(sumHormiga * 100 / ingreso)
            },
            'total': sumTotal
        },
    }

@app.route('/chat/fondo', methods=["POST"])
def getChatResponse():
    return{
        "status": "chat not available"
    }
    messages = request.json['messages']
    print(messages)
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    model_response = get_chatgpt_response(messages)
    return {"response": model_response}

if __name__ == "__main__":
    app.run(debug=True) #Con el True en debug se reinicia cuando hay cambios