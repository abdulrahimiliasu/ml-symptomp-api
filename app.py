from flask import Flask
import numpy as np
from scipy.stats import mode
import training


encoder, data_dict, final_svm_model, final_rf_model, final_nb_model = training.train()

def predictDisease(symptoms):
    symptoms = symptoms.split(",")

    # creating input data for the models
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        print(data_dict["symptom_index"])
        index = data_dict["symptom_index"][symptom]
        input_data[index] = 1

    # reshaping the input data and converting it
    # into suitable format for model predictions
    input_data = np.array(input_data).reshape(1, -1)

    # generating individual outputs
    rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
    nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
    svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]

    # making final prediction by taking mode of all predictions
    final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])[0][0]
    predictions = {
        "rf_model_prediction": rf_prediction,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": nb_prediction,
        "final_prediction": final_prediction
    }
    return predictions

app = Flask(__name__)

# Itching&Skin_Rash&Nodal_Skin_Eruptions&Dischromic_Patches

@app.route('/')
def hello():
    return "Symptom API"

@app.route('/predict/<syms>')
def hello_world(syms):
    ss = syms.split("&")
    s = ""
    for a in ss:
        a = a.replace('_', ' ')
        s += f'{a},'
    predictions = predictDisease(s[:-1])
    return predictions


if __name__ == '__main__':
    app.run(port=12345, debug=True)
