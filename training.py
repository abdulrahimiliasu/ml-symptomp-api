import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix
import json


def train():

	DATA_PATH = "/Training.csv"
	data = pd.read_csv(DATA_PATH).dropna(axis = 1)

	disease_counts = data["prognosis"].value_counts()

	# PROGNOSIS COUNT: 131

	temp_df = pd.DataFrame({
		"Disease": disease_counts.index,
		"Counts": disease_counts.values
	})


	encoder = LabelEncoder()
	data["prognosis"] = encoder.fit_transform(data["prognosis"])

	X = data.iloc[:,:-1]
	y = data.iloc[:, -1]
	X_train, X_test, y_train, y_test =train_test_split(
	X, y, test_size = 0.2, random_state = 24)

	print(f"Train: {X_train.shape}, {y_train.shape}")
	print(f"Test: {X_test.shape}, {y_test.shape}")


	def cv_scoring(estimator, X, y):
		return accuracy_score(y, estimator.predict(X))


	models = {
		"SVC":SVC(),
		"Gaussian NB":GaussianNB(),
		"Random Forest":RandomForestClassifier(random_state=18)
	}


	for model_name in models:
		model = models[model_name]
		scores = cross_val_score(model, X, y, cv = 10,
								n_jobs = -1,
								scoring = cv_scoring)
		print("=="*30)
		print(model_name)
		print(f"Scores: {scores}")
		print(f"Mean Score: {np.mean(scores)}")


	svm_model = SVC()
	svm_model.fit(X_train, y_train)
	preds = svm_model.predict(X_test)

	print(f"Accuracy on train data by SVM Classifier\
	: {accuracy_score(y_train, svm_model.predict(X_train))*100}")

	print(f"Accuracy on test data by SVM Classifier\
	: {accuracy_score(y_test, preds)*100}")

	nb_model = GaussianNB()
	nb_model.fit(X_train, y_train)
	preds = nb_model.predict(X_test)
	print(f"Accuracy on train data by Naive Bayes Classifier\
	: {accuracy_score(y_train, nb_model.predict(X_train))*100}")

	print(f"Accuracy on test data by Naive Bayes Classifier\
	: {accuracy_score(y_test, preds)*100}")
	cf_matrix = confusion_matrix(y_test, preds)

	rf_model = RandomForestClassifier(random_state=18)
	rf_model.fit(X_train, y_train)
	preds = rf_model.predict(X_test)
	print(f"Accuracy on train data by Random Forest Classifier\
	: {accuracy_score(y_train, rf_model.predict(X_train))*100}")

	print(f"Accuracy on test data by Random Forest Classifier\
	: {accuracy_score(y_test, preds)*100}")


	final_svm_model = SVC()
	final_nb_model = GaussianNB()
	final_rf_model = RandomForestClassifier(random_state=18)
	final_svm_model.fit(X, y)
	final_nb_model.fit(X, y)
	final_rf_model.fit(X, y)

	test_data = pd.read_csv("/Testing.csv").dropna(axis=1)

	test_X = test_data.iloc[:, :-1]
	test_Y = encoder.transform(test_data.iloc[:, -1])

	svm_preds = final_svm_model.predict(test_X)
	nb_preds = final_nb_model.predict(test_X)
	rf_preds = final_rf_model.predict(test_X)

	final_preds = [mode([i,j,k])[0][0] for i,j,
				k in zip(svm_preds, nb_preds, rf_preds)]

	print(f"Accuracy on Test dataset by the combined model\
	: {accuracy_score(test_Y, final_preds)*100}")

	joblib.dump(final_nb_model, "Naive_Base.joblib")
	joblib.dump(final_rf_model, "Random_Forest.joblib")
	joblib.dump(final_svm_model, "Support_Vector.joblib")

	symptoms = X.columns.values

	symptom_index = {}
	for index, value in enumerate(symptoms):
		symptom = " ".join([i.capitalize() for i in value.split("_")])
		symptom_index[symptom] = index

	print(type(symptom_index))
	data_dict = {
		"symptom_index":symptom_index,
		"predictions_classes":encoder.classes_
	}

	a_file = open("symptom_index.json", "w")
	json.dump(symptom_index, a_file)
	a_file.close()

	np.save('classes.npy', encoder.classes_)
	return encoder, data_dict, final_svm_model, final_rf_model, final_nb_model


