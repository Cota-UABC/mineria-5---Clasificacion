from config import *
from basic_funcs import *
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

def bayes(X_train, X_test, y_train, y_test, file_f):
	model = GaussianNB()

	model.fit(X_train, y_train)

	y_pred = model.predict(X_test)

	accuracy = accuracy_score(y_test, y_pred)

	tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
	positives = tp + fn  
	negatives = tn + fp

	if file_f:
		save_predictions(X_test, y_test, y_pred)

	return accuracy, tp, fp, tn, fn, positives, negatives