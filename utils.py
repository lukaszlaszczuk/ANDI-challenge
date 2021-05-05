import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from configuration import RANDOM_STATE, CLASS_MAPPING

def custom_residual_function(model, X, y):
    probs = model.predict_proba(X)
    real_class_pred = probs.flatten()[[int(probs.shape[1]*i+val) for i, val in enumerate(y)]]
    return np.abs((1 - real_class_pred).astype('float64'))

def custom_predict(model, X):
    return model.predict_proba(X)

def custom_predict_normal(model, X):
    return model.predict(X)

def custom_predict_old(model, X):
    pred_class = np.argmax(model.predict_proba(X), axis=1)
    return pred_class.astype('float')

def loss_accuracy(y_real, y_pred):
    return 1 - accuracy_score(y_real, y_pred)

def loss_multiclass(y_real, y_proba):
    proba_real = np.array([el[y_real[i]] for i, el in enumerate(y_proba)])
    return np.mean(1 - proba_real)

def categorical_cross_entropy(y_real, y_proba):        
    return np.mean(-np.log(y_proba.flatten()[[int(y_proba.shape[1]*i+value) for i,value in enumerate(y_real)]]))

def load_model(path):
    model = joblib.load(path)
    return model

def load_data(path, ratio):
    X = pd.read_csv(path)
    X = X.drop(['file', 'Alpha'], axis=1)
    y, X = X['motion'], X.drop(['motion'], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=RANDOM_STATE)
    return x_train, x_test, y_train, y_test
