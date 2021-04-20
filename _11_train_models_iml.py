import argparse
import json
import joblib
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split

from configuration import RANDOM_STATE

parser = argparse.ArgumentParser()
parser.add_argument('--datapath', help='Path to characteristic data')
parser.add_argument('--hyperparampath', help='Path to hyperparameters file')
parser.add_argument('--savepath', help='Path to save model')


def prepare_data(path):
    X = pd.read_csv(path)
    X = X.drop(['file', 'Alpha'], axis=1)
    y, X = X['motion'], X.drop(['motion'], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=1 / 6, random_state=RANDOM_STATE)
    return x_train, x_test, y_train, y_test


def load_hyperparams(hyperparampath):
    with open(hyperparampath, 'r') as f:
        return json.load(f)


def train_model(hyperparams, x_train, y_train):
    model = GradientBoostingClassifier()
    model.set_params(**hyperparams)
    model.fit(x_train, y_train)
    return model


def save_model(model, savepath):
    joblib.dump(model, savepath)


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    print(args.datapath)
    datapath = args.datapath
    hyperparampath = args.hyperparampath
    savepath = args.savepath

    x_train, x_test, y_train, y_test = prepare_data(datapath)
    hyperparams = load_hyperparams(hyperparampath)
    model = train_model(hyperparams, x_train, y_train)
    save_model(model, savepath)
