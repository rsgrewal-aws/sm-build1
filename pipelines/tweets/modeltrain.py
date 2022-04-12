
import os

import pandas as pd
from sklearn.linear_model import LogisticRegression
#from sklearn.externals import joblib
import logging
import joblib
import pickle
import xgboost as xgb
import argparse
import json

_logger = logging.getLogger()
_logger.setLevel(logging.INFO)
_logger.addHandler(logging.StreamHandler())

def merge_two_dicts(x, y):
    """Merges two dicts, returning a new copy."""
    z = x.copy()
    z.update(y)
    return z

def listLocalDirectory(dirPath="."):
    for path, dnames, fnames in os.walk(dirPath):
        _logger.info(f"List::path={path}::dirNames={dnames}::fileNames={fnames}::")

def textToVectors(text, vectorizer):
    vector = vectorizer.transform([text])
    return sum(vector.toarray()[0])

def vectorizerText(textArray):
    from sklearn.feature_extraction.text import TfidfVectorizer
    # create the transform
    vectorizer = TfidfVectorizer()
    # tokenize and build vocab
    vectorizer.fit(textArray)
    return vectorizer

def model_fn(model_dir):
    model = xgb.Booster()
    try:
        model.load_model(os.path.join(model_dir,'xgboost-model.json'))
    except:
        #ignore model must be of type xgboost-model
        print("error in loading the JSON version of xgboost model")
        model.load_model(os.path.join(model_dir,'xgboost-model'))
        
    return model

#####  Estimator does not use the entry point script you have to use the sklearn container
#####  so for estimator xgboost 1.01 that is in pickle format and so has to be loaded as pickle
#####  -- since  weh have 1.5 version of xgboost -- we have to save it as json and then reload it and then do the predictions ---
#####  that will solve the problem 

####  SO THIS CLASS IS NOT REALLY USED UNLESS we use sklearn estimator 

def _parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))

    return parser.parse_known_args()


if __name__ == "__main__":
    _logger.info(f"Model:xgboost:version:{xgb.__version__}")
    args, unknown_args = _parse_args()
    _logger.info(f"Model:xgboost:unknown_args={unknown_args}::args={args}::")
    
    training_data_directory = "/opt/ml/input/data/train"
    train_data = os.path.join(training_data_directory, "train.csv")
    _logger.info(f"Model:Logistic:regression:Reading input data from {training_data_directory}:")

    train_df = pd.read_csv(train_data, header=None)
    X_train = train_df.iloc[:,1:]
    y_train = train_df.iloc[:,:1]
    _logger.info(f"Model:train_df={train_df.shape}::X_train:shape={X_train.shape}:: y_train={y_train.shape}::")
     
    #model = LogisticRegression(class_weight="balanced", solver="lbfgs")
    param_dict = { 'objective':'binary:logistic'}
    model = xgb.XGBClassifier(**param_dict)
    _logger.info("Model:Training XGBOOST model")
    model.fit(X_train, y_train)
    
    #model_output_directory = os.path.join("/opt/ml/model", "model.joblib")
    #model_output_directory = os.path.join("/opt/ml/model", "xgboost-model.pkl")
    #_logger.info("OLDER:PKL:Model:Saving model to {}".format(model_output_directory))
    
    #pickle.dump(model, open(model_output_directory, 'wb'))
    #joblib.dump(model, model_output_directory)
    
    model_output_directory = os.path.join("/opt/ml/model", "xgboost-model.json")
    _logger.info("Model:Saving model to {}".format(model_output_directory))
    model.save_model(model_output_directory)

    _logger.info("Model:Trained:ALL Done added !!!")

       
    
    # ----------------   OLD TEMPLATE CODE --------------------#
