import os
import xgboost as xgb
import logging


_logger = logging.getLogger()
_logger.setLevel(logging.INFO)
_logger.addHandler(logging.StreamHandler())


def model_fn(model_dir):
    _logger.info("Inference ::Starting XGBOOST:: Model load::")
    model = xgb.Booster()
    try:
        model.load_model(os.path.join(model_dir,'xgboost-model.json'))
    except:
        #ignore model must be of type sagemaker algorithim which uses Pickle 
        _logger.info("error in loading the JSON version of xgboost model")
        model.load_model(os.path.join(model_dir,'xgboost-model'))
        import pickle as pkl 
        import tarfile

        t = tarfile.open('model.tar.gz', 'r:gz')
        t.extractall()

        model = pkl.load(open(os.path.join(model_dir,'xgboost-model'), 'rb'))
        _logger.info("Loaded model using Pickle as sagemaker uses that for xgboost model")
        
    return model
