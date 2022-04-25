
import os
import xgboost as xgb
import logging
import traceback

_logger = logging.getLogger()
_logger.setLevel(logging.INFO)
_logger.addHandler(logging.StreamHandler())


def model_fn(model_dir):
    _logger.info("Inference ::Starting XGBOOST:: Model load::")
    AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")
    _logger.info(f"Inference ::Starting XGBOOST:: AWS_S3_BUCKET={AWS_S3_BUCKET}::AWS_ACCESS_KEY_ID={AWS_ACCESS_KEY_ID}::")
    
    model = xgb.Booster()
    try:
        _logger.info("Inference ::Starting via JSON:")
        model.load_model(os.path.join(model_dir,'xgboost-model.json'))
        return model
    except:
        #ignore model must be of type sagemaker algorithim which uses Pickle 
        _logger.info(f"error in loading the via JSON version of xgboost model:exp={traceback.format_exc()}")
    
    
    # --    DIRECT  
    try:
        _logger.info("Inference ::Starting XGBOOST:: Model:direct:in-built-algo:")
        model.load_model(os.path.join(model_dir,'xgboost-model'))
        return model
    except:
        _logger.info(f"error in loading the Model:direct: version of xgboost model:exp={traceback.format_exc()}")

    # -- DIRECT -- PKL 
    try:
        import pickle as pkl
        _logger.info("Inference ::Starting XGBOOST:: PKL:Model:direct:in-built-algo:")
        with open(os.path.join(model_dir, "xgboost-model"), "rb") as f:
            model = pkl.load(f)
        return model
    except:
        _logger.info(f"error in loading the Model:PKL:Model:direct: version of xgboost model:exp={traceback.format_exc()}")
    
    # -- PICKLE JSON
    try:
        _logger.info("Inference ::Starting XGBOOST:: Model:TAR:PICKLE:JSON:direct:in-built-algo:")
        import tarfile
        import pickle as pkl 
        t = tarfile.open('model.tar.gz', 'r:gz')
        t.extractall()
        model = pkl.load(open(os.path.join(model_dir,'xgboost-model.json'), 'rb'))
        return model
    except:
        #ignore model must be of type sagemaker algorithim which uses Pickle 
        _logger.info(f"error in loading the Model:TAR:PICKLE:JSON: version of xgboost model:exp={traceback.format_exc()}")
        
    # -- PICKLE DIRECT    
    try:
        _logger.info("Inference ::Starting XGBOOST:: Model:TAR:PICKLE:direct:direct:in-built-algo:")
        import tarfile
        import pickle as pkl 
        t = tarfile.open('model.tar.gz', 'r:gz')
        t.extractall()
        model = pkl.load(open(os.path.join(model_dir,'xgboost-model'), 'rb'))
        return model
    except:
        #ignore model must be of type sagemaker algorithim which uses Pickle 
        _logger.info(f"error in loading the Model:TAR:PICKLE:direct version of xgboost model:exp={traceback.format_exc()}")
    
    _logger.info(f"FATAL: All method of loading failed -- RETURNING Nothing !!!")
    return model


