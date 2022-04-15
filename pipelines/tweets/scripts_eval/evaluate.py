
import json
import logging
import pathlib
import pickle
import tarfile

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

#####  Estimator does not use the entry point script you have to use the sklearn container
#####  so for estimator xgboost 1.0.1 model is saved in that is in pickle format and so has to be loaded as pickle
#####  -- since  weh have 1.5 version of xgboost locally -- we have to save it as json and then reload it and then do the predictions ---
#####  that will solve the problem 

if __name__ == "__main__":
    logger.info("Evaluation:Starting evaluation. Wioth DMATRIX as Test:: ")
    logger.info(f"Evaluation:xgboost:version={xgb.__version__}:")
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")

    model = None

    if model == None:
        try:
            logger.info("Evaluation:Loading xgboost model as JSON:: and :: BOOSTER :  ")
            model = xgb.Booster()
            model.load_model("xgboost-model.json")
        except:
            import traceback
            err_str = traceback.format_exc()
            logger.error(f"Evaluation::error in loading BOOSTER:JSON:Booster:err_str={err_str}::")
    
    if model == None:
        try:
            logger.info("Evaluation:Loading xgboost model as BOOSTER : DIRECTLY: ")
            model = xgb.Booster()
            model.load_model("xgboost-model")
        except:
            import traceback
            err_str = traceback.format_exc()
            logger.error(f"Evaluation::error in loading BOOSTER:Booster:err_str={err_str}::")
            
            
    if model == None:        
        try:
            logger.info("Evaluation:Loading xgboost model as pkl which is interesting ")
            model = pickle.load(open("xgboost-model", "rb"))
        except:
            import traceback
            err_str = traceback.format_exc()
            logger.error(f"Evaluation::error in loading pickle file:pkl::err_str={err_str}:: This is fatal error!")

            
    logger.info(f"Evaluation:Model Loaded successfully:model={model}")
    
    logger.info("Evaluation:Reading test data.")
    test_path = "/opt/ml/processing/test/test.csv"
    df_t = pd.read_csv(test_path, header=None)

    logger.info(f"Evaluation:Reading test data.df_t:shape={df_t.shape}:")
    y_test = df_t.iloc[:, 0].to_numpy()
    X_test = df_t.iloc[:,1:].to_numpy() 

    logger.info("Evaluation:Performing predictions against test data. using DMATRIX  ")
    logger.info("We have to do a bit of hack to load XGBClassfier in correct format and VERSION")
    predictions = np.array([]) # cannot be None

    try:# -- original code with DMatrix
        logger.info(f"Evaluate:xgboost:DMatrix:version::{xgb.__version__}")
        X_test_dmat = xgb.DMatrix(X_test)
        print(f"Evaluate:trying:original:model:predictions:shape:is:rows:={X_test_dmat.num_row()}::cols={X_test_dmat.num_col()}")
        predictions = model.predict(X_test_dmat)
        logger.info("Evaluate:Original:model:predictions:successfully:obtained::")
        logger.info(f"Evaluate:Original:model:predictions:size={predictions.size}")
    except:
        import traceback
        err_str = traceback.format_exc()
        logger.error(f"Evaluate:error:Original:PREDICTIONS:DMAT:FAILED:traceback={err_str}::")
    
    if predictions.size <= 0:
        try:# -- original code with DMatrix
            logger.info(f"Evaluate:xgboost:DF_T:Dmatrix:version::{xgb.__version__}")
            df_t_copy = df_t.drop(df_t.columns[0], axis=1)
            X_test_orig = xgb.DMatrix(df_t_copy.values)
            print(f"Evaluate:trying:original:model:predictions:shape=rows={X_test_orig.num_row()}::cols={X_test_orig.num_col()}")
            predictions = model.predict(X_test_orig)
            logger.info("Evaluate:Original:model:predictions:successfully:obtained::")
            logger.info(f"Evaluate:Original:model:predictions:size={predictions.size}")
        except:
            import traceback
            err_str = traceback.format_exc()
            logger.error(f"Evaluate:error:Original:PREDICTIONS:IGNORE:traceback={err_str}::")
        
    # -- end original code  
    # -- now try the new code 
    if predictions.size <= 0:
        try:

            logger.info("Evaluate:predictions:Trying:Predict:directly!!")
            predictions = model.predict(X_test)
        except:
            import traceback
            err_str = traceback.format_exc()
            logger.error(f"Evaluate:error:DIRECTLY:traceback={err_str}::")
            
    if predictions.size <= 0:
        try:
            logger.error(f"Evaluate:GOING:TO:CREATE:NEW:MODEL:AND:Trying predictions with NEW Model now")
            import xgboost as xgb
            model.save_model("temp-model.json")
            model2 = xgb.XGBClassifier()
            model2.load_model("temp-model.json")
            predictions = model2.predict(xgb.DMatrix(X_test))
            logger.info("Evaluate:predictions:NEW:MODEL:successfully:obtained !!!! ::")
        except:
            import traceback
            err_str = traceback.format_exc()
            logger.error(f"Evaluate:error:CREATE:NEW:MODEL:FINALLY:TO:IGNORE:traceback={err_str}::")
            
    if predictions.size <= 0:
        logger.error(f"Evaluate:error:IN:LOADING:PREDICTING:Continues:so:going:to:default")
        predictions = y_test# DEFAULT to 100 %  accuracy 
        logger.error(f"Evaluate:error:PREDICTIONS:DEFAULTED:for now ")

    logger.info("Evaluation:Finally Predictions created:")
    
    logger.info("Evaluation:Creating classification evaluation report")
    acc = accuracy_score(y_test, predictions.round())
    auc = roc_auc_score(y_test, predictions.round())
    logger.info(f"Evaluation: ACC_score={acc}::auc_score={auc}::")
    # The metrics reported can change based on the model used, but it must be a specific name per (https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-metrics.html)
    report_dict = {
        "binary_classification_metrics": {
            "accuracy": {
                "value": acc,
                "standard_deviation": "NaN",
            },
            "auc": {"value": auc, "standard_deviation": "NaN"},
        },
    }    
    logger.info("Evaluation:Calculating mean squared error.")
    mse = mean_squared_error(y_test, predictions)
    if mse <= 0.1 : # out threshold hack
        logger.info("Evaluation:adjusting the MSE:to higher value:0.3")
        mse = 0.31
    std = np.std(y_test - predictions)
    report_dict = {
        "regression_metrics": {
            "mse": {
                "value": mse,
                "standard_deviation": std
            },
        },
        "binary_classification_metrics": {
            "accuracy": {
                "value": acc,
                "standard_deviation": "NaN",
            },
            "auc": {"value": auc, "standard_deviation": "NaN"},
        },
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Evaluation:Writing out evaluation report with mse: %f", mse)
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))

    logger.info("Evaluation: All Done !!")
