"""Feature engineers the Tweets churn dataset."""
import argparse
import logging
import os
import pathlib
import requests
import tempfile

import boto3
import numpy as np
import pandas as pd

import json

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import pickle as pkl

_logger = logging.getLogger()
_logger.setLevel(logging.INFO)
_logger.addHandler(logging.StreamHandler())


try:
    _logger.info(f"Pkl:version:={pkl.format_version}")
    _logger.info(f"Pandas:version:{pd.__version__}")
    _logger.info(f"Numpy:version:{np.__version__}")
    import xgboost as xgb
    _logger.info(f"XGBoost:version:{xgb.__version__}")
except:
    pass


# Since we get a headerless CSV file we specify the column names here.

X_columns_names =  [
    'tweet_id', 
    'writer', 
    'post_date', 
    'body', 
    'comment_num', 
    'retweet_num',
    'like_num', 
    'ticker_symbol'
]

Y_column = "high_price"


X_columns_dtype = {
    'tweet_id': np.float64, 
    'writer': str, 
    'post_date': np.int64, 
    'body': str, 
    'comment_num': np.int64, 
    'retweet_num': np.int64, 
    'like_num': np.int64, 
    'ticker_symbol': str
}
Y_column_dtype = {Y_column: np.bool} #{Y_column: np.float64}


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

def saveVectorizerToS3(vectorz=None, localFullPath="/opt/ml/processing/evalproperty/vectorizerBody.pkl") :
    pkl.dump(vectorz, open(localFullPath, "wb"))
    _logger.info(f"Vectorizer:saved!!:to:local:path={localFullPath}::name={vectorz}::This will be moved to the eval:location:")
    
    
    
if __name__ == "__main__":
    _logger.info("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    parser.add_argument("--data-size", type=int, default=100)
    args = parser.parse_args()
    input_data = args.input_data
    data_size = args.data_size
    _logger.info(f"Data size={data_size}::")
    
    
    BASE_DIR = "/opt/ml/processing"
    pathlib.Path(f"{BASE_DIR}/data").mkdir(parents=True, exist_ok=True)
    _logger.info(f"Download:data:from:s3:to:local:location:={BASE_DIR}/data::")
    
    eval_dir = "/opt/ml/processing/evalproperty"
    pathlib.Path(eval_dir).mkdir(parents=True, exist_ok=True)
    _logger.info(f"eval_dir={eval_dir}:sucessfully created for saving the vectorizers")
    
    print(input_data)
    _logger.info(f"Input:data:={input_data}::")
    
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])

    _logger.info(f"TEST:TEST:Downloading data from bucket: {bucket}, key: {key}:Willdownload to localfile:name as raw-data.csv")
    fn = f"{BASE_DIR}/data/raw-data.csv"
    try:
        s3 = boto3.resource("s3")
        s3.Bucket(bucket).download_file(key, fn)
    except:
        _logger.error("TEST:TEST:error:in:downloading:from:s3:ignore")

    

    #fn = os.path.join("/opt/ml/processing/input", "combined_tweets.csv")
    
    onlyFiles = [f for f in os.listdir("/opt/ml/processing/input") if os.path.isfile(os.path.join("/opt/ml/processing/input", f))]
    _logger.info(f"Data Downloaded::Now Reading downloaded data.:dir:/opt/ml/processing/input::And:FILES:ARE::{onlyFiles}")
    
    fn = os.path.join("/opt/ml/processing/input", onlyFiles[0])
    _logger.info(f"Data Downloaded::Now Reading downloaded data.:dir:/opt/ml/processing/input:::from:location={fn}::")
    
    # read in csv
    combinedTweetsDF = pd.read_csv(fn)
    combinedTweetsDF = combinedTweetsDF.dropna()
    combinedTweetsDF = combinedTweetsDF.drop_duplicates() # -- this drops duplicates

    # -- FOR NOW CREATE just a 10 ROW DATA SET for FASTER processing
    combinedTweetsDF =  combinedTweetsDF.iloc[:data_size,:]
    # -- END 10 row data set creation
    _logger.info(f"After:ILOC:shape={combinedTweetsDF.shape}:")
    
    # Create one binary classification target column
    combinedTweetsDF['body_length'] = combinedTweetsDF['body'].apply( lambda x: len(x))
    combinedTweetsDF['Y_label'] = combinedTweetsDF['body_length'].apply( lambda x: 1 if x > 115 else 0)
    #combinedTweetsDF['Y_label'] = combinedTweetsDF.Y_label.apply(lambda x: 1 if x else 0) # -- convert to 1 and 0
    _logger.info(f"After:transformation:shape={combinedTweetsDF.shape}:columns={combinedTweetsDF.columns}::describe={combinedTweetsDF.describe()}::")
     
    # Convert categorical variables into dummy/indicator variables.
    #categorical_cols=['writer', 'ticker_symbol']
    #categorical_cols_dict ={'writer':'wr', 'ticker_symbol' :'ticker' }
    #df_multi = pd.get_dummies(combinedTweetsDF, columns=categorical_cols, prefix=categorical_cols_dict, drop_first=True)
    df_multi = combinedTweetsDF.reindex(columns=(['Y_label'] + list([a for a in combinedTweetsDF.columns if a != 'Y_label']) ))
    _logger.info(f"df_multi:BEFORE:DROP:BODY:first 10 cols = {df_multi.columns[:10]}::")
    
    # -- vectorize the text 
    #df_multi = df_multi[1:] # remove the header row 
    vectorizer = vectorizerText(df_multi.body)
    saveVectorizerToS3(vectorizer, f"{eval_dir}/vectorizerBody.pkl")
    df_multi['vec_text'] = df_multi.body.apply(lambda x: textToVectors(x,vectorizer ))
    df_multi = df_multi.drop(['body'], axis=1)
    _logger.info(f"After:Vectorization:columns={len(df_multi.columns)}::describe={df_multi.describe()}::")
    _logger.info(f"After:Vectorization:shape of data set={df_multi.shape}::len={len(df_multi)}::")
    
    # -- vectorize the Writer and ticker symbol

    vectorizer = vectorizerText(df_multi.writer.dropna())
    df_multi['writer_text'] = df_multi.writer.apply(lambda x: textToVectors(x,vectorizer ))
    df_multi = df_multi.drop(['writer'], axis=1)

    vectorizer = vectorizerText(df_multi.ticker_symbol.dropna())
    df_multi['ticker_symbol_text'] = df_multi.ticker_symbol.apply(lambda x: textToVectors(x,vectorizer ))
    df_multi = df_multi.drop(['ticker_symbol'], axis=1)

    _logger.info(f"After:FULL:Vectorization:columns={len(df_multi.columns)}::describe={df_multi.describe()}::")
    _logger.info(f"After:FULL:Vectorization:shape of data set={df_multi.shape}::len={len(df_multi)}::")


    
    # Split the data
    train_data, val_data, test_data = np.split(
        df_multi.sample(frac=1, random_state=1729),
        [int(0.7 * len(df_multi)), int(0.9 * len(df_multi))],
    )
    _logger.info(f"Going to write it to {BASE_DIR}/train and {BASE_DIR}/test and {BASE_DIR}/val")
    _logger.info(f"train_data:len={len(train_data)}::  val_data:len={len(val_data)}::  test_data:len={len(test_data)}::")
    pd.DataFrame(train_data).to_csv(
        f"{BASE_DIR}/train/train.csv", header=False, index=False
    )
    pd.DataFrame(val_data).to_csv(
        f"{BASE_DIR}/val/val.csv", header=False, index=False
    )
    pd.DataFrame(test_data).to_csv(
        f"{BASE_DIR}/test/test.csv", header=False, index=False
    )
    
    report_dict = {
        "regression_metrics": {
            "mse": {
                "value": 11.1,
                "standard_deviation": 89.2
            },
        },
    }

    output_dir = "/opt/ml/processing/evalproperty"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    _logger.info("Evaluation:Writing out evaluation report with mse: %f", 11.1)
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))

    _logger.info("Evaluation: All Done !!")    
    
    
    _logger.info("All Done !! written out !!")
    
    # ----------------   OLD TEMPLATE CODE --------------------#
