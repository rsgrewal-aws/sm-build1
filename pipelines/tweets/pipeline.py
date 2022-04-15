
import os

import boto3
import sagemaker
import sagemaker.session

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
    FrameworkProcessor
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
    JsonGet,
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
)
from sagemaker.workflow.step_collections import RegisterModel

from sagemaker.workflow.conditions import (
    ConditionGreaterThanOrEqualTo,
)

from sagemaker.sklearn import SKLearnModel
from sagemaker.xgboost import XGBoostModel
from sagemaker.model import Model

from sagemaker.workflow.steps import CreateModelStep


BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_sagemaker_client(region):
    """Gets the sagemaker client.

        Args:
            region: the aws region to start the session
            default_bucket: the bucket to use for storing the artifacts

        Returns:
            `sagemaker.session.Session instance
    """
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def get_pipeline(
    region,
    sagemaker_project_arn=None,
    role=None,
    default_bucket=None,
    model_package_group_name="TweetsPackageGroup",
    pipeline_name="TweetsPipeline",
    base_job_prefix="Tweets",
):
    """Gets a SageMaker ML Pipeline instance working with on abalone data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    sm_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sm_session)
        
    #default_bucket = sm_session.default_bucket()

    print(f"Using:role={role}:")
    print(f"using SageMaker session={sm_session}:")

    # Parameters for pipeline execution
    processing_instance_count = ParameterInteger(
            name="ProcessingInstanceCount", default_value=1
    )
    processing_instance_type = ParameterString(
            name="ProcessingInstanceType", default_value="ml.m5.xlarge"
    )
    training_instance_type = ParameterString(
            name="TrainingInstanceType", default_value="ml.m5.xlarge"
    )
    model_approval_status = ParameterString(
            name="ModelApprovalStatus",
            default_value="PendingManualApproval",  # ModelApprovalStatus can be set to a default of "Approved" if you don't want manual approval.
    )
    input_data = ParameterString(
            name="InputDataUrl",
            default_value=f"s3://{default_bucket}/data/finance/combined_tweets.csvv",  # Change this to point to the s3 location of your raw input data.
    )
    print(f"pipeline:get_pipeline::processor:")
    # Cache Pipeline steps to reduce execution time on subsequent executions

    from sagemaker.workflow.steps import CacheConfig
    cache_config = CacheConfig(enable_caching=True, expire_after="1d")
    print(f"pipeline::get_pipeline:cache:config:enabled:")

    print(f"Pipeline_name={pipeline_name}:")
    print(f"Pipeline:base:job:prefix={base_job_prefix}:")



    # Processing step for feature engineering
    sklearn_processor = SKLearnProcessor(
            framework_version="0.23-1",
            instance_type=processing_instance_type,
            instance_count=processing_instance_count,
            base_job_name=f"smjobs-sklearn-tweets-preprocess/{base_job_prefix}" ,#f"{BASE_JOB_PREFIX}-sklearn-TweetsChurn-preprocess",  # choose any name
            sagemaker_session=sm_session,
            role=role,
        )

    inputs_p=[
        ProcessingInput(
            source=f"s3://{default_bucket}/data/finance/combined_tweets.csv",
            destination='/opt/ml/processing/input'
        ),
     ]
    outputs_p=[
        ProcessingOutput(
            s3_upload_mode="EndOfJob",
            output_name='train',
            source='/opt/ml/processing/train',
            destination=f's3://{default_bucket}/data/finance/curated/small/train'
        ),
        ProcessingOutput(
            s3_upload_mode="EndOfJob",
            output_name='test',
            source='/opt/ml/processing/test',
            destination=f's3://{default_bucket}/data/finance/curated/small/test'
        ),
        ProcessingOutput(
            s3_upload_mode="EndOfJob",
            output_name='validation',
            source='/opt/ml/processing/val',
            destination=f's3://{default_bucket}/data/finance/curated/small/validation'
        ),
        ProcessingOutput(
            output_name="evaluation-property-pass",
            source="/opt/ml/processing/evalproperty",
            destination=f's3://{default_bucket}/data/finance/curated/small/evalproperty'
        ),


    ]
    # -- if we d onot create a output then this directory is never creatd on tbe processing job
    evaluation_report_preproc = PropertyFile(
        name="EvaluationReportPreproc",
        output_name="evaluation-property-pass", # -- matches the processing output name
        path="evaluation.json",
    )

    job_arguments_p=["--input-data", f"s3://{default_bucket}/data/finance/combined_tweets.csv", 
                  "--data-size", "10000"]
    step_process = ProcessingStep(
            name="PreProcTweetsModelPipe",  # choose any name
            processor=sklearn_processor,
            inputs=inputs_p,
            outputs=outputs_p,
            property_files=[evaluation_report_preproc],
            code=os.path.join(BASE_DIR, "preprocess_tweets.py"),
            job_arguments=job_arguments_p,
            cache_config=cache_config
        )    

    print(f"SageMaker:pipeline:get_pipeline::Preproc:step:added={step_process}")







    import boto3
    from sagemaker.xgboost.estimator import XGBoost
    from sagemaker import TrainingInput

    # -- CANNOT USE this for Sagemaker Algorithims 
    metrics_definetion = [
            {'Name': 'train:loss', 'Regex': 'loss: ([0-9\\.]+)'},
            {'Name': 'train.accuracy', 'Regex': 'accuracy: ([0-9\\.]+)'},
            {'Name': 'validation.loss', 'Regex': 'val_loss: ([0-9\\.]+)'},
            {'Name': 'validation.accuracy', 'Regex': 'val_accuracy: ([0-9\\.]+)'},
    ]
    xgb_hyperparams = dict (
            objective="reg:linear",
            num_round=50,
            max_depth=5,
            eta=0.2,
            gamma=4,
            min_child_weight=6,
            subsample=0.7,
            silent=0,
        )

    use_spot_instances = True
    max_run = 3600
    max_wait = 7200 if use_spot_instances else None

    xgb_custom_estimator = XGBoost(
        role=role, 
        entry_point=os.path.join(BASE_DIR, 'modeltrain.py'),
        framework_version="1.3-1",
        instance_count=1,
        instance_type='ml.m5.large', # - 'local', only if docker is installed locally 
        output_path=f's3://{default_bucket}/pipeline/model/xgbtrain/modeltweet',
        use_spot_instances=use_spot_instances,
        max_run=max_run,
        max_wait=max_wait,
        hyperparameters=xgb_hyperparams,
        base_job_name=f"TrainTweetsModelPipe/{base_job_prefix}",
        code_location=f"s3://{default_bucket}/pipeline/model/xgbtrain/code", 
        #source_dir="scripts", # This line will tell SageMaker to first install defined dependencies from scrits/requirements.txt,
        # -- and then to upload all code inside of this folder to your container.
        #metric_definitions=metrics_definetion, # -- using XgBoost cannot override default SageMaker metrics

    )

    step_train = TrainingStep(
        name="TrainTweetsStep",
        estimator=xgb_custom_estimator,
        inputs={
            "train": TrainingInput( # -- name need to match output name of pre proc
                    s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                        "train" 
                    ].S3Output.S3Uri,
                    content_type="text/csv",
            ),
            "validation": TrainingInput(
                    s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                        "validation"
                    ].S3Output.S3Uri,
                    content_type="text/csv",
            ),
        },
        cache_config=cache_config
    )

    print(f"SageMaker:pipeline:get_pipeline::TRAINING:step:added={step_train}")






    # processing step for evaluation
    # -- FrameworkProcessor and XgBoostProcessor work best since we can do requirememts.txt in source_dir
    # -- SKLearnProcessor will not work since we need additonal libraries

    image_uri = sagemaker.image_uris.retrieve(
            framework="xgboost",  # we are using the Sagemaker built in xgboost algorithm
            region=region,
            version="1.3-1", #"1.0-1",
            py_version="py3",
            instance_type=training_instance_type,
    )

    est_cls = sagemaker.xgboost.estimator.XGBoost
    framework_version_str="1.3-1"
    framework_processor_eval = FrameworkProcessor( #  ScriptProcessor( #  FrameworkProcessor
            estimator_cls=est_cls,
            image_uri=image_uri,
            framework_version=framework_version_str,
            command=["python3"],
            instance_type=processing_instance_type,
            instance_count=1,
            base_job_name=f"script-tweets-eval/{base_job_prefix}",
            sagemaker_session=sm_session,
            role=role, 
    )
    run_args = framework_processor_eval.get_run_args(
        code="evaluate.py",#os.path.join(BASE_DIR,  "scripts_eval/evaluate.py"),
        source_dir=os.path.join(BASE_DIR,  "scripts_eval"),
        inputs=[
                ProcessingInput(
                    source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                    destination="/opt/ml/processing/model",
                ),
                ProcessingInput(
                    source=step_process.properties.ProcessingOutputConfig.Outputs[
                        "test"
                    ].S3Output.S3Uri,
                    destination="/opt/ml/processing/test",
                ),
        ],
        outputs=[
                ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
        ],
        arguments=None
    )
    evaluation_report = PropertyFile(
            name="TweetsEvaluationReport",
            output_name="evaluation",
            path="evaluation.json",
    )
    step_eval = ProcessingStep(
            name="EvaluateTweetsModelPipe",
            processor=framework_processor_eval,
            inputs=run_args.inputs,
            outputs=run_args.outputs,
            code=run_args.code,
            property_files=[evaluation_report],

    )
    print(f"SageMaker:pipeline:get_pipeline::EVALUATION:step:added={step_eval}")






    # register model step that will be conditionally executed
    model_metrics = ModelMetrics(
            model_statistics=MetricsSource(
                s3_uri="{}/evaluation.json".format(
                    step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
                ),
                content_type="application/json"
            )
    )
    model_tags = [
        {'Key': 'sagemaker:deployment-stage', 'Value': 'prod'},
        {'Key': 'sagemaker:short-description', 'Value': 'test-describe'},
        {'Key': 'sagemaker:project-name', 'Value': 'test-name'},
    ]
    ##  -----  TESTING Create Model froma pre ptrained and use that to host ---- ###
    # -- THIS MODEL has been trained in SM but different package and all  
    pretrained_s3="s3://sagemaker-grewaltempl/pipeline/model/xgbtrain/modeltweet/pipelines-vqlln8kv20ti-TrainTweetsStep-EG5BCPA1eB/output/model.tar.gz"
    xgboost_model = XGBoostModel(
        #model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        model_data=pretrained_s3, 
        entry_point='inference.py',
        source_dir=os.path.join(BASE_DIR,  'xgboost_source_dir'),
        #code_location=f"s3://{sagemaker_session.default_bucket()}/imlabs/pipeline/model/pipe_tweets/{base_job_prefix}",
        framework_version='1.3-1',
        py_version='py3',
        sagemaker_session=sm_session,
        role=role
    )
    step_create_xgboost_model = CreateModelStep(
        name="XGBoostFromSavedModel",
        model=xgboost_model,
        inputs=sagemaker.inputs.CreateModelInput(instance_type="ml.m4.large"),
    )    

    ##  ------  END TESTING PRE TRAINED MODEL ----------------------------------##
    
    step_register = RegisterModel(
            name="RegisterTweetsModel",
            estimator=xgb_custom_estimator,
            model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
            #model=xgboost_model,
            content_types=["text/csv"],
            response_types=["text/csv"],
            inference_instances=["ml.t2.medium", "ml.m5.large"],
            transform_instances=["ml.m5.large"],
            model_package_group_name=model_package_group_name,
            approval_status=model_approval_status,
            model_metrics=model_metrics,
            tags=model_tags,
            description="Test-Description",
    )


    cond_lte_register = ConditionGreaterThanOrEqualTo(  # You can change the condition here
            left=JsonGet(
                step=step_eval,
                #step_name=step_eval.name,#"EvaluateTweetsModel", # has to match the step evaluation name # old --step=step_process
                property_file=evaluation_report,
                json_path="regression_metrics.mse.value",  # This should follow the structure of your report_dict defined in the 
            ),
            right=0.01,  # You can change the threshold here
    )
    step_cond_register = ConditionStep(
            name="TweetsRegisterAccuracyCond",
            conditions=[cond_lte_register],
            if_steps=[step_register],
            else_steps=[],
    )
    print(f"Sagemaker:pipelines: Finally register:condition:step:created={step_cond_register}:")






    # pipeline instance
    pipeline = Pipeline(
            name=pipeline_name,
            parameters=[
                processing_instance_type,
                processing_instance_count,
                training_instance_type,
                model_approval_status,
                input_data,
            ],
            steps=[step_process, step_train, step_create_xgboost_model,step_eval, step_cond_register ],
            sagemaker_session=sm_session,
    )
    print(f"Finally Pipeline created={pipeline}:")


    return pipeline



