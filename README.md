## DIS SageMaker

### Overview
In this repository, we use [Amazon SageMaker](https://aws.amazon.com/sagemaker) to build and train DIS model using PyTorch Estimator.

This implementation shows how to do the following:
* Fine-tune DIS model with PyTorch on Amazon SageMaker
* Monitor your model training with Mlflow


### Prerequisites
If you want to try out each step yourself, make sure that you have the following in place:
* An AWS account
* An Amazon Simple Storage Service [(Amazon S3) bucket](https://aws.amazon.com/s3/)
* A running [SageMaker notebook instance](https://docs.aws.amazon.com/sagemaker/latest/dg/nbi.html)

### Setup environment
* Amazon SageMaker Studio Lab environment comes with a base image installed that includes key packages and resources, and it uses conda environments to encapsulate the software packages that are needed to run notebooks, which makes it easy to setup our environment with one click.
* To set up the required environment from the notebook kernel select `conda_pytorch_p310`



### Get started
1) Clone this repository into your Amazon SageMaker notebook instance:

2) Upload your data into an S3 bucket, you can use the provided script for that `data_prep.py` .

3) Follow the step-by-step guide by executing the `train.ipynb` notebook.


### Monitor model training jobs with Mlflow
There are only two steps needed to enable mlflow on your local machine for live training monitoring which are:

1) Sync the S3 bucket ml runs folder with your local by running every minute by running: `watch -n 60 aws s3 sync s3://your-bucket-name/path/to/s3-folder /path/to/local-folder` .

2) CD to the local folder from step 1 and run : `mlflow ui` .

