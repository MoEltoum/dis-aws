import os
import boto3
import subprocess


def launch_tensorboard(logdir):
    command = f"tensorboard --logdir={logdir}"
    process = subprocess.Popen(command, shell=True)
    process.wait()  # Wait for TensorBoard to finish (optional)


def get_s3_directory(s3_bucket, s3_prefix, local_dir):
    # Initialize S3 client
    s3 = boto3.client('s3')

    # Create local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    # List objects in the specified bucket and prefix
    response = s3.list_objects_v2(Bucket=s3_bucket, Prefix=s3_prefix)

    # Copy each object to the local directory
    for obj in response.get('Contents', []):
        key = obj ['Key']
        local_file_path = os.path.join(local_dir, os.path.basename(key))
        s3.download_file(bucket_name, key, local_file_path)
        print(f'Downloaded {key} to {local_file_path}')


# Specify the bucket and prefix (directory) containing the TensorBoard logs
bucket_name = 'dis-sagemaker'  # your S3 bucket name
prefix = 'fast-flow/experiment-2/outputs/tensorboard'  # your S3 tensorboard path

# Define the local directory where you want to copy the files
local_directory = os.path.join(os.path.join(os.getcwd(), "tensorboard"))  # your local directory path
# get s3 tensorboard events
get_s3_directory(bucket_name, prefix, local_directory)
# Launch TensorBoard
launch_tensorboard(local_directory)
