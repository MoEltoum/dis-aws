import os
import boto3
import subprocess

def launch_tensorboard(logdir):
    command = f"tensorboard --logdir={logdir}"
    process = subprocess.Popen(command, shell=True)
    process.wait()  # Wait for TensorBoard to finish (optional)


# Initialize S3 client
s3 = boto3.client('s3')

# Specify the bucket and prefix (directory) containing the TensorBoard logs
bucket_name = 'dis-sagemaker'
prefix = 'fast-flow/experiment-2/outputs/tensorboard'

# Define the local directory where you want to copy the files
local_directory = '/home/aos-mo/Documents/ML_models/dis/tensorboard'

# Create local directory if it doesn't exist
os.makedirs(local_directory, exist_ok=True)

# List objects in the specified bucket and prefix
response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

# Copy each object to the local directory
for obj in response.get('Contents', []):
    key = obj['Key']
    local_file_path = os.path.join(local_directory, os.path.basename(key))
    s3.download_file(bucket_name, key, local_file_path)
    print(f'Downloaded {key} to {local_file_path}')

# Launch TensorBoard
launch_tensorboard(local_directory)
