# Use an official PyTorch runtime as a parent image
#FROM pytorch/pytorch:1.8.1-cuda11.1-runtime
FROM python:3.9-slim

# Set the working directory to /opt/ml/code
WORKDIR /opt/ml/code

# Copy the current directory contents into the container at /opt/ml/code
COPY . /opt/ml/code

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Define environment variable
ENV SAGEMAKER_PROGRAM train.py

# Run train_segmentation.py when the container launches
ENTRYPOINT ["python", "train.py"]
