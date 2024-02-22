# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

RUN apt-get update && apt-get install -y \
	python3-pip software-properties-common wget && \
	rm -rf /var/lib/apt/lists/* \

# Set the working directory to /opt/ml/code
WORKDIR /opt/ml/code

# Copy the current directory contents into the container at /opt/ml/code
COPY . /opt/ml/code
#COPY code/train.py /opt/ml/code/train.py
# Install any needed packages specified in requirements.txt
RUN pip --no-cache-dir install --upgrade pip

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Define environment variable
#ENV SAGEMAKER_PROGRAM train.py

# Run train_segmentation.py when the container launches
ENTRYPOINT ["python", "/opt/ml/code/train.py"]

