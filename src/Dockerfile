# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# Install Python 3.8
RUN apt-get update && apt-get install -y python3.8 python3.8-dev python3-pip && \
    ln -s /usr/bin/python3.8 /usr/bin/python3

# Update pip
RUN python3 -m pip install --upgrade pip

# Set the working directory to /opt/ml/code
WORKDIR /opt/ml/code

# Copy the current directory contents into the container at /opt/ml/code
COPY . /opt/ml/code

# Install any needed packages specified in requirements.txt
RUN pip --no-cache-dir install --upgrade pip

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run train_segmentation.py when the container launches
ENTRYPOINT ["python3.8", "/opt/ml/code/train.py"]