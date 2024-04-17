import os
from pathlib import Path
import wget
import pandas as pd
import boto3


def create_dirs():
    # create directories
    cwd = os.getcwd()
    Path(os.path.sep.join([cwd, "data"])).mkdir(parents=True, exist_ok=True)
    data_dir = os.path.join(cwd, 'data')

    Path(os.path.sep.join([data_dir, "train"])).mkdir(parents=True, exist_ok=True)
    train_dir = os.path.join(data_dir, 'train')

    Path(os.path.sep.join([data_dir, "val"])).mkdir(parents=True, exist_ok=True)
    val_dir = os.path.join(data_dir, 'val')

    Path(os.path.sep.join([data_dir, "test"])).mkdir(parents=True, exist_ok=True)
    test_dir = os.path.join(data_dir, 'test')

    Path(os.path.sep.join([data_dir, "data_logs"])).mkdir(parents=True, exist_ok=True)
    logs_dir = os.path.join(data_dir, "data_logs")

    return train_dir, val_dir, test_dir, logs_dir


def add_unique_id_brand(data, last_count=1):
    # add unique id
    id_list = []
    for x in range(last_count, len(data) + 1):
        id_list.append(x)

    data ['unique_id'] = id_list

    # get brand
    brand_list = []
    img_list = data ['source_image_url'].to_list()
    for i in img_list:
        if len(i.split('__')) > 1:
            brand_lst = i.split('/') [-1].split('__') [:2]
            brand = '_'.join(brand_lst)
        else:
            brand = 'Unknown'
        brand_list.append(brand)
    data ['brand'] = brand_list

    # get colour
    colour_list = []
    img_list = data ['source_image_url'].to_list()
    for i in img_list:
        if len(i.split('__')) > 1:
            colour = i.split('/') [-1].split('__') [4]
        else:
            colour = 'Unknown'
        colour_list.append(colour)
    data ['colour'] = colour_list

    return data


def data_logs(data, data_dir):
    with open(os.path.join(data_dir, "data_logs.txt"), "w") as f:
        print('Number of total images:  ', len(data), file=f)
        print('\n', file=f)
        # last_id = data['unique_id'].max()
        # print('The latest ID count: ', last_id, file=f)
        # print('\n', file=f)

        vehicle_count = data ['reg'].nunique()
        print('Number of Vehicles : ', vehicle_count, file=f)
        print('\n', file=f)

        brand_count = dict(data ['brand'].value_counts())
        # creating a dictionary of brand type value counts
        y = list(brand_count.keys())
        print('Number of brands/classes: ', len(y), file=f)
        print('\n', file=f)
        print('Number of images per class : ', file=f)
        print("{:<35} {:<15}".format('Class', 'Images'), file=f)
        for k, v in brand_count.items():
            print("{:<35} {:<15}".format(k, v), file=f)

        # colour logs
        colour_count = dict(data ['colour'].value_counts())
        # creating a dictionary of colour type value counts
        print('\n', file=f)
        print('Number of images per colour : ', file=f)
        print("{:<35} {:<15}".format('Colour', 'Images'), file=f)
        for kk, vv in colour_count.items():
            print("{:<35} {:<15}".format(kk, vv), file=f)

    return f


def split_data(data):
    # split data 90% train, 8% val, and 2% test
    train_data = data [:int(len(data) * 0.90)]
    val_data = data [int(len(data) * 0.90):int(len(data) * 0.98)]
    test_data = data [int(len(data) * 0.98):]
    return train_data, val_data, test_data


def download_data(in_data, out_dir):
    # download URL data
    # create image and mask directories
    Path(os.path.sep.join([out_dir, "im"])).mkdir(parents=True, exist_ok=True)  # images dir
    Path(os.path.sep.join([out_dir, "gt"])).mkdir(parents=True, exist_ok=True)  # masks dir
    images_dir = os.path.join(out_dir, 'im')
    masks_dir = os.path.join(out_dir, 'gt')

    images = in_data ['source_image_url']
    masks = in_data ['mask_url']

    # download images
    for i in images:
        image_name = str(in_data.loc [in_data ['source_image_url'] == i, 'unique_id'].iloc [0]) + ".jpg"
        print('Downloading Image #: ', image_name)
        try:
            wget.download(i, os.path.join(images_dir, image_name))
        except:
            pass
    print('Images download complete')
    # download masks
    for i in masks:
        mask_name = str(in_data.loc [in_data ['mask_url'] == i, 'unique_id'].iloc [0]) + ".jpg"
        print('Downloading mask #: ', mask_name)
        try:
            wget.download(i, os.path.join(masks_dir, mask_name))
        except:
            pass
    print('Masks download complete')


def rename_mask_ext(msk_dir):
    for msk_name in os.listdir(msk_dir):
        in_filename = os.path.join(msk_dir, msk_name)
        if not os.path.isfile(in_filename): continue
        os.path.splitext(msk_name)
        newname = in_filename.replace('.jpg', '.png')
        os.rename(in_filename, newname)


def filter_data(data_dir):
    # define images and masks directories
    images_dir = os.path.join(data_dir, 'im')
    masks_dir = os.path.join(data_dir, 'gt')
    images_path = os.listdir(images_dir)
    masks_path = os.listdir(masks_dir)

    # check quantity and remove unmatch images
    if len(images_path) == len(masks_path):
        print('Download quantity matched')
        print(len(images_path), ' images saved')
        print(len(masks_path), ' masks saved')
        rename_mask_ext(masks_dir)
    elif len(images_path) != len(masks_path):
        for i, j in zip(images_path, masks_path):
            if i not in masks_path:
                i = os.path.join(images_dir, i)
                os.remove(i)
            elif j not in images_path:
                j = os.path.join(masks_dir, j)
                os.remove(j)
        rename_mask_ext(masks_dir)
        print('Data filtration completed')
        print(len(images_path), ' images saved')
        print(len(masks_path), ' masks saved')


def data_prep(csv_data):
    # step one data prep
    print('[INFO]: Start raw data preprocessing ...')
    train_dir, val_dir, test_dir, log_dir = create_dirs()
    raw_data = pd.read_csv(csv_data)
    data = add_unique_id_brand(raw_data)

    # logging preprocessed data
    data_logs(data, log_dir)

    print('[INFO]: Data preprocessing completed')
    print('\n')

    # split and download data
    train_data, val_data, test_data = split_data(data)

    # download and filter data
    print('[INFO]: Start downloading data')
    # download training data
    print('[INFO]: Downloading training data')
    download_data(train_data, train_dir)
    # filter training data
    print('[INFO]: Filtering training data')
    filter_data(train_dir)
    # download validation data
    print('\n')
    print('[INFO]: Downloading validation data')
    download_data(val_data, val_dir)
    # filter validation data
    print('[INFO]: Filtering validation data')
    filter_data(val_dir)
    # download testing data
    print('\n')
    print('[INFO]: Downloading testing data')
    download_data(test_data, test_dir)
    # filter testing data
    print('[INFO]: Filtering testing data')
    filter_data(test_dir)
    print('\n')
    print('[INFO]: Data Prep completed')
    return train_dir, val_dir


def upload_folder_to_s3(local_folder_path, s3_bucket_name, s3_folder_path):
    s3 = boto3.client('s3')

    for root, dirs, files in os.walk(local_folder_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_file_path, local_folder_path)
            # Convert backslashes to forward slashes
            s3_key = os.path.join(s3_folder_path, relative_path).replace("\\", "/")
            s3.upload_file(local_file_path, s3_bucket_name, s3_key)

def download_folder_from_s3(bucket_name, s3_folder, local_folder):
    s3_client = boto3.client('s3')

    # Create the local folder if it doesn't exist
    os.makedirs(local_folder, exist_ok=True)

    # List objects in the S3 folder recursively
    paginator = s3_client.get_paginator('list_objects_v2')
    operation_parameters = {'Bucket': bucket_name, 'Prefix': s3_folder}
    page_iterator = paginator.paginate(**operation_parameters)

    for page in page_iterator:
        if 'Contents' in page:
            for obj in page ['Contents']:
                # Construct the local file path
                local_file_path = os.path.join(local_folder, os.path.relpath(obj ['Key'], s3_folder))

                # Create any necessary local directories
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

                # Download the object
                s3_client.download_file(bucket_name, obj ['Key'], local_file_path)

if __name__ == "__main__":
    # set up required inputs
    raw_data_path = os.path.join(os.getcwd(), "raw_data.csv")  # your raw CSV data path
    s3_bucket = 'dis-sagemaker'  # Your S3 bucket where data will be uploaded
    s3_data_prefix = 'fast-flow/data_test'  # Your S3 data path

    # prepare and download data
    # data_prep(raw_data_path)

    # upload data to S3 bucket
    local_path = os.path.join(os.getcwd(), "data")
    # upload_folder_to_s3(local_path, s3_bucket, s3_data_prefix)

    # download data from S3
    download_folder_from_s3(s3_bucket, s3_data_prefix, local_path)
