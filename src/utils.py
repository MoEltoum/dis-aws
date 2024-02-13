import gc
import os
import time
from pathlib import Path
import logging
from csv_logger import CsvLogger
import numpy as np
import wget
import pandas as pd
import torch.optim as optim
from skimage import io
from torch.autograd import Variable
import torch.nn as nn
from data_loader_cache import get_im_gt_name_dict, create_dataloaders, GOSRandomHFlip, GOSNormalize
from basics import f1_mae_torch
from torch.utils.tensorboard import SummaryWriter
from models.isnet import *

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------- 1: Model Utils ---------------


def get_gt_encoder(train_dataloaders, train_datasets, valid_dataloaders, valid_datasets, hypar, train_dataloaders_val,
                   train_datasets_val):
    torch.manual_seed(hypar ["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(hypar ["seed"])

    print("define gt encoder ...")
    net = ISNetGTEncoder()
    # load the existing model gt encoder
    if (hypar ["gt_encoder_model"] != ""):
        model_path = hypar ["model_path"] + "/" + hypar ["gt_encoder_model"]
        if torch.cuda.is_available():
            net.load_state_dict(torch.load(model_path))
            net.cuda()
        else:
            net.load_state_dict(torch.load(model_path, map_location="cpu"))
        print("gt encoder restored from the saved weights ...")
        return net

    if torch.cuda.is_available():
        net.cuda()

    print("--- define optimizer for GT Encoder---")
    optimizer = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    model_path = hypar ["model_path"]
    model_save_fre = hypar ["model_save_fre"]
    max_ite = hypar ["max_ite"]
    batch_size_train = hypar ["batch_size_train"]
    batch_size_valid = hypar ["batch_size_valid"]

    if (not os.path.exists(model_path)):
        os.mkdir(model_path)

    ite_num = hypar ["start_ite"]  # count the total iteration number
    ite_num4val = 0  #
    running_loss = 0.0  # count the total loss
    running_tar_loss = 0.0  # count the target output loss
    last_f1 = [0 for x in range(len(valid_dataloaders))]

    train_num = train_datasets [0].__len__()

    net.train()

    start_last = time.time()
    gos_dataloader = train_dataloaders [0]
    epoch_num = hypar ["max_epoch_num"]
    notgood_cnt = 0
    for epoch in range(epoch_num):  # set the epoch num as 100000

        for i, data in enumerate(gos_dataloader):

            if (ite_num >= max_ite):
                print("Training Reached the Maximal Iteration Number ", max_ite)
                exit()

            # start_read = time.time()
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            # get the inputs
            labels = data ['label']

            if (hypar ["model_digit"] == "full"):
                labels = labels.type(torch.FloatTensor)
            else:
                labels = labels.type(torch.HalfTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                labels_v = Variable(labels.cuda(), requires_grad=False)
            else:
                labels_v = Variable(labels, requires_grad=False)

            # y zero the parameter gradients
            start_inf_loss_back = time.time()
            optimizer.zero_grad()

            ds, fs = net(labels_v)  # net(inputs_v)
            loss2, loss = net.compute_loss(ds, labels_v)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_tar_loss += loss2.item()

            # del outputs, loss
            del ds, loss2, loss
            end_inf_loss_back = time.time() - start_inf_loss_back

            print("GT Encoder Training>>>" + model_path.split('/') [
                -1] + " - [epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f, time-per-iter: %3f s, time_read: %3f" % (
                      epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val,
                      running_tar_loss / ite_num4val, time.time() - start_last,
                      time.time() - start_last - end_inf_loss_back))
            start_last = time.time()

            if ite_num % model_save_fre == 0:  # validate every 2000 iterations
                notgood_cnt += 1
                tmp_f1, tmp_mae, val_loss, tar_loss, i_val, tmp_time = valid_gt_encoder(net, train_dataloaders_val,
                                                                                        train_datasets_val, hypar,
                                                                                        epoch)

                net.train()  # resume train

                tmp_out = 0
                print("last_f1:", last_f1)
                print("tmp_f1:", tmp_f1)
                for fi in range(len(last_f1)):
                    if (tmp_f1 [fi] > last_f1 [fi]):
                        tmp_out = 1
                print("tmp_out:", tmp_out)
                if (tmp_out):
                    notgood_cnt = 0
                    last_f1 = tmp_f1
                    tmp_f1_str = [str(round(f1x, 4)) for f1x in tmp_f1]
                    tmp_mae_str = [str(round(mx, 4)) for mx in tmp_mae]
                    maxf1 = '_'.join(tmp_f1_str)
                    meanM = '_'.join(tmp_mae_str)
                    # .cpu().detach().numpy()
                    model_name = "/GTENCODER-gpu_itr_" + str(ite_num) + \
                                 "_traLoss_" + str(np.round(running_loss / ite_num4val, 4)) + \
                                 "_traTarLoss_" + str(np.round(running_tar_loss / ite_num4val, 4)) + \
                                 "_valLoss_" + str(np.round(val_loss / (i_val + 1), 4)) + \
                                 "_valTarLoss_" + str(np.round(tar_loss / (i_val + 1), 4)) + \
                                 "_maxF1_" + maxf1 + \
                                 "_mae_" + meanM + \
                                 "_time_" + str(np.round(np.mean(np.array(tmp_time)) / batch_size_valid, 6)) + ".pth"
                    torch.save(net.state_dict(), model_path + model_name)

                running_loss = 0.0
                running_tar_loss = 0.0
                ite_num4val = 0

                if (tmp_f1 [0] > 0.99):
                    print("GT encoder is well-trained and obtained...")
                    return net

                if (notgood_cnt >= hypar ["early_stop"]):
                    print("No improvements in the last " + str(
                        notgood_cnt) + " validation periods, so training stopped !")
                    exit()

    print("Training Reaches The Maximum Epoch Number")
    return net


def valid_gt_encoder(net, valid_dataloaders, valid_datasets, hypar, epoch=0):
    net.eval()
    print("Validating...")
    epoch_num = hypar ["max_epoch_num"]

    val_loss = 0.0
    tar_loss = 0.0

    tmp_f1 = []
    tmp_mae = []
    tmp_time = []

    start_valid = time.time()
    for k in range(len(valid_dataloaders)):

        valid_dataloader = valid_dataloaders [k]
        valid_dataset = valid_datasets [k]

        val_num = valid_dataset.__len__()
        mybins = np.arange(0, 256)
        PRE = np.zeros((val_num, len(mybins) - 1))
        REC = np.zeros((val_num, len(mybins) - 1))
        F1 = np.zeros((val_num, len(mybins) - 1))
        MAE = np.zeros((val_num))

        val_cnt = 0.0
        i_val = None

        for i_val, data_val in enumerate(valid_dataloader):

            imidx_val, labels_val, shapes_val = data_val ['imidx'], data_val ['label'], data_val ['shape']

            if (hypar ["model_digit"] == "full"):
                labels_val = labels_val.type(torch.FloatTensor)
            else:
                labels_val = labels_val.type(torch.HalfTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                labels_val_v = Variable(labels_val.cuda(), requires_grad=False)
            else:
                labels_val_v = Variable(labels_val, requires_grad=False)

            t_start = time.time()
            ds_val = net(labels_val_v) [0]
            t_end = time.time() - t_start
            tmp_time.append(t_end)

            loss2_val, loss_val = net.compute_loss(ds_val, labels_val_v)

            # compute F measure
            for t in range(hypar ["batch_size_valid"]):
                val_cnt = val_cnt + 1.0
                print("num of val: ", val_cnt)
                i_test = imidx_val [t].data.numpy()

                pred_val = ds_val [0] [t, :, :, :]  # B x 1 x H x W

                # recover the prediction spatial size to the orignal image size
                pred_val = torch.squeeze(
                    F.upsample(torch.unsqueeze(pred_val, 0), (shapes_val [t] [0], shapes_val [t] [1]), mode='bilinear'))

                ma = torch.max(pred_val)
                mi = torch.min(pred_val)
                pred_val = (pred_val - mi) / (ma - mi)  # max = 1

                gt = np.squeeze(io.imread(valid_dataset.dataset ["ori_gt_path"] [i_test]))  # max = 255
                with torch.no_grad():
                    gt = torch.tensor(gt).to(device)

                pre, rec, f1, mae = f1_mae_torch(pred_val * 255, gt, valid_dataset, i_test, mybins, hypar)

                PRE [i_test, :] = pre
                REC [i_test, :] = rec
                F1 [i_test, :] = f1
                MAE [i_test] = mae

            del ds_val, gt
            gc.collect()
            torch.cuda.empty_cache()

            # if(loss_val.data[0]>1):
            val_loss += loss_val.item()  # data[0]
            tar_loss += loss2_val.item()  # data[0]

            print("[validating: %5d/%5d] val_ls:%f, tar_ls: %f, f1: %f, mae: %f, time: %f" % (
                i_val, val_num, val_loss / (i_val + 1), tar_loss / (i_val + 1), np.amax(F1 [i_test, :]), MAE [i_test],
                t_end))

            del loss2_val, loss_val

        print('============================')
        PRE_m = np.mean(PRE, 0)
        REC_m = np.mean(REC, 0)
        f1_m = (1 + 0.3) * PRE_m * REC_m / (0.3 * PRE_m + REC_m + 1e-8)
        # print('--------------:', np.mean(f1_m))
        tmp_f1.append(np.amax(f1_m))
        tmp_mae.append(np.mean(MAE))
        print("The max F1 Score: %f" % (np.max(f1_m)))
        print("MAE: ", np.mean(MAE))

    return tmp_f1, tmp_mae, val_loss, tar_loss, i_val, tmp_time


def train(net, optimizer, train_dataloaders, train_datasets, valid_dataloaders, valid_datasets, hypar,
          train_dataloaders_val, train_datasets_val, logs_path, tensorboard_path):

    # train logs
    csvlogger = CsvLogger(filename=logs_path,
                          delimiter=',',
                          level=logging.INFO,
                          fmt=f'%(asctime)s{","}%(message)s',
                          datefmt='%Y/%m/%d , %H:%M:%S',
                          header=['date', 'time', 'epoch', 'batch', 'ite', 'train_loss', 'tar', 'sec_per_iter'])

    # tensorboard path
    writer = SummaryWriter(tensorboard_path)

    if hypar ["interm_sup"]:
        print("Get the gt encoder ...")
        featurenet = get_gt_encoder(train_dataloaders, train_datasets, valid_dataloaders, valid_datasets, hypar,
                                    train_dataloaders_val, train_datasets_val)
        # freeze the weights of gt encoder
        for param in featurenet.parameters():
            param.requires_grad = False

    model_path = hypar ["model_path"]
    model_save_fre = hypar ["model_save_fre"]
    max_ite = hypar ["max_ite"]
    batch_size_train = hypar ["batch_size_train"]
    batch_size_valid = hypar ["batch_size_valid"]

    if (not os.path.exists(model_path)):
        os.mkdir(model_path)

    ite_num = hypar ["start_ite"]  # count the total iteration number
    ite_num4val = 0  #
    running_loss = 0.0  # count the toal loss
    running_tar_loss = 0.0  # count the target output loss
    last_f1 = [0 for x in range(len(valid_dataloaders))]

    train_num = train_datasets [0].__len__()

    net.train()

    start_last = time.time()
    gos_dataloader = train_dataloaders [0]
    epoch_num = hypar ["max_epoch_num"]
    notgood_cnt = 0
    for epoch in range(epoch_num):  # set the epoch num as 100000

        for i, data in enumerate(gos_dataloader):

            if (ite_num >= max_ite):
                print("Training Reached the Maximal Iteration Number ", max_ite)

            # start_read = time.time()
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            # get the inputs
            inputs, labels = data ['image'], data ['label']

            if (hypar ["model_digit"] == "full"):
                inputs = inputs.type(torch.FloatTensor)
                labels = labels.type(torch.FloatTensor)
            else:
                inputs = inputs.type(torch.HalfTensor)
                labels = labels.type(torch.HalfTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                            requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

            # y zero the parameter gradients
            start_inf_loss_back = time.time()
            optimizer.zero_grad()

            if hypar ["interm_sup"]:
                # forward + backward + optimize
                ds, dfs = net(inputs_v)
                _, fs = featurenet(labels_v)  ## extract the gt encodings
                loss2, loss = net.compute_loss_kl(ds, labels_v, dfs, fs, mode='MSE')
            else:
                # forward + backward + optimize
                ds, _ = net(inputs_v)
                loss2, loss = net.compute_loss(ds, labels_v)

            # writer.add_scalar("Loss/train", loss, epoch)

            loss.backward()
            optimizer.step()

            # # print statistics
            running_loss += loss.item()
            running_tar_loss += loss2.item()

            # del outputs, loss
            del ds, loss2, loss
            end_inf_loss_back = time.time() - start_inf_loss_back

            print(">>>" + model_path.split('/') [
                -1] + " - [epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f, time-per-iter: %3f s, time_read: %3f" % (
                      epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val,
                      running_tar_loss / ite_num4val, time.time() - start_last,
                      time.time() - start_last - end_inf_loss_back))

            csvlogger.info("%3d, %5d, %d, %3f, %3f, %3f s" % (
                epoch + 1, (i + 1) * batch_size_train, ite_num, running_loss / ite_num4val,
                running_tar_loss / ite_num4val, time.time() - start_last))

            start_last = time.time()

            if ite_num % model_save_fre == 0:  # validate every 2000 iterations
                notgood_cnt += 1
                net.eval()
                tmp_f1, tmp_mae, val_loss, tar_loss, i_val, tmp_time = valid(net, valid_dataloaders, valid_datasets,
                                                                             hypar, epoch)
                net.train()  # resume train

                tmp_out = 0
                print("last_f1:", last_f1)
                print("tmp_f1:", tmp_f1)
                for fi in range(len(last_f1)):
                    if (tmp_f1 [fi] > last_f1 [fi]):
                        tmp_out = 1
                print("tmp_out:", tmp_out)

                if (tmp_out):
                    notgood_cnt = 0
                    last_f1 = tmp_f1
                    tmp_f1_str = [str(round(f1x, 4)) for f1x in tmp_f1]
                    tmp_mae_str = [str(round(mx, 4)) for mx in tmp_mae]
                    maxf1 = '_'.join(tmp_f1_str)
                    meanM = '_'.join(tmp_mae_str)
                    # .cpu().detach().numpy()
                    model_name = "/gpu_itr_" + str(ite_num) + \
                                 "_traLoss_" + str(np.round(running_loss / ite_num4val, 4)) + \
                                 "_traTarLoss_" + str(np.round(running_tar_loss / ite_num4val, 4)) + \
                                 "_valLoss_" + str(np.round(val_loss / (i_val + 1), 4)) + \
                                 "_valTarLoss_" + str(np.round(tar_loss / (i_val + 1), 4)) + \
                                 "_maxF1_" + maxf1 + \
                                 "_mae_" + meanM + \
                                 "_time_" + str(np.round(np.mean(np.array(tmp_time)) / batch_size_valid, 6)) + ".pth"
                    torch.save(net.state_dict(), model_path + model_name)

                    # writer.add_scalars('dis',
                    #                    {'TrainLoss': np.round(running_loss / ite_num4val, 4),
                    #                     'TrainTarLoss': np.round(running_tar_loss / ite_num4val, 4),
                    #                     'ValLoss': np.round(val_loss / (i_val + 1), 4),
                    #                     'ValTarLoss': np.round(tar_loss / (i_val + 1), 4),
                    #                     'F1': round(tmp_f1[0], 4),
                    #                     'MAE': round(tmp_mae[0], 4)
                    #                     }, ite_num)
                    writer.add_scalars('train',
                                       {'Loss': np.round(running_loss / ite_num4val, 4),
                                        'TarLoss': np.round(running_tar_loss / ite_num4val, 4)}, ite_num)
                    writer.add_scalars('valid',
                                       {'Loss': np.round(val_loss / (i_val + 1), 4),
                                        'TarLoss': np.round(tar_loss / (i_val + 1), 4)}, ite_num)
                    writer.add_scalars('F1_score', {'F1': round(tmp_f1[0], 4)}, ite_num)
                    writer.add_scalars('MAE', {'mae': round(tmp_mae[0], 4)}, ite_num)

                running_loss = 0.0
                running_tar_loss = 0.0
                ite_num4val = 0

                if (notgood_cnt >= hypar ["early_stop"]):
                    print("No improvements in the last " + str(
                        notgood_cnt) + " validation periods, so training stopped !")
                    logging.info("No improvements in the last " + str(
                        notgood_cnt) + " validation periods, so training stopped !")
                    exit()

    print("Training Reaches The Maximum Epoch Number")


def valid(net, valid_dataloaders, valid_datasets, hypar, epoch=0):
    net.eval()
    print("Validating...")
    epoch_num = hypar ["max_epoch_num"]

    val_loss = 0.0
    tar_loss = 0.0
    val_cnt = 0.0

    tmp_f1 = []
    tmp_mae = []
    tmp_time = []

    start_valid = time.time()

    for k in range(len(valid_dataloaders)):

        valid_dataloader = valid_dataloaders [k]
        valid_dataset = valid_datasets [k]

        val_num = valid_dataset.__len__()
        mybins = np.arange(0, 256)
        PRE = np.zeros((val_num, len(mybins) - 1))
        REC = np.zeros((val_num, len(mybins) - 1))
        F1 = np.zeros((val_num, len(mybins) - 1))
        MAE = np.zeros((val_num))

        for i_val, data_val in enumerate(valid_dataloader):
            val_cnt = val_cnt + 1.0
            imidx_val, inputs_val, labels_val, shapes_val = data_val ['imidx'], data_val ['image'], data_val ['label'], \
                data_val ['shape']

            if (hypar ["model_digit"] == "full"):
                inputs_val = inputs_val.type(torch.FloatTensor)
                labels_val = labels_val.type(torch.FloatTensor)
            else:
                inputs_val = inputs_val.type(torch.HalfTensor)
                labels_val = labels_val.type(torch.HalfTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_val_v, labels_val_v = Variable(inputs_val.cuda(), requires_grad=False), Variable(
                    labels_val.cuda(), requires_grad=False)
            else:
                inputs_val_v, labels_val_v = Variable(inputs_val, requires_grad=False), Variable(labels_val,
                                                                                                 requires_grad=False)

            t_start = time.time()
            ds_val = net(inputs_val_v) [0]
            t_end = time.time() - t_start
            tmp_time.append(t_end)

            loss2_val, loss_val = net.compute_loss(ds_val, labels_val_v)

            # compute F measure
            for t in range(hypar ["batch_size_valid"]):
                i_test = imidx_val [t].data.numpy()

                pred_val = ds_val [0] [t, :, :, :]  # B x 1 x H x W

                # recover the prediction spatial size to the orignal image size
                pred_val = torch.squeeze(
                    F.upsample(torch.unsqueeze(pred_val, 0), (shapes_val [t] [0], shapes_val [t] [1]), mode='bilinear'))

                # pred_val = normPRED(pred_val)
                ma = torch.max(pred_val)
                mi = torch.min(pred_val)
                pred_val = (pred_val - mi) / (ma - mi)  # max = 1

                if len(valid_dataset.dataset ["ori_gt_path"]) != 0:
                    gt = np.squeeze(io.imread(valid_dataset.dataset ["ori_gt_path"] [i_test]))  # max = 255
                else:
                    gt = np.zeros((shapes_val [t] [0], shapes_val [t] [1]))
                with torch.no_grad():
                    gt = torch.tensor(gt).to(device)

                pre, rec, f1, mae = f1_mae_torch(pred_val * 255, gt, valid_dataset, i_test, mybins, hypar)

                PRE [i_test, :] = pre
                REC [i_test, :] = rec
                F1 [i_test, :] = f1
                MAE [i_test] = mae

                del ds_val, gt
                gc.collect()
                torch.cuda.empty_cache()

            # if(loss_val.data[0]>1):
            val_loss += loss_val.item()  # data[0]
            tar_loss += loss2_val.item()  # data[0]

            print("[validating: %5d/%5d] val_ls:%f, tar_ls: %f, f1: %f, mae: %f, time: %f" % (
                i_val, val_num, val_loss / (i_val + 1), tar_loss / (i_val + 1), np.amax(F1 [i_test, :]), MAE [i_test],
                t_end))

            del loss2_val, loss_val

        print('============================')
        PRE_m = np.mean(PRE, 0)
        REC_m = np.mean(REC, 0)
        f1_m = (1 + 0.3) * PRE_m * REC_m / (0.3 * PRE_m + REC_m + 1e-8)

        tmp_f1.append(np.amax(f1_m))
        tmp_mae.append(np.mean(MAE))

    return tmp_f1, tmp_mae, val_loss, tar_loss, i_val, tmp_time


def outputs_dir():
    outputs = os.path.join(os.getcwd(), 'outputs')

    if not os.path.exists(outputs):
        saved_weights = os.path.join(outputs, 'saved_weights')
        tensorboard_dir = os.path.join(outputs, 'tensorboard')
        os.mkdir(outputs)
        os.mkdir(saved_weights)
        os.mkdir(tensorboard_dir)
    print('Outputs directory: ', outputs)
    return outputs


# --------------- 2: Data Utils ---------------

def create_dirs():
    # create directories
    cwd = os.getcwd()
    Path(os.path.sep.join([cwd, "data"])).mkdir(parents=True, exist_ok=True)
    data_dir = os.path.join(cwd, 'data')

    Path(os.path.sep.join([data_dir, "train"])).mkdir(parents=True, exist_ok=True)
    train_dir = os.path.join(data_dir, 'train')
    # create train cache folder
    Path(os.path.sep.join([train_dir, "cache"])).mkdir(parents=True, exist_ok=True)

    Path(os.path.sep.join([data_dir, "val"])).mkdir(parents=True, exist_ok=True)
    val_dir = os.path.join(data_dir, 'val')
    # create val cache folder
    Path(os.path.sep.join([val_dir, "cache"])).mkdir(parents=True, exist_ok=True)

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
    # split data 70% train, 20% val, and 10% test
    train_data = data [:int(len(data) * 0.7)]
    val_data = data [int(len(data) * 0.7):int(len(data) * 0.9)]
    test_data = data [int(len(data) * 0.9):]
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
    raw_data = pd.read_csv(csv_data)[:50]
    # clean_data = clean_data_player(raw_data, player_ids)
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


def create_inputs(train_dir, val_dir):
    dataset_train = {"name": 'TRAIN-DATA',
                     "im_dir": os.path.join(train_dir, 'im'),
                     "gt_dir": os.path.join(train_dir, 'gt'),
                     "im_ext": ".jpg",
                     "gt_ext": ".png",
                     "cache_dir": os.path.join(train_dir, 'cache')}

    dataset_val = {"name": 'VAL-DATA',
                   "im_dir": os.path.join(val_dir, 'im'),
                   "gt_dir": os.path.join(val_dir, 'gt'),
                   "im_ext": ".jpg",
                   "gt_ext": ".png",
                   "cache_dir": os.path.join(val_dir, 'cache')}

    train_datasets = [dataset_train]
    valid_datasets = [dataset_val]

    return train_datasets, valid_datasets
