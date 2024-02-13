from utils import *
import argparse


def main(hypar):
    # --- Step 0: Build datasets ---
    train_datasets, valid_datasets = create_inputs(hypar ["train_inputs"], hypar ["val_inputs"])

    # --- Step 1: Build dataloaders ---
    print("--- create training dataloader ---")
    # collect training dataset
    train_nm_im_gt_list = get_im_gt_name_dict(train_datasets, flag="train")
    # build dataloader for training datasets
    train_dataloaders, train_datasets = create_dataloaders(train_nm_im_gt_list,
                                                           cache_size=hypar ["cache_size"],
                                                           cache_boost=hypar ["cache_boost_train"],
                                                           my_transforms=[
                                                               GOSRandomHFlip(),
                                                               GOSNormalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0]),
                                                           ],
                                                           batch_size=hypar ["batch_size_train"],
                                                           shuffle=True)
    train_dataloaders_val, train_datasets_val = create_dataloaders(train_nm_im_gt_list,
                                                                   cache_size=hypar ["cache_size"],
                                                                   cache_boost=hypar ["cache_boost_train"],
                                                                   my_transforms=[
                                                                       GOSNormalize([0.5, 0.5, 0.5],
                                                                                    [1.0, 1.0, 1.0]),
                                                                   ],
                                                                   batch_size=hypar ["batch_size_valid"],
                                                                   shuffle=False)
    print(len(train_dataloaders), " train dataloaders created")

    print("--- create valid dataloader ---")
    # build dataloader for validation or testing
    valid_nm_im_gt_list = get_im_gt_name_dict(valid_datasets, flag="valid")
    # build dataloader for training datasets
    valid_dataloaders, valid_datasets = create_dataloaders(valid_nm_im_gt_list,
                                                           cache_size=hypar ["cache_size"],
                                                           cache_boost=hypar ["cache_boost_valid"],
                                                           my_transforms=[
                                                               GOSNormalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0]),
                                                               # GOSResize(hypar["input_size"])
                                                           ],
                                                           batch_size=hypar ["batch_size_valid"],
                                                           shuffle=False)
    print(len(valid_dataloaders), " valid dataloaders created")

    # --- Step 2: Build Model and Optimizer ---
    print("--- build model ---")
    net = hypar ["model"]

    # convert to half precision
    if (hypar ["model_digit"] == "half"):
        net.half()
        for layer in net.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()

    if torch.cuda.is_available():
        net.cuda()

    if (hypar ["restore_model"] != ""):
        print("restore model from:")
        print(hypar ["model_path"] + "/" + hypar ["restore_model"])
        if torch.cuda.is_available():
            net.load_state_dict(torch.load(hypar ["model_path"] + "/" + hypar ["restore_model"]))
        else:
            net.load_state_dict(torch.load(hypar ["model_path"] + "/" + hypar ["restore_model"], map_location="cpu"))

    print("--- define optimizer ---")
    optimizer = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    # --- Step 3: checking outputs directory ---
    # outputs_dir()

    # --- Step 4: Train and Valid Model ---

    train(net,
          optimizer,
          train_dataloaders,
          train_datasets,
          valid_dataloaders,
          valid_datasets,
          hypar,
          train_dataloaders_val, train_datasets_val,
          hypar["logs_path"], hypar["tensorboard_path"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    default_hyper = {"logs_path": "/home/aos-mo/Documents/ML_models/dis_2/outputs/logs/logs.csv",
                     "tensorboard_path": "/home/aos-mo/Documents/ML_models/dis_2/outputs/tensorboard",
                     "model_path": "/home/aos-mo/Documents/ML_models/dis_2/outputs/saved_weights",
                     "train_inputs": "/home/aos-mo/Documents/ML_models/dis_2/data/train",
                     "val_inputs": "/home/aos-mo/Documents/ML_models/dis_2/data/val",
                     "interm_sup": False,
                     "valid_out_dir": "",
                     "restore_model": "",
                     "start_ite": 0,
                     "gt_encoder_model": "",
                     "model_digit": "full",
                     "seed": 0,
                     "cache_size": [612, 612],
                     "cache_boost_train": False,
                     "cache_boost_valid": False,
                     "input_size": [612, 612],
                     "crop_size": [612, 612],
                     "random_flip_h": 1,
                     "random_flip_v": 0,
                     "model": ISNetDIS(),
                     "early_stop": 2,
                     "model_save_fre": 100,
                     "batch_size_train": 1,
                     "batch_size_valid": 1,
                     "max_ite": 100,
                     "max_epoch_num": 10
                     }

    # parser.add_argument('--train_inputs', default='/home/aos-mo/Documents/ML_models/dis_2/data/train')
    # parser.add_argument('--val_inputs', default='/home/aos-mo/Documents/ML_models/dis_2/data/val')
    parser.add_argument('--hyperparams', default=default_hyper)

    args, _ = parser.parse_known_args()

    main(args.hyperparams)
