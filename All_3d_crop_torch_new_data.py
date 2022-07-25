import glob
import os
import argparse
import shutil
import tempfile
import time
from datetime import datetime
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch.optim as optim
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from torchsummary import summary
import random
import numpy as np
import configparser
import gc
import math
from utils.training_torch_utils import BoxCrop, Dulicated, Annotate, train, validation, Data_progressing, plot_loss_metric
import utils.config as config
from efficientnet_3d.model_3d import EfficientNet3D
from resnet_3d import resnet_3d
from ModelsGenesis import unet3d

import sys
sys.path.append("/tf/jacky831006/classification_torch/NAS-Lung/") 
from models.cnn_res import ConvRes
from models.net_sphere import AngleLoss

from monai.config import print_config
from monai.data import CacheDataset, DataLoader, Dataset
from monai.metrics import compute_meandice
from monai.networks.layers import Norm
from monai.networks.nets import UNet, densenet ,SENet
from monai.utils import set_determinism
from monai.transforms import (
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadNiftid,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
    Resized,
    RandAffined,
    Rand3DElasticd
)
print_config()
import functools
# let all of print can be flush = ture
print = functools.partial(print, flush=True)

'''
使用全部都有label的新資料
train,valid 都直接使用true label
test 則是使用predict 或是 true label
使用true label 則transformer不能使用CropForegroundd
'''

def get_parser():
    parser = argparse.ArgumentParser(description='spleen classification')
    parser.add_argument('-f', '--file', help=" The config file name. ", type=str)
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    if args.file.endswith('ini'):
        cfgpath = f'/tf/jacky831006/classification_torch/config/{args.file}'
    else:
        cfgpath = f'/tf/jacky831006/classification_torch/config/{args.file}.ini'

    # Data hyperparameter import
    #cfgpath ='/tf/jacky831006/classification_torch/config/densenet_blockless_new_datasplit_with_object_config.ini'
    #cfgpath ='/tf/jacky831006/classification_torch/config/all_test_config_3.ini'
    #cfgpath ='/tf/jacky831006/classification_torch/config/all_test_resize_config.ini'
    conf = configparser.ConfigParser()
    conf.read(cfgpath)

    # Augmentation
    num_samples = conf.getint('Augmentation','num_sample')
    size = eval(conf.get('Augmentation','size'))
    prob = conf.getfloat('Rand3DElasticd','prob')
    sigma_range = eval(conf.get('Rand3DElasticd','sigma_range'))
    magnitude_range = eval(conf.get('Rand3DElasticd','magnitude_range'))
    translate_range = eval(conf.get('Rand3DElasticd','translate_range'))
    rotate_range = eval(conf.get('Rand3DElasticd','rotate_range'))
    scale_range = eval(conf.get('Rand3DElasticd','scale_range'))

    # Data_setting
    architecture = conf.get('Data_Setting','architecture')
    if architecture == 'efficientnet':
        structure_num = conf.get('Data_Setting', 'structure_num')
    gpu_num = conf.getint('Data_Setting','gpu')
    seed = conf.getint('Data_Setting','seed')
    cross_kfold = conf.getint('Data_Setting','cross_kfold')
    bounding_box_resize = conf.getboolean('Data_Setting','bounding_box_resize')
    normal_structure = conf.getboolean('Data_Setting','normal_structure')
    data_split_ratio = eval(conf.get('Data_Setting','data_split_ratio'))
    imbalance_data_ratio = conf.getint('Data_Setting','imbalance_data_ratio')
    epochs = conf.getint('Data_Setting','epoch')
    early_stop = conf.getint('Data_Setting','early_stop')
    traning_batch_size = conf.getint('Data_Setting','traning_batch_size')
    valid_batch_size = conf.getint('Data_Setting','valid_batch_size')
    testing_batch_size = conf.getint('Data_Setting','testing_batch_size')
    dataloader_num_workers = conf.getint('Data_Setting','dataloader_num_workers')
    init_lr = conf.getfloat('Data_Setting','init_lr')
    optimizer = conf.get('Data_Setting','optimizer')
    lr_decay_rate = conf.getfloat('Data_Setting','lr_decay_rate')
    lr_decay_epoch = conf.getint('Data_Setting','lr_decay_epoch')
    cropping = conf.getboolean('Data_Setting','cropping')
    predict_test = conf.getboolean('Data_Setting','predict_test')

    # Dataloader define
    if cropping:
        train_transforms = Compose([
                LoadNiftid(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
                Orientationd(keys=["image", 'label'], axcodes="RAS"),
                #CropForegroundd(keys=["image"], source_key="image"),
                Dulicated(keys= ["image","label"], num_samples = num_samples),
                Annotate(keys=["image", "label"]), # trans label from img to coordinate
                BoxCrop(keys= ["image","label"]), # crop image as label and set label as injury label
                Resized(keys=['image'], spatial_size = size),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True,
                ),
                Rand3DElasticd(
                    keys=["image"],
                    mode=("bilinear"),
                    prob=prob,
                    sigma_range=sigma_range,
                    magnitude_range=magnitude_range,
                    spatial_size=size,
                    translate_range=translate_range,
                    rotate_range=rotate_range,
                    scale_range=scale_range,
                    padding_mode="border")
                
                #ToTensord(keys=["image", "label"])
            ]
        )
        valid_transforms = Compose([
                LoadNiftid(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
                Orientationd(keys=["image", 'label'], axcodes="RAS"),
                #CropForegroundd(keys=["image"], source_key="image"),
                Annotate(keys=["image", "label"]), # trans label from img to coordinate
                BoxCrop(keys= ["image","label"]), # crop image as label and set label as injury label
                Resized(keys=['image'], spatial_size = size),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True,
                )
                ]
            )
        # Predict label
        if not predict_test:
            test_transforms = valid_transforms
        else:
            test_transforms = Compose([
                    LoadNiftid(keys=["image"]),
                    AddChanneld(keys=["image"]),
                    Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear")),
                    Orientationd(keys=["image"], axcodes="RAS"),
                    CropForegroundd(keys=["image"], source_key="image"),
                    BoxCrop(keys= ["image","label"]),
                    Resized(keys=['image'], spatial_size = size),
                    ScaleIntensityRanged(
                        keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True,
                    )
                ]
            )
            # valid use predicted
            #valid_transforms = test_transforms
    else:
        train_transforms = Compose([
            LoadNiftid(keys=["image"]),
            AddChanneld(keys=["image"]),
            Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear")),
            Orientationd(keys=["image"], axcodes="RAS"),
            CropForegroundd(keys=["image"], source_key="image"),
            #BoxCrop(keys= ["image","label"]),
            Resized(keys=['image'], spatial_size = size),
            ScaleIntensityRanged(
                keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True,
            ),
            Dulicated(keys= ["image","label"], num_samples = num_samples),
            Rand3DElasticd(
                keys=["image"],
                mode=("bilinear"),
                prob=prob,
                sigma_range=sigma_range,
                magnitude_range=magnitude_range,
                spatial_size=size,
                translate_range=translate_range,
                rotate_range=rotate_range,
                scale_range=scale_range,
                padding_mode="border")
        ]
        )
        valid_transforms = Compose([
                LoadNiftid(keys=["image"]),
                AddChanneld(keys=["image"]),
                Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear")),
                Orientationd(keys=["image"], axcodes="RAS"),
                CropForegroundd(keys=["image"], source_key="image"),
                #BoxCrop(keys= ["image","label"]),
                Resized(keys=['image'], spatial_size = size),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True,
                )
            ]
        )
        test_transforms = valid_transforms

    # Data import 
    # old data
    #train_df = pd.read_csv("/tf/jacky831006/spleen_train.csv")
    #test_df = pd.read_csv("/tf/jacky831006/spleen_test.csv")
    #box_df = pd.read_csv("/tf/jacky831006/faster-rcnn.pytorch-0.4/bouding_box_1.2_20210809065008_0.4162.csv")

    # cross_valid 
    #cross_valid = True
    #kfold = 5

    # Training by cross validation
    accuracy_list = []
    test_accuracy_list = []
    file_list = []
    epoch_list = []
    # Data import 
    #all_df = pd.read_csv('/tf/jacky831006/classification_torch/spleen_with_object_data_train.csv')
    train_df = pd.read_csv('/tf/jacky831006/object_detect_data_new/spleen_train_20220310.csv')
    valid_df = pd.read_csv('/tf/jacky831006/object_detect_data_new/spleen_valid_20220310.csv')
    test_df = pd.read_csv('/tf/jacky831006/object_detect_data_new/spleen_test_20220310.csv')

    # standard_new_5fold
    if bounding_box_resize:
        box_df_name = "/tf/jacky831006/faster-rcnn.pytorch-1.0/bouding_box_1.2_20220316081421_0.5636_resize_new_data_latest.csv"
        #box_df_name = "/tf/jacky831006/faster-rcnn.pytorch-1.0/bouding_box_1.2_20210914125734_0.4623_resize_with_object.csv"
    else:
        box_df_name = "/tf/jacky831006/faster-rcnn.pytorch-1.0/bouding_box_1.2_20220316081421_0.5636_new_data_final.csv"


    box_df = pd.read_csv(box_df_name)
    print(f'bounding box file:{box_df_name}',flush = True)
    # Valid bounding box file
    valid_box_df =  pd.read_csv("/tf/jacky831006/faster-rcnn.pytorch-1.0/bouding_box_1.2_20220316081421_0.5636_resize_new_data_valid_final.csv")
    #path=list(train_df.path)

    #train_ratio = 0.70
    #validation_ratio = 0.10
    #test_ratio = 0.20
    #data_split_ratio = (train_ratio,validation_ratio,test_ratio)
    if cross_kfold*data_split_ratio[2] != 1 and cross_kfold!=1:
        raise RuntimeError("Kfold number is not match test data ratio")

    first_start_time = time.time()

    def data_crop_progress(df, dicts):
        for index, row in df.iterrows():
            if row['spleen_injury'] == 0:
                image = row['source']
                label = row['source'].replace('image','label')
                cls = 0
            else:
                image = f'/tf/jacky831006/object_detect_data_new/pos/image/{row["chartNo"]}@venous_phase.nii.gz'
                label = f'/tf/jacky831006/object_detect_data_new/pos/label/{row["chartNo"]}@venous_phase.nii.gz'
                cls = 1
            dicts.append({'image':image,'label':label,'class':cls})
        return dicts

    def data_non_crop_progress(df,dicts):
        for index, row in df.iterrows():
            if row['spleen_injury'] == 0:
                image = row['source']
                label = 0
            else:
                image = f'/tf/jacky831006/object_detect_data_new/pos/image/{row["chartNo"]}@venous_phase.nii.gz'
                label = 1
            dicts.append({'image':image,'label':label})
        return dicts

    train_data_dicts = []
    valid_data_dicts = []
    test_data_dicts = []
    if cropping:
        train_data_dicts = data_crop_progress(train_df,train_data_dicts)
        valid_data_dicts = data_crop_progress(valid_df,valid_data_dicts)
        if not predict_test:
            test_data_dicts = data_crop_progress(test_df,test_data_dicts)
        else:
            for index,row in box_df.iterrows():
                image = row['Path']
                label = row['BBox']
                test_data_dicts.append({'image':image,'label':label})
        # valid use predicted
        '''
            valid_data_dicts = []
            for index,row in valid_box_df.iterrows():
                image = row['Path']
                label = row['BBox']
                valid_data_dicts.append({'image':image,'label':label})
        '''
    else:
        train_data_dicts = data_non_crop_progress(train_df,train_data_dicts)
        valid_data_dicts = data_non_crop_progress(valid_df,valid_data_dicts)
        test_data_dicts = data_non_crop_progress(test_df,test_data_dicts)

    print(f'\n Train:{len(train_data_dicts)},Valid:{len(valid_data_dicts)},Test:{len(test_data_dicts)}')
    
    # set augmentation seed 
    set_determinism(seed=0)
    train_ds = CacheDataset(data=train_data_dicts, transform=train_transforms, cache_rate=1, num_workers=dataloader_num_workers)
    train_loader = DataLoader(train_ds, batch_size=traning_batch_size, shuffle=True, num_workers=dataloader_num_workers)

    valid_ds = CacheDataset(data=valid_data_dicts, transform=valid_transforms, cache_rate=1, num_workers=dataloader_num_workers)
    val_loader = DataLoader(valid_ds, batch_size=valid_batch_size, num_workers=dataloader_num_workers)

    # Create DenseNet121, CrossEntropyLoss and Adam optimizer
    device = torch.device("cuda",gpu_num)
    if architecture == 'densenet':
        if normal_structure:
            # Normal DenseNet121
            model = densenet.densenet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)
        else:
            # Delete last dense block
            model = densenet.DenseNet(spatial_dims=3, in_channels=1, out_channels=2, block_config=(6, 12, 40)).to(device)
    # Create EfficientNet, CrossEntropyLoss and Adam optimizer
    elif architecture == 'efficientnet':
        model = EfficientNet3D.from_name(f"efficientnet-{structure_num}", in_channels=1, num_classes=2, image_size=size, normal=normal_structure).to(device)
    elif architecture == 'resnet':
        model = resnet_3d.generate_model(101,normal=normal_structure).to(device)
    elif architecture == 'CBAM':
        if size[0] == size[1] == size[2]:
            model = ConvRes(size[0], [[64, 64, 64], [128, 128, 256], [256, 256, 256, 512]], normal=normal_structure).to(device)
        else:
            raise RuntimeError("CBAM model need same size in x,y,z axis")

    # Imbalance loss
    weights = torch.tensor([1/(imbalance_data_ratio+1),imbalance_data_ratio/(imbalance_data_ratio+1)]).to(device)
    if architecture == 'CBAM' and not normal_structure:
        loss_function = AngleLoss()
    else:
        loss_function = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), init_lr)
    if lr_decay_epoch == 0:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=lr_decay_rate, patience=epochs, verbose =True)
        #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=epochs,gamma=lr_decay_rate, verbose=True )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=lr_decay_rate, patience=lr_decay_epoch, verbose =True)
        #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_epoch,gamma=lr_decay_rate, verbose=True )

    # file name (time)
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")               
    root_logdir = "/tf/jacky831006/classification_torch/tfboard"     
    logdir = "{}/run-{}/".format(root_logdir, now) 

    # tfboard file path
    # 創一個主目錄 之後在train內的sumamaryWriter都會以主目錄創下面路徑
    writer = SummaryWriter(logdir)
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    # check_point path
    check_path = f'/tf/jacky831006/classification_torch/training_checkpoints/{now}'
    if not os.path.isdir(check_path):
        os.makedirs(check_path)
    print(f'\n Weight location:{check_path}',flush = True)
    if cross_kfold == 1:
        print(f'\n Processing begining',flush = True)
    else:
        print(f'\n Processing fold #{k}',flush = True)

    data_num = len(train_ds)
    #test_model = train(model, device, data_num, epochs, optimizer, loss_function, train_loader, \
    #                    val_loader, early_stop, init_lr, lr_decay_rate, lr_decay_epoch, check_path)

    test_model = train(model, device, data_num, epochs, optimizer, loss_function, train_loader, \
                        val_loader, early_stop, scheduler, check_path)
                    
    # plot train loss and metric 
    plot_loss_metric(config.epoch_loss_values, config.metric_values, check_path)
    print(f'Before {config.best_metric}')
    # remove dataloader to free memory
    del train_ds
    del train_loader
    del valid_ds
    del val_loader
    gc.collect()

    # Avoid ram out of memory
    test_ds = CacheDataset(data=test_data_dicts, transform=test_transforms, cache_rate=1, num_workers=dataloader_num_workers)
    test_loader = DataLoader(test_ds, batch_size=testing_batch_size, num_workers=dataloader_num_workers)
    # validation is same as testing
    print(f'\n After {config.best_metric}')
    if config.best_metric != 0:
        load_weight = f'{check_path}/{config.best_metric}.pth'
        model.load_state_dict(torch.load(load_weight))

    # record paramter
    accuracy_list.append(config.best_metric)
    file_list.append(now)
    epoch_list.append(config.best_metric_epoch)

    # reset config parameter
    config.best_metric = 0
    config.best_metric_epoch = 0
    config.metric_values = list()
    config.epoch_loss_values = list()

    test_acc = validation(model, test_loader, device)
    test_accuracy_list.append(test_acc)
    del test_ds
    del test_loader
    gc.collect()

    print(f'\n Best accuracy:{config.best_metric}, Best test accuracy:{test_acc}')

    if cross_kfold != 1:
        print(f'\n Mean accuracy:{sum(accuracy_list)/len(accuracy_list)}')

    final_end_time = time.time()
    hours, rem = divmod(final_end_time-first_start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    all_time = "All time:{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
    print(all_time)
    # write some output information in ori ini
    conf['Data output'] = {}
    conf['Data output']['Running time'] = all_time
    conf['Data output']['Data file name'] = str(file_list)
    # ini write in type need str type
    conf['Data output']['Best accuracy'] = str(accuracy_list)
    conf['Data output']['Best Test accuracy'] = str(test_accuracy_list)
    conf['Data output']['Best epoch'] = str(epoch_list)

    with open(cfgpath, 'w') as f:
        conf.write(f)

