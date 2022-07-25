import glob
import os
import shutil
import tempfile
import argparse
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from torchsummary import summary
import random
from datetime import datetime
from sklearn.metrics import confusion_matrix, roc_curve, auc
import cv2
from scipy.ndimage import zoom
from skimage.transform import resize
from utils.grad_cam_torch_utils import test, plot_confusion_matrix, plot_roc, plot_dis, df_plot, zipDir, confusion_matrix_CI
from utils.training_torch_utils import  BoxCrop, Annotate, Data_progressing
from efficientnet_3d.model_3d import EfficientNet3D
from resnet_3d import resnet_3d
import sys
sys.path.append("/tf/jacky831006/classification_torch/NAS-Lung/") 
from models.cnn_res import ConvRes
#from grad_cam_torch_split import Grad_CAM
import configparser
import pickle
import math
import subprocess
import gc

from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import CacheDataset, DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import compute_meandice
from monai.networks.layers import Norm
from monai.networks.nets import UNet, densenet ,SENet
#from monai.visualize import GradCAM
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
)
print_config()
import functools
# let all of print can be flush = ture
print = functools.partial(print, flush=True)

def get_parser():
    parser = argparse.ArgumentParser(description='spleen classification')
    parser.add_argument('-f', '--file', help=" The config file name. ", type=str)
    parser.add_argument('-c', '--cam_type', help=" The CAM type (LayerCAM(L) or GradCAM(G)). ", type=str)
    parser.add_argument('-g','--gt', help="  The test file use ground truth (Default is false)", default=False, type=bool)
    parser.add_argument('-l','--label', help="  The Cam map show as label", default=True, type=bool)
    return parser

# Data hyperparameter import
#cfgpath = '/tf/jacky831006/classification_torch/config/densenet_blockless_config_5.ini'
#cfgpath ='/tf/jacky831006/classification_torch/config/all_test_config_2.ini'
#cfgpath ='/tf/jacky831006/classification_torch/config/all_test_config.ini'
#cfgpath ='/tf/jacky831006/classification_torch/config/all_test_resize_config.ini'
#cfgpath ='/tf/jacky831006/classification_torch/config/all_CBAM_normal_config.ini'
#cfgpath ='/tf/jacky831006/classification_torch/config/all_CBAM_config_2.ini'
#cfgpath ='/tf/jacky831006/classification_torch/config/all_CBAM_all_config.ini'
#cfgpath ='/tf/jacky831006/classification_torch/config/all_5fold_crop_config.ini'
#cfgpath = '/tf/jacky831006/classification_torch/config/Dense_blockless_new.ini'

parser = get_parser()
args = parser.parse_args()
if args.file.endswith('ini'):
    cfgpath = f'/tf/jacky831006/classification_torch/config/{args.file}'
else:
    cfgpath = f'/tf/jacky831006/classification_torch/config/{args.file}.ini'

conf = configparser.ConfigParser()
conf.read(cfgpath)

# Augmentation
size = eval(conf.get('Augmentation','size'))

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

if conf.has_option('Data_Setting', 'cutoff'):
    cutoff = conf.getfloat('Data_Setting','cutoff')


# Data output
data_file_name = eval(conf.get('Data output','data file name'))
data_acc = eval(conf.get('Data output','best accuracy'))

# set parameter 
grad_cam_only = False

# heatmap_type: detail, one_picture, all
# cam type: LayerCAM, GradCAM
if args.cam_type not in ['L','G','GradCAM','LayerCAM']:
    raise ValueError("Input error! Only GradCAM(G) and LayerCAM(L) type")
elif args.cam_type == 'L':
    cam_type = 'LayerCAM'
elif args.cam_type == 'G':
    cam_type = 'GradCAM'
else:
    cam_type = args.cam_type
    
heatmap_type = 'all'
input_shape = size
split_num = 16
#input_shape = (128,128,128)
#output_shape = (512,512,128)
#train_df = pd.read_csv('/tf/jacky831006/object_detect_data_new/spleen_train_20220310.csv')
#valid_df = pd.read_csv('/tf/jacky831006/object_detect_data_new/spleen_valid_20220310.csv')
test_df = pd.read_csv('/tf/jacky831006/object_detect_data_new/spleen_test_20220310.csv')
bounding_box_resize = True
# standard_new_5fold
if bounding_box_resize:
    #box_df_path = "/tf/jacky831006/faster-rcnn.pytorch-1.0/bouding_box_1.2_20220316081421_0.5636_resize_new_data_test_final.csv"
    box_df_path = "/tf/jacky831006/faster-rcnn.pytorch-1.0/bouding_box_1.2_20220316081421_0.5636_resize_new_data_test_final_by_val.csv"
    #box_df_name = "/tf/jacky831006/faster-rcnn.pytorch-1.0/bouding_box_1.2_20210914125734_0.4623_resize_with_object.csv"
else:
    box_df_path = "/tf/jacky831006/faster-rcnn.pytorch-1.0/bouding_box_1.2_20220316081421_0.5636_new_data_test_final.csv"
test_df_path = box_df_path
box_df = pd.read_csv(box_df_path)
print(f'bounding box file:{box_df_path}',flush = True)

# reset test_df order as box_df
test_df = test_df.set_index('chartNo')
test_df = test_df.reindex(index=box_df['chartNo'])
test_df = test_df.reset_index()

def data_non_crop_progress(df,dicts):
    dicts =[
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip([i for i in df.source.tolist()], [i for i in df.spleen_injury.tolist()] )
    ]
    return dicts

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

for k in range(len(data_file_name)):
    # cross_validation fold number
    fold = k
    file_name = cfgpath.split("/")[-1][:-4]
    if bounding_box_resize:
        file_name = f'{file_name}_resize'
    if args.gt:
        file_name = f'{file_name}_gt'
    if not args.label:
        file_name = f'{file_name}_predicted'
    if len(data_file_name)==1:
        dir_path = f'/tf/jacky831006/classification_torch/grad_cam_image_new/{file_name}/'
    else:
        dir_path = f'/tf/jacky831006/classification_torch/grad_cam_image_new/{file_name}/{fold}'
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    # not edited
    #load_weight = '/tf/jacky831006/classification_torch/training_checkpoints/20211124114558/0.9.pth'
    load_weight = f'/tf/jacky831006/classification_torch/training_checkpoints/{data_file_name[k]}/{data_acc[k]}.pth'
    #load_weight = '/tf/jacky831006/classification_torch/training_checkpoints/20211130084741/0.9.pth'
    #load_weight = '/tf/jacky831006/classification_torch/training_checkpoints/20211227035213/0.8333333333333334.pth'
    #load_weight = '/tf/jacky831006/classification_torch/training_checkpoints/20211224191352/0.7666666666666667_last.pth'
    #load_weight = '/tf/jacky831006/classification_torch/training_checkpoints/20220106083225/0.7833333333333333_last.pth'
    test_data_dicts = []
    if cropping:
        if args.gt:
            test_data_dicts = data_crop_progress(test_df,test_data_dicts)
        else:
            for index,row in box_df.iterrows():
                        image = row['Path']
                        label = row['BBox']
                        test_data_dicts.append({'image':image,'label':label})
    else:
        test_data_dicts = data_non_crop_progress(test_df,test_data_dicts)

    print(f'Fold:{fold}, file:{data_file_name}, acc:{data_acc}')

    # if only grad cam plot
    if not grad_cam_only:
        if cropping:
            if args.gt:
                test_transforms = Compose([
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
            else:
                test_transforms = Compose(
                    [
                        LoadNiftid(keys=["image"]),
                        AddChanneld(keys=["image"]),
                        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear")),
                        Orientationd(keys=["image"], axcodes="RAS"),
                        CropForegroundd(keys=["image"], source_key="image"),
                        BoxCrop(keys= ["image","label"]),
                        Resized(keys=['image'], spatial_size = input_shape),
                        ScaleIntensityRanged(
                            keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True,
                        )
                    ]
                )
        else:
            test_transforms = Compose(
                [
                    LoadNiftid(keys=["image"]),
                    AddChanneld(keys=["image"]),
                    Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear")),
                    Orientationd(keys=["image"], axcodes="RAS"),
                    CropForegroundd(keys=["image"], source_key="image"),
                    #BoxCrop(keys= ["image","label"]),
                    Resized(keys=['image'], spatial_size = input_shape),
                    ScaleIntensityRanged(
                        keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True,
                    )
                ]
            )

        print("Collecting:", datetime.now(), flush=True)

        test_ds = CacheDataset(data=test_data_dicts, transform=test_transforms, cache_rate=1,num_workers=dataloader_num_workers)
        test_data = DataLoader(test_ds, batch_size=testing_batch_size, num_workers=dataloader_num_workers)

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

        model.load_state_dict(torch.load(load_weight))

        # testing predicet
        y_pre = test(model, test_data, device)

        # ROC curve figure
        y_label = test_df['spleen_injury'].values
        optimal_th = plot_roc(y_pre, y_label, dir_path, f'{file_name}_{fold}')
    
        # Data distributions
        pos_list = []
        neg_list = []
        for i in zip(y_label,y_pre):
            if i[0] == 1:
                pos_list.append(i[1][1])
            else:
                neg_list.append(i[1][1])

        plot_dis(pos_list, neg_list, dir_path, f'{file_name}_{fold}')
        print(f'cutoff value:{optimal_th}')
        # Select cutoff value by roc curve
        y_pre_n = list()
        
        for i in range(y_pre.shape[0]):
            if 'cutoff' in locals():
                if y_pre[i][1] < cutoff:
                    y_pre_n.append(0)
                else:
                    y_pre_n.append(1)
            else:
                if y_pre[i][1] < optimal_th:
                    y_pre_n.append(0)
                else:
                    y_pre_n.append(1)
            #if y_pre[i][1] > optimal_th:
            #    y_pre_n.append(1)
            #else:
            #    y_pre_n.append(0)
        
        y_list = list(test_df.spleen_injury)

        # write csv
        test_df['pre_label']=np.array(y_pre_n)
        test_df['ori_pre']=list(y_pre)
        test_df = test_df[test_df.columns.drop(list(test_df.filter(regex='Unnamed')))]
        test_df.to_csv(f"{dir_path}/{file_name}_{fold}.csv",index = False)
        # Spleen grade 
        df_plot(test_df,dir_path,file_name,fold)
        
        # confusion matrix
        result=confusion_matrix(y_list,y_pre_n)
        (tn, fp, fn, tp)=confusion_matrix(y_list,y_pre_n).ravel()
        plot_confusion_matrix(result, classes=[0, 1], title='Confusion matrix')
        plt.savefig(f"{dir_path}/{file_name}_{fold}.png")
        plt.close()
        #plt.show()
        # 取小數點到第二位
        ACC, PPV, NPV, Sensitivity, Specificity = confusion_matrix_CI(tn, fp, fn, tp)
        print(f'Modifed Test Accuracy: {ACC}')
        print("PPV:",PPV,"NPV:",NPV,"Sensitivity:",Sensitivity,"Specificity:",Specificity)

        del test_ds
        del test_data
        gc.collect()

    # Grad cam (close every times)

    for i in range(math.ceil(len(test_data_dicts)/split_num)):
        # if input not str, all need to transfer to str
        print(f'--------Fold {i}--------',flush= True)
        grad_cam_run = subprocess.run(["python3","/tf/jacky831006/classification_torch/All_structure/All_grad_cam_torch_split.py", 
                                    "-W", load_weight, "-B", box_df_path, "-D", test_df_path, "-C", cfgpath, "-F", str(i),
                                    "-S", str(split_num), "-H", heatmap_type, "-G", cam_type, "-O", file_name, "-T", str(args.gt), "-L", str(args.label)], 
                                    stdout=subprocess.PIPE, universal_newlines=True)
        print(grad_cam_run.stdout)
    print(f'Fold {fold} is finish!')

# zip video dictionary
AIS345_path = f"{dir_path}/{cam_type}/AIS345"
AIS12_path = f"{dir_path}/{cam_type}/AIS12"
neg_path = f"{dir_path}/{cam_type}/NEG"

path_list = [AIS12_path, AIS345_path, neg_path]
for path in path_list:
    zipDir(f'{path}/video',f'{path}_video.zip')


print('All is done !')
