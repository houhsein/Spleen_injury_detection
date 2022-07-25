import glob
import os
import shutil
import tempfile
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
from utils.training_torch_utils import Annotate, BoxCrop
from utils.grad_cam_torch_utils import plot_heatmap_detail, plot_heatmap_one_picture, plot_vedio, get_last_conv_name, GradCAM, LayerCAM, Backup
from efficientnet_3d.model_3d import EfficientNet3D
from resnet_3d import resnet_3d
import sys, getopt
sys.path.append("/tf/jacky831006/classification_torch/NAS-Lung/") 
from models.cnn_res import ConvRes
#from grad_cam_torch_split import Grad_CAM
import configparser
import pickle
import math

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
import functools
# let all of print can be flush = ture
print = functools.partial(print, flush=True)
#print_config()

# "input_weight=", "bounding_box=", "data_file=", "fold=", "output="
def usage():
    print(f'''Usage:{sys.argv[0]} [-W  model weight | --input_weight][-B bounding box | --bounding_box][-D data file | --data_file][-C config file | --config]
    [-F data fold | --fold][-S data split number | --split][-H heatmap type of grad cam | --heatmap_type][-G cam type | --cam_type][-O output name | --output]
    [-T Use True label | --ground_truth][-L CAM map show as label | --Label]
    -h help
    -w model weight
    -B bounding box
    -D data file
    -C config file
    -F data fold
    -S data split number
    -H heatmap type of grad cam (detail or one picture)
    -G cam type
    -O output file path
    -T Use True label
    -L CAM map show as label
    ''')


def CAM_plot(model, test_data, test_df, output_file_name, size, device, first, detail, cam_type, architecture, Label, cropping):
    channel_first = True
    input_shape = size
    # cropping img output size 固定 128,128,64, whole img output size 固定 300,300,64
    if cropping:
        output_shape = 128,128,64
    else:
        output_shape = 300,300,64 

    #file_name = "DenseNet_crop_blockless_5_2"
    dir_path = f'/tf/jacky831006/classification_torch/grad_cam_image_new/{output_file_name}/{cam_type}'
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    # Old testing data

    # 分成OIS 3,4,5 跟OIS 1,2 就好
    AIS345_path = f"{dir_path}/AIS345"
    AIS12_path = f"{dir_path}/AIS12"
    AIS345_total = f"{dir_path}/AIS345_total"
    AIS12_total = f"{dir_path}/AIS12_total"
    neg_path = f"{dir_path}/NEG"
    neg_total = f"{dir_path}/NEG_total"
    layer_dic = {
        'efficientnet':[],
        'resnet':['layer1.2.conv3','layer2.3.conv3','layer3.25.conv3'],
        'densenet':['features.denseblock1.denselayer6.layers.conv2','features.denseblock2.denselayer12.layers.conv2',
                    'features.denseblock3.denselayer40.layers.conv2'],
        'CBAM':['layers.3.conv3.0','layers.8.conv3.0','layers.14.conv3.0']
        }
    if architecture == 'efficientnet':
        inputs = torch.rand(1, 1, 128, 128, 64).to(device)
        _, endindex = model.extract_endpoints(inputs)
        layer_name = [f"_blocks.{i-1}._project_conv" for i in endindex.values()] + ['_conv_head']
        layer_dic['efficientnet'] = layer_name
    
    test_sel= test_df.dropna(subset=['spleen_injury_class'])
    AIS345_df = test_sel[test_sel.spleen_injury_class == 'OIS 3,4,5']
    AIS12_df = test_sel[test_sel.spleen_injury_class == 'OIS 1,2']
    k = first

    for testdata in test_data:
        test_images = testdata['image'].to(device)
        test_labels = testdata['label'].to(device)
        out_images = testdata['ori_image'].to(device)
        file_name = testdata['image_meta_dict']['filename_or_obj']
        '''
        GradCam need reset at every epoch
        '''

        if cam_type == "GradCAM":
            if architecture == 'CBAM':
                layer_name = get_last_conv_name(model)[-2]
            else:
                layer_name = get_last_conv_name(model)[-1]
            grad_cam = GradCAM(model, layer_name, device)
            
        elif cam_type == "LayerCAM":
            layer_name = layer_dic[architecture]
            grad_cam = LayerCAM(model, layer_name, device)
        # label true means show the map as label, false means show the map as predicted
        if Label == 'True':     
            result_list = grad_cam(test_images, test_labels, index_sel=test_labels)
        else:
            result_list = grad_cam(test_images, test_labels)
        grad_cam.remove_handlers()

        for i in range(len(result_list)):
            print(f"Read file in line {k}",flush = True)
            file_path = test_df.ID[k]
            #if not os.path.isdir(f"{test_path}/{file_path}"):
            #    os.makedirs(f"{test_path}/{file_path}")
            if file_path in list(AIS345_df.ID):
                final_path = f"{AIS345_path}/{file_path}" 
                total_final_path = f"{AIS345_total}/{file_path}_total" 
            elif file_path in list(AIS12_df.ID):
                final_path = f"{AIS12_path}/{file_path}"
                total_final_path = f"{AIS12_total}/{file_path}_total" 
            else:
                final_path = f"{neg_path}/{file_path}"
                total_final_path = f"{neg_total}/{file_path}_total" 
                #print("Label is negative, pass it !")
                #k += 1
                #continue 
            
            if not os.path.isdir(final_path):
                os.makedirs(final_path)
            if not os.path.isdir(total_final_path):
                os.makedirs(total_final_path)    

            if channel_first:
            #    image = test_images[i,0,:,:,:].cpu().detach().numpy()
                image = out_images[i,0,:,:,:].cpu().detach().numpy()   
            else:
            #    image = test_images[i,:,:,:,0].cpu().detach().numpy()
                image = out_images[i,:,:,:,0].cpu().detach().numpy()
            heatmap_total = result_list[i]
            
            # image and heatmap resize (z aixs didn't chage)
            #image = zoom(image, (output_shape[0]/input_shape[0], output_shape[1]/input_shape[1], output_shape[2]/input_shape[2]))
            heatmap_total = zoom(heatmap_total,(output_shape[0]/input_shape[0], output_shape[1]/input_shape[1], output_shape[2]/input_shape[2]))

            if detail == 'detail':
                #print('detail is true',flush=True)
                for j in range(image.shape[-1]):
                    plot_heatmap_detail(heatmap_total[:,:,j],image[:,:,j],f"{final_path}/{j:03}.png")
                    plot_vedio(final_path)
            elif detail == 'one_picture':
                #print('detail is false',flush=True)
                plot_heatmap_one_picture(heatmap_total,image,f'{total_final_path}/total_view.png')
            elif detail == 'all':
                plot_heatmap_one_picture(heatmap_total,image,f'{total_final_path}/total_view.png')
                for j in range(image.shape[-1]):
                    plot_heatmap_detail(heatmap_total[:,:,j],image[:,:,j],f"{final_path}/{j:03}.png")
                    plot_vedio(final_path)
            k += 1
            print(f'{file_path} is already done!',flush = True)

def main(argv):
    try:
        opts, args = getopt.getopt(argv[1:], 'hW:B:D:C:F:S:H:G:O:T:L:', ["help", "input_weight=", "bounding_box=", "data_file=", "config=", "fold=", "split=", "heatmap_type=","cam_type=", "output=", "ground_truth=", "Label="])
    except getopt.GetoptError:
        print(f'{argv[0]} -W <Input data weight> -B <bounding box df> -D <data file> -F <Fold number of data> -S <split number of data> -H <heatmap type> -G <cam type> -O <output path> -T <Use ground truth> -L <CAM map show as label>')
        sys.exit()

    for name, value in opts:
        if name in ('-h', '--help'):
            usage()
            sys.exit()
        elif name in ('-W', '--input_weight'):
            load_weight = value
        elif name in ('-B', '--bounding_box'):
            bounding_box = value
        elif name in ('-D', '--data_file'):
            data_file = value
        elif name in ('-C', '--config'):
            cfgpath = value
        elif name in ('-F', '--fold'):
            fold = int(value)
        elif name in ('-S', '--split'):
            split_num = int(value)
        elif name in ('-H', '--heatmap_type'):
            heatmap_type = value
        elif name in ('-G', '--cam_type'):
            cam_type = value
        elif name in ('-O', '--output'):
            output_file = value
        elif name in ('-T', '--ground_truth'):
            ground_truth = value
        elif name in ('-L', '--Label'):
            Label = value

    # Data hyperparameter import
    #cfgpath ='/tf/jacky831006/classification_torch/config/densenet_blockless_config_5.ini'
    #print(cfgpath)
    conf = configparser.ConfigParser()
    conf.read(cfgpath)
    #print(conf.sections())
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
    
    # set parameter 
    input_shape = size
    # file_name = "DenseNet_crop_blockless_5_2"
    #load_weight = '/tf/jacky831006/classification_torch/training_checkpoints/20210910085735/0.84.pth'
    batch_size = 8
    # label true means show the map as label, false means show the map as predicted  
    # Old testing data (迺逾醫師 label)
    # Data import 
    test_df = pd.read_csv(data_file) # "/tf/jacky831006/faster-rcnn.pytorch-0.4/bouding_box_1.2_20220316081421_0.5636_resize_new_data_test_final.csv"
    #test_df = pd.read_csv(data_file) # "/tf/jacky831006/classification_torch/spleen_test_new.csv"
    #box_df = pd.read_csv(bounding_box) # "/tf/jacky831006/faster-rcnn.pytorch-0.4/bouding_box_1.2_20210809065008_0.4162_new.csv"
    box_df = pd.read_csv(bounding_box) # "/tf/jacky831006/faster-rcnn.pytorch-0.4/bouding_box_1.2_20220316081421_0.5636_resize_new_data_test_final.csv"
    #path_t=list(test_df.path)
    #new_path_t=[i.replace("/data/","/tf/")for i in path_t]
    #label_one_hot=np.eye(2)[train_df.spleen_injury.values.astype('int64')]

    #test_data = box_df[box_df.Path.isin(new_path_t)]
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
    '''
    if cropping:
        test_data_dicts = []
        for index,row in test_df.iterrows():
            image = row['Path']
            label = row['BBox']
            test_data_dicts.append({'image':image,'label':label})
    else:
        test_df['spleen_injury'] = np.array([0 if i else 1 for i in test_df.spleen_injury_class.isna().tolist()])
        test_data_dicts =[
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip([i for i in test_df.source.tolist()], [i for i in test_df.spleen_injury.tolist()] )
        ]
    '''
    test_data_dicts = []
    if cropping:
        if ground_truth == 'True':
            test_data_dicts = data_crop_progress(test_df,test_data_dicts)
        else:
            for index,row in box_df.iterrows():
                        image = row['Path']
                        label = row['BBox']
                        test_data_dicts.append({'image':image,'label':label})
    else:
        test_data_dicts = data_non_crop_progress(test_df,test_data_dicts)
        
    # select data fold
    test_data_dicts_sel = test_data_dicts[split_num*fold:split_num*(fold+1)]
    #print(test_data_dicts_sel,flush = True)

    if cropping:
        if ground_truth == 'True':
            test_transforms = Compose([
                LoadNiftid(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
                Orientationd(keys=["image", 'label'], axcodes="RAS"),
                #CropForegroundd(keys=["image"], source_key="image"),
                Annotate(keys=["image", "label"]), # trans label from img to coordinate
                BoxCrop(keys= ["image","label"]), # crop image as label and set label as injury label
                Backup(keys=["image"]), # Save ori image for gradcam output
                Resized(keys=['image'], spatial_size = size),
                Resized(keys=['ori_image'], spatial_size = (128,128,64)),
                ScaleIntensityRanged(
                    keys=["image","ori_image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True,
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
                    Backup(keys=["image"]), # Save ori image for gradcam output
                    Resized(keys=['image'], spatial_size = input_shape),
                    Resized(keys=['ori_image'], spatial_size = (128,128,64)),
                    ScaleIntensityRanged(
                    keys=["image","ori_image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True,
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
                Backup(keys=["image"]), # Save ori image for gradcam output
                Resized(keys=['image'], spatial_size = input_shape),
                Resized(keys=['ori_image'], spatial_size = (300,300,64)),
                ScaleIntensityRanged(
                    keys=["image","ori_image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True,
                )
            ]
        )        

    print("Collecting:", datetime.now(), flush=True)
    
    if cam_type == 'LayerCAM':
        batch_size = 1
    test_ds = CacheDataset(data=test_data_dicts_sel, transform=test_transforms, cache_rate=1, num_workers=dataloader_num_workers)
    test_data = DataLoader(test_ds, batch_size=batch_size, num_workers=dataloader_num_workers)

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
    model.load_state_dict(torch.load(load_weight))

    # Plot grad cam
    CAM_plot(model, test_data, test_df, output_file, size, device, split_num*fold, heatmap_type, cam_type, architecture, Label, cropping)

    

if __name__ == '__main__':
    main(sys.argv)