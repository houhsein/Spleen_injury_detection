import os
import time
from datetime import datetime
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import random
import numpy as np
import utils.config as config
import matplotlib.pyplot as plt
import os, psutil
import functools
from skimage.measure import label as sk_label
from skimage.measure import regionprops as sk_regions
from skimage.transform import resize
# let all of print can be flush = ture
print = functools.partial(print, flush=True)

#-------- Dataloder --------
# After augmnetation with resize, crop spleen area and than transofermer 
class BoxCrop(object):
    '''
    Croping image by bounding box label after augmentation 
    input: keys=["image", "label"]
    label:
    [[x1,y1,x2,y2,z1,z2,class]...]
    image:
    [1,x,y,z]
    output dictionary add 
        im_info: [x,y,z,scale_x_y,scale_z]
        num_box: 1 (All is one in our data)
    '''
    def __init__(self,keys):
        self.keys = keys

    def __call__(self, data):
        d = dict(data)
        image = d['image']
        label = d['label']
        # only one label
        if type(label) == type(np.array([])):
            label_list = label.tolist()
        else:
        # more than one label
        # select the first label      
            label_list = eval(label)[0]
        if label_list[1]>=label_list[3] or label_list[0]>=label_list[2] or label_list[4]>=label_list[5]:
            raise RuntimeError(f"{d['image_meta_dict']['filename_or_obj']} bounding box error")
                #print(f"{d['image_meta_dict']['filename_or_obj']} bounding box error ")
        out_image = image[0, int(label_list[1]):int(label_list[3]), int(label_list[0]):int(label_list[2]), int(label_list[4]):int(label_list[5])]
        d['image'] = np.expand_dims(out_image,axis=0)
        d['label'] = label_list[6]
        #print(d['image'].shape)
        return d

# Dulicated dataset by num_samples
class Dulicated(object):
    '''
    Dulicated data for augmnetation
    '''
    def __init__(self,
                 keys,
                 num_samples: int = 1):
        self.keys = keys
        self.num_samples = num_samples

    def __call__(self, data):
        d = dict(data)
        image = d['image']
        label = d['label']
        results: List[Dict[Hashable, np.ndarray]] = [dict(data) for _ in range(self.num_samples)]
            
        for key in data.keys():            
            for i in range(self.num_samples):
                results[i][key] = data[key]
        return results
        #return d

# True label
class Annotate(object):
    '''
    transform mask to bounding box label after augmentation
    check the image shape to know scale_x_y, scale_z 
    input: keys=["image", "label"]
    output dictionary add 
        im_info: [x,y,z,scale_x_y,scale_z]
        num_box: 1 (All is one in our data)
    '''
    def __init__(self,keys):
        self.keys = keys

    def __call__(self, data):
        d = dict(data)
        #image = d[self.keys[0]]
        #label = d[self.keys[1]]
        image = d['image']
        label = d['label']
        label = label.squeeze(0)
        annotations = np.zeros((1, 7))
        annotation = mask2boundingbox(label)
        if annotation == 0:
            annotation = annotations
            raise ValueError('Dataloader data no annotations')
            #print("Dataloader data no annotations")
        else:
            # add class label
            cls = d['class']
            annotation = np.array(annotation)
            annotation = np.append(annotation, cls)
            #annotation = np.expand_dims(annotation,0)
        #print(annotation.shape)
        #print(image.shape)
        d['label'] = annotation
        return d

def mask2boundingbox(label):
    if torch.is_tensor(label):
        label = label.numpy()   
    sk_mask = sk_label(label) 
    regions = sk_regions(label.astype(np.uint8))
    #global top, left, low, bottom, right, height 
    #print(regions)
    # check regions is empty
    if not regions:
        return 0

    for region in regions:
        # print('[INFO]bbox: ', region.bbox)
        # region.bbox (x1,y1,z1,x2,y2,z2)
        # top, left, low, bottom, right, height = region.bbox
        y1, x1, z1, y2, x2, z2 = region.bbox
   # return left, top, right, bottom, low, height
    return x1, y1, x2, y2, z1, z2

#-------- Running setting -------- 
'''
def adjust_learning_rate_by_step(optimizer, epoch, init_lr, decay_rate=.5 ,lr_decay_epoch=40):
    #Sets the learning rate to initial LR decayed by e^(-0.1*epochs)
    lr = init_lr * (decay_rate ** (epoch // lr_decay_epoch))
    for param_group in optimizer.param_groups:
        #param_group['lr'] =  param_group['lr'] * math.exp(-decay_rate*epoch)
        param_group['lr'] = lr
        #lr = init_lr * (0.1**(epoch // lr_decay_epoch))
    #print('LR is set to {}'.format(param_group['lr']))
    return optimizer , lr

def adjust_learning_rate(optimizer, epoch, init_lr, decay_rate=.5):
    #Sets the learning rate to initial LR decayed by e^(-0.1*epochs)
    lr = init_lr * decay_rate 
    for param_group in optimizer.param_groups:
        #param_group['lr'] =  param_group['lr'] * math.exp(-decay_rate*epoch)
        param_group['lr'] = lr
        #lr = init_lr * (0.1**(epoch // lr_decay_epoch))
    #print('LR is set to {}'.format(param_group['lr']))
    return optimizer , lr
'''

def train(model, device, data_num, epochs, optimizer, loss_function, train_loader, valid_loader, early_stop, scheduler, check_path):
    # Let ini config file can be writted
    #global best_metric
    #global best_metric_epoch
    #val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    trigger_times = 0

    #epoch_loss_values = list()
    
    writer = SummaryWriter()
    for epoch in range(epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epochs}")
        # record ram memory used
        process = psutil.Process(os.getpid())
        print(f'RAM used:{process.memory_info().rss/ 1024 ** 3} GB')
        model.train()
        epoch_loss = 0
        step = 0

        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data['image'].to(device), batch_data['label'].long().to(device)
            optimizer.zero_grad()
            #inputs, labels = Variable(inputs), Variable(labels)
            outputs = model(inputs)
            #print(f'outputs:{outputs.size()}')
            #print(f'labels:{labels.size()}')
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = data_num // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        config.epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        # Early stopping & save best weights by using validation
        metric = validation(model, valid_loader, device)
        scheduler.step(metric)

        # checkpoint setting
        if metric > best_metric:
            # reset trigger_times
            trigger_times = 0
            best_metric = metric
            best_metric_epoch = epoch + 1
            torch.save(model.state_dict(), f"{check_path}/{best_metric}.pth")
            print('trigger times:', trigger_times)
            print("saved new best metric model")
        else:
            trigger_times += 1
            print('trigger times:', trigger_times)
            # Save last 3 epoch weight
            if early_stop - trigger_times <= 3 or epochs - epoch <= 3:
                torch.save(model.state_dict(), f"{check_path}/{metric}_last.pth")
                print("save last metric model")
        print(
            "current epoch: {} current accuracy: {:.4f} best accuracy: {:.4f} at epoch {}".format(
                epoch + 1, metric, best_metric, best_metric_epoch
            )
        )
        writer.add_scalar("val_accuracy", metric, epoch + 1)

        # early stop 
        if trigger_times >= early_stop:
            print('Early stopping!\nStart to test process.')
            print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
            return model
        
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    config.best_metric = best_metric
    config.best_metric_epoch = best_metric_epoch
    writer.close()
    #print(f'training_torch best_metric:{best_metric}',flush =True)
    #print(f'training_torch config.best_metric:{config.best_metric}',flush =True)
    return model

class AngleLoss_predict(nn.Module):
    def __init__(self, gamma=0):
        super(AngleLoss_predict, self).__init__()
        self.gamma = gamma
        self.it = 1
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        cos_theta, phi_theta = input
        target = target.view(-1, 1)  # size=(B,1)
        index = cos_theta.data * 0.0  # size=(B, Classnum)
        # index = index.scatter(1, target.data.view(-1, 1).long(), 1)
        #index = index.byte()
        index = index.bool()  
        index = Variable(index)
        # index = Variable(torch.randn(1,2)).byte()

        self.lamb = max(self.LambdaMin, self.LambdaMax / (1 + 0.1 * self.it))
        output = cos_theta * 1.0  # size=(B,Classnum)
        output1 = output.clone()
        # output1[index1] = output[index] - cos_theta[index] * (1.0 + 0) / (1 + self.lamb)
        # output1[index1] = output[index] + phi_theta[index] * (1.0 + 0) / (1 + self.lamb)
        output[index] = output1[index]- cos_theta[index] * (1.0 + 0) / (1 + self.lamb)+ phi_theta[index] * (1.0 + 0) / (1 + self.lamb)
        return(output)

def validation(model, val_loader, device):
    #metric_values = list()
    model.eval()
    with torch.no_grad():
        num_correct = 0.0
        metric_count = 0
        for val_data in val_loader:
            val_images, val_labels = val_data['image'].to(device), val_data['label'].to(device)
            val_outputs = model(val_images)
            # base on AngleLoss
            if isinstance(val_outputs, tuple):
                val_outputs = AngleLoss_predict()(val_outputs,val_labels)
            value = torch.eq(val_outputs.argmax(dim=1), val_labels)
            metric_count += len(value)
            num_correct += value.sum().item()
        metric = num_correct / metric_count
        config.metric_values.append(metric)
        #print(f'validation metric:{config.metric_values}',flush =True)
    return metric


def plot_loss_metric(epoch_loss_values,metric_values,save_path):
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    plt.title("Val Accuracy")
    x = [i + 1 for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.savefig(f'{save_path}/train_loss_metric.png')

def kfold_split(file, kfold, seed, type, fold):
    if type == 'pos':
        d = {}
        file_list = ['file']
        file_list.extend([f'pos_split_df_{i}' for i in range(kfold)])
        d['file'] = file
        for i in range(kfold):
            d[f'test_pos_df_{i}'] = d[file_list[i]].groupby(["gender","age_range","spleen_injury_class"],group_keys=False).apply(lambda x: x.sample(frac=1/(kfold-i),random_state=1))
            d[f'pos_split_df_{i}'] = d[file_list[i]].drop(d[f'test_pos_df_{i}'].index.to_list())
        output_file = d[f'test_pos_df_{fold}']

    elif type == 'neg':
        file_list = [f'neg_split_df_{i}' for i in range(kfold)]
        file_list = np.array_split(file.sample(frac=1,random_state=seed), kfold)
        output_file = file_list[fold]
        
    return output_file

def Data_progressing(pos_file, neg_file, box_df, imbalance_data_ratio, data_split_ratio, seed, fold, save_file = False, cropping = True):
    # Pos data progress
    for index, row in pos_file.iterrows():
        if row['OIS']==row['OIS']:
            pos_file.loc[index,'spleen_injury_grade'] = row['OIS']
        else:
            pos_file.loc[index,'spleen_injury_grade'] = row['R_check']

    new_col= 'age_range'
    new_col_2 = 'spleen_injury_class'
    bins = [0,30,100]
    bins_2 = [0,2,5]
    label_2 = ['OIS 1,2','OIS 3,4,5']
    pos_file[new_col] = pd.cut(x=pos_file.age, bins=bins)
    pos_file[new_col_2] = pd.cut(x=pos_file.spleen_injury_grade, bins=bins_2, labels=label_2)

    # positive need select column and split in kfold 
    test_pos_df = kfold_split(pos_file, int(1/data_split_ratio[2]), seed, 'pos', fold)
    train_pos_file = pos_file.drop(test_pos_df.index.to_list())
    valid_pos_df = train_pos_file.groupby(['gender','age_range','spleen_injury_class'],group_keys=False).apply(lambda x: x.sample(frac=data_split_ratio[1]/(1-data_split_ratio[2]),random_state=seed))
    train_pos_df = train_pos_file.drop(valid_pos_df.index.to_list())
    
    # negative only need split in kfold 
    neg_sel_df = neg_file.sample(n=len(pos_file),random_state=seed)
    test_neg_df =  kfold_split(neg_sel_df, int(1/data_split_ratio[2]), seed, 'neg', fold)
    train_neg_file = neg_file.drop(test_neg_df.index.to_list())
    valid_neg_df = train_neg_file.sample(n=len(valid_pos_df),random_state=seed)
    train_neg_df = train_neg_file.drop(valid_neg_df.index.to_list()).sample(n=len(train_pos_df)*imbalance_data_ratio,random_state=seed)

    train_df = pd.concat([train_neg_df,train_pos_df])
    valid_df = pd.concat([valid_neg_df,valid_pos_df])
    test_df = pd.concat([test_neg_df,test_pos_df])

    train_data = box_df[box_df.Path.isin(train_df.source.to_list())]
    valid_data = box_df[box_df.Path.isin(valid_df.source.to_list())]
    test_data = box_df[box_df.Path.isin(test_df.source.to_list())]

    train_df['spleen_injury'] = np.array([0 if i else 1 for i in train_df.spleen_injury_class.isna().tolist()])
    valid_df['spleen_injury'] = np.array([0 if i else 1 for i in valid_df.spleen_injury_class.isna().tolist()])
    test_df['spleen_injury'] = np.array([0 if i else 1 for i in test_df.spleen_injury_class.isna().tolist()])

    if save_file:
        test_df_output = pd.merge(test_data.loc[:,['ID','Path','BBox','Posibility']],test_df,left_on='Path',right_on='source',suffixes = ['','_x'])
        valid_df_output = pd.merge(test_data.loc[:,['ID','Path','BBox','Posibility']],test_df,left_on='Path',right_on='source',suffixes = ['','_x'])
        test_df_output = test_df_output.drop(['ID_x'],axis=1)
        valid_df_output = valid_df_output.drop(['ID_x'],axis=1)
        test_df_output = test_df_output.loc[:,test_df_output.columns[~test_df_output.columns.str.contains('Unnamed')]]
        valid_df_output = valid_df_output.loc[:,valid_df_output.columns[~valid_df_output.columns.str.contains('Unnamed')]]
        valid_df_output.to_csv(f'{save_file}/fold{fold}_valid.csv',index = False)
        test_df_output.to_csv(f'{save_file}/fold{fold}_test.csv',index = False)

    if cropping:
        train_data_dicts = []
        for index,row in train_data.iterrows():
            image = row['Path']
            label = row['BBox']
            train_data_dicts.append({'image':image,'label':label})
        valid_data_dicts = []
        for index,row in valid_data.iterrows():
            image = row['Path']
            label = row['BBox']
            valid_data_dicts.append({'image':image,'label':label})
        test_data_dicts = []
        for index,row in test_data.iterrows():
            image = row['Path']
            label = row['BBox']
            test_data_dicts.append({'image':image,'label':label})
    else:
        train_data_dicts =[
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip([i for i in train_df.source.tolist()], [i for i in train_df.spleen_injury.tolist()] )
        ]
        valid_data_dicts =[
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip([i for i in valid_df_output.source.tolist()], [i for i in valid_df_output.spleen_injury.tolist()] )
        ]
        test_data_dicts =[
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip([i for i in test_df_output.source.tolist()], [i for i in test_df_output.spleen_injury.tolist()] )
        ]

    
    return train_data_dicts, valid_data_dicts, test_data_dicts


class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)      
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重. 当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(FocalLoss, self).__init__()
        if alpha is None:  # alpha 是平衡因子
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, list):
                self.alpha = torch.Tensor(alpha)
            else:
                self.alpha = torch.zeros(class_num)
                self.alpha[0] += alpha
                self.alpha[1:] += (1-alpha)
        self.gamma = gamma  # 指数
        self.class_num = class_num  # 类别数目
        self.size_average = size_average  # 返回的loss是否需要mean一下

    def forward(self, preds, labels):
        """
        focal_loss损失计算        
        :param preds:   预测类别. size:[B,N,C] or [B,C]  分别对应与检测与分类任务, B 批次, N检测框数, C类别数        
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """        
        # assert preds.dim()==2 and labels.dim()==1        
        preds = preds.view(-1,preds.size(-1))        
        self.alpha = self.alpha.to(preds.device)        
        preds_softmax = F.softmax(preds, dim=1) # 这里并没有直接使用log_softmax, 因为后面会用到softmax的结果(当然你也可以使用log_softmax,然后进行exp操作) 
        preds_softmax = preds_softmax.clamp(min=0.0001,max=1.0)  # 避免數值過小 進log後 loss 為nan   
        preds_logsoft = torch.log(preds_softmax)
        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll )        
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))        
        self.alpha = self.alpha.gather(0,labels.view(-1))        
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = torch.mul(self.alpha, loss.t())        
        if self.size_average:        
            loss = loss.mean()        
        else:            
            loss = loss.sum()        
        return loss