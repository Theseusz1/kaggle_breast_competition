# RSNA Screening Mammography Breast Cancer Detection

## 1.赛题分析

对于北美放射学会给出的乳腺癌数据集进行图像分类，判断对于患者而言患癌症的问题，本质为图像分类问题。本次赛题的难点在于数据集的高度不平衡(2%左右的正类样本)以及数据集十分庞大(300G+)的dicom文件，所以数据的预处理是本次竞赛中重要的部分。

在这次比赛中有许多数据处理步骤十分巧妙，有的参赛选手使用CBIS外部数据集训练Unet+的分割辅助本次模型的训练，还有的参赛选手使用yolov5(v7)辅助提取CT图片中特定区域，或者使用Opencv库对图片进行roi区域提取。同时，数据增强也是非常重要的方法，因为正类样本数量及其少，一切预处理方法都是为了让模型学习到更多的分类特征。

本赛题评分标准为pf1指标。

## 2.数据集特征

![image-20230301192231196](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20230301192231196.png)

除了图片数据，赛题还提供了更多额外信息，下面依次对其含义进行研究：

site_id:代表图片数据来源的医院

patient_id:代表患者的身份编号

image_id:代表目标图片的编号

laterality:代表乳房图片的左侧或者右侧

view:代表图片拍摄的方向

age:代表目标患者的年龄，以年为单位

implant:代表患者乳房是否有植入物信息

density:代表患者乳房的密度分级指标(只在training的csv数据中存在)

machine_id:代表图片来源的拍摄机器编号

cancer:关键标签，代表患者是否患有癌症

biopsy:代表患者是否进行了后续的活检(只在training的csv数据中存在)

invasive:代表乳房是否对于检验呈阳性，无论是否有侵袭性的癌症(只在training的csv中存在)

BIRADS:代表患者需要后续随访的标签

prediction_id:在提交时用于区分预测值的标签(只存在于测试csv中)

difficult_negative_case:代表是否为极难区分的图片，(只在training的csv中存在)

## 3.思路分析

首先对于图片数据，对于dicom格式是直接送入模型进行训练的，所以我们首先考虑将图片转换为png或jpeg格式的，而对于图片转换应当考虑如下几个问题：

(1).图片转换为png(jpeg)图片的尺寸问题，对于高分辨率图片转换为低分辨率图片的过程，必然存在像素损失，对于阳性样本本来就很少的样本，如果调整为过小的尺寸，则可能会损失大量关键信息。但考虑到gpu设备的内存问题，过大的尺寸必然导致训练速度慢甚至难以进行训练，经过折中考虑，首先选择$512*512$图片进行训练，大致对于训练数据进行探索，后续采用$1024*1024$以及更大尺寸进行训练。

(2).图片转换库的选择问题：常见的dicom转换为png格式的库为pydicom和dicomsdl，而采用gpu转换的库还有nvidia支持的dali库，使用gpu对于转换过程进行加速。对于本次比赛，竞赛要求为笔记本运行时长不允许超过9h，所以选择合适的库就变成了很关键的问题，经过实验，三者的速度为dali>dicomsdl>pydicom，由于对于dali处理图片并不熟悉且所需要的代码量不小，同时dicomsdl转换所有图片大致需要5h，已可以满足<9H的时间需求，故本次竞赛的提交过程和训练过程的数据准备均采用dicomsdl进行编写。

第二是对于数据不平衡问题的处理，常见的思路有：1.自定义采样器，对于样本少的正类样本进行多次采样。2.调整损失函数，对于二分类问题使用带权重的交叉熵损失函数。3.对于损失函数，还可以采用FocalLoss抑制样本不均匀问题。具体使用哪种方法应当采取实验决定，对于不同的预处理方法，可能需要采用的处理不平衡方法也不同，对于不同的模型，FocalLoss和带权重的交叉熵损失函数可能也无绝对的好坏之分，一切处理方法都应当适应实际数据。

第三是对于图片的增强问题，对于pytorch编写的pipeline，常用的图像增强库有torchvision(基于pytorch官方实现)，albumentations(基于torchvision和opencv实现)，Opencv。在kaggle竞赛中albumentations是常用的图像增强库，包含内容十分全面。具体的图像增强同样应取决于数据。

第四是对于模型的选择，对于分类模型，截止目前常用的有resnet系列及其变体，efficientnet系列及其变体，在比赛后期排行榜前列使用很多的Convnext新版本。在竞赛后期经过各个参赛者的交流讨论，模型选择对于比赛成绩的影响并没有很大，数据处理对于结果的影响远大于模型的选择，故在本次比赛中主要采用了efficientnetv2s进行训练，efficientnetv2s优点在于相较于efficientnet有了速度上的提升，同时所需的gpu空间减少，相较resnet有性能上的提升，由于gpu配置有限，没有进行Convnextv2的实验，但从最终结果来看，Convnetxt的性能应略优于efficientnetv2s。

## 4.图片预处理(通过dicomsdl将原始图片转换为png图像)

此处以提交过程的图片处理作为例子：

```python
#对于图片处理需要用到的库进行安装
try:
    import dicomsdl
else:
    !pip install dicomsdl
try:
    import gdcm
else:
    !pip install gdcm
try:
    import pylibjpeg
else:
    !pip install pylibjpeg
#对用到的其他库进行载入
import glob
import shutil
import os
from PIL import Image
import cv2
```

```python
#对于目标图片进行路径筛选
test_images = glob.glob("/kaggle/input/rsna-breast-cancer-detection/test_images/*/*.dcm")
save_path = "/kaggle/temp/"#临时保存temp文件夹中，便于训练过程的图片载入
INPUT_SIZE = 1024#目标图片尺寸
os.makedirs(save_path, exist_ok = True)#创建temp文件夹，如果存在则不操作
```

```python
df = pd.read_csv('/kaggle/input/rsna-breast-cancer-detection/test.csv')#载入test过程所需的csv
```

```python
#编写区域提取函数
def img2roi(img):
    # Binarize the image
    bin_img = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)[1]

    # Make contours around the binarized image, keep only the largest contour
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)

    # Find ROI from largest contour
    ys = contour.squeeze()[:, 0]
    xs = contour.squeeze()[:, 1]
    roi =  img[np.min(xs):np.max(xs), np.min(ys):np.max(ys)]
    
    return roi
```

```python
def process(path, size):
    patient = path.split('/')[-2]
    image = path.split('/')[-1][:-4]
    dicom = dicomsdl.open(path)
    img = dicom.pixelData(storedvalue=False)
    img = (img - img.min()) / (img.max()-img.min())
    #对于不同的背景颜色，转换为相同的背景
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        img = 1.0- img
    img = (img*255).astype(np.uint8)
    img = img2roi(img)
    final_img = Image.fromarray(img)
    #将图片大小转换为1024*512，便于统一输入模型的图片大小
    final_img = final_img.resize((int(INPUT_SIZE/2),int(INPUT_SIZE)),Image.Resampling.LANCZOS)
    final_img.save(save_path+f"{patient}_{image}.png")
```

```python
_ = Parallel(n_jobs=4)(
    delayed(process)(uid, size=INPUT_SIZE)
    for uid in tqdm(test_images)
)#采取多线程处理，加快处理速度
#len(os.listdir(save_path))
#查看是否所有图片都转换完成
```

实际训练过程中使用kaggle制作好的dataset直接进行训练，减少大量时间，但提交过程因为不允许访问互联网，故应当采取上述编写方式进行编写。

## 5.不平衡问题csv的处理

#### 方法1：

```python
#自定义采样器
from torch.utils.data import WeightedRandomSampler
def getweight(cur):
    cancer_weight = (cur.shape[0]-cur.cancer.sum())/cur.cancer.sum()/8
    normal_weight = 1
    weights = []
    for i in range(len(cur)):
        if cur.iloc[i]['cancer'] == 1:
            weights.append(cancer_weight)
        else:
            weights.append(normal_weight)
    return weights
```

在加载数据时，使用实例化后的WeightedRandomSampler作为sampler参数传入dataloader,注意这里dataloader中的shuffle应当为false,否则报错。

#### 方法2：

人为重复癌症数据N次，这里使用3次

```python
df_train = pd.read_csv('/kaggle/input/rsna-breast-cancer-detection/train.csv')
pos = df_train[df_train['cancer'] == 1]
for i in range(3):
    df_train = df_train.append(pos)
df_train.index = range(len(df_train))
```

对于上述两种方法，本质差别不大，仅在于对于重复次数(本质为权重问题)的不同处理，在本次竞赛中使用法1实验发现过拟合验证，考虑参数调整不合适，同时考虑到shuffle和测试集数据分布大致为2%,使用法2稍微进行阳性样本的过采样。

## 6.csv数据的后续处理

使用患者patient_id作为分组条件，分为N_FOLDS组，便于后续进行交叉验证：

```python
CATEGORY_AUX_TARGETS = ['site_id', 'laterality', 'view', 'implant', 'biopsy', 'invasive', 'BIRADS', 'density', 'difficult_negative_case', 'machine_id', 'age']
TARGET = 'cancer'
ALL_FEAT = [TARGET] + CATEGORY_AUX_TARGETS
N_FOLDS = 5
FOLDS = np.array(os.environ.get('FOLDS', '0,1,2,3,4').split(',')).astype(int)
```



```python
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
#分为5组，便于后续5折交叉验证
split = StratifiedGroupKFold(N_FOLDS)
for k, (_, test_idx) in enumerate(split.split(df_train, df_train.cancer, groups=df_train.patient_id)):
    df_train.loc[test_idx, 'split'] = k
df_train.split = df_train.split.astype(int)#将数据属于哪一组标记在csv文件最后一列
df_train.groupby('split').cancer.mean()#查看每一组的癌症患者占比
```

```python
df_train.age.fillna(df_train.age.mean(), inplace=True)#补充年龄数据中缺少项
df_train['age'] = pd.qcut(df_train.age, 10, labels=range(10), retbins=False).astype(int)#将年龄分为10组进行编码
#对其他行进行int类型的编码转换
df_train[CATEGORY_AUX_TARGETS] = df_train[CATEGORY_AUX_TARGETS].apply(LabelEncoder().fit_transform)
```

### 7.图像增强

```python
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2 
#对于训练集的图像增强
augmentation = A.Compose([
    #A.Transpose(p=0.5),#考虑到测试集与训练集的数据分布，不采用transpose变换
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightness(p=0.5, limit=0.2),
    A.RandomContrast(p=0.5, limit=0.2),   
    A.OneOf([
        A.MotionBlur(blur_limit=5),
        A.MedianBlur(blur_limit=5),
        A.GaussianBlur(blur_limit=5),
        A.GaussNoise(var_limit=(0.5, 30))],
        p=0.5),
    A.CLAHE(clip_limit=4.0, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
    A.Resize(1024, 512),
    A.Cutout(max_h_size=int(1024 * 0.1), max_w_size=int(512 * 0.1), num_holes=4, p=0.5),  
    A.Normalize(),
    ToTensorV2(),
])
#对于验证集的图像增强
aug_resize_norm = A.Compose([
    A.Resize(1024, 512),
    A.Normalize(),
    ToTensorV2(),
])
```

## 8.模型的导入

```python
try:
    import timm
except:
    !pip install timm -q
```

## 9.模型的结构改进

```python
class BreastCancerModel(torch.nn.Module):
    def __init__(self, aux_classes, model_type=Config.MODEL_TYPE, dropout=0.):
        super().__init__()
        self.model = create_model(model_type, pretrained=True, drop_rate = 0.4, drop_path_rate = 0.3)
        self.backbone_dim = self.model(torch.randn(1, 3, 1024, 512)).shape[-1]
		#cancer标签的输出头
        self.nn_cancer = torch.nn.Sequential(
            torch.nn.Linear(self.backbone_dim, 1),
        )
        #其他辅助标签的输出头
        self.nn_aux = torch.nn.ModuleList([
            torch.nn.Linear(self.backbone_dim, n) for n in aux_classes
        ])

    def forward(self, x):
        # returns logits
        x = self.model(x)

        cancer = self.nn_cancer(x).squeeze()
        aux = []
        for nn in self.nn_aux:
            aux.append(nn(x).squeeze())
        return cancer, aux

    def predict(self, x):
        cancer, aux = self.forward(x)
        sigaux = []
        for a in aux:
            sigaux.append(torch.softmax(a, dim=-1))
        return torch.sigmoid(cancer), sigaux

AUX_TARGET_NCLASSES = df_train[CATEGORY_AUX_TARGETS].max() + 1
```

这里我们使用竞赛中给的额外数据作为辅助损失训练模型，具体操作方法如下：模型对于imagenet最后的fc层为1280->1000，我们将后面跟上1000->1作为癌症标签的输出头，同理1000->mi作为第i个标签的输出头，在后续计算损失时同时计算癌症加其他标签的损失，然后梯度回传训练模型参数，这样相当于通过辅助标签辅助了主要的癌症标签的训练，而对于loss中两种损失比例的调整便可以调整两者的梯度回传力度。

## 10.DataSet的重写

```python
class BreastCancerDataSet(torch.utils.data.Dataset):
    def __init__(self, df, path, transforms=None):
        super().__init__()
        self.df = df
        self.path = path
        self.transforms = transforms

    def __getitem__(self, i):

        path = f'{self.path}/{self.df.iloc[i].patient_id}_{self.df.iloc[i].image_id}.png'
        try:
            img = cv2.imread(path)
        except Exception as ex:
            print(path, ex)
            return None

        if self.transforms is not None:
            img = self.transforms(image=img)['image']
        else:
            img = img.astype(np.float32)
        if TARGET in self.df.columns:
            cancer_target = torch.as_tensor(self.df.iloc[i].cancer)
            cat_aux_targets = torch.as_tensor(self.df.iloc[i][CATEGORY_AUX_TARGETS])
            return img, cancer_target, cat_aux_targets

        return img

    def __len__(self):
        return len(self.df)
```

## 11.损失函数的选择

法1：BCEFocalLoss

```python
class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.94, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pt, target):
        loss = - self.alpha * ((1 - pt) ** self.gamma) * target * torch.log(pt) - (1 - self.alpha) * (pt ** self.gamma) * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss
```

法2：

```python
 cancer_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                            y_cancer_pred,
                            y_cancer.to(float).to(DEVICE),
                            pos_weight=torch.tensor([config.POSITIVE_TARGET_WEIGHT]).to(DEVICE)
                        )
```

两者其实差别不大，discussion区讨论后认为binary_cross_entropy_with_logits已足以完成本次竞赛工作，通过实验我癌症标签使用Focalloss，而在辅助标签时使用普通cross_entropy作为损失函数，同时两者使用超参数调整总损失。

## 12.其他需要的辅助函数

#### 1.验证指标相关

```python
from sklearn.metrics import accuracy_score
def accuracy(labels, predictions, thr):
    acc = accuracy_score(labels, predictions>thr)
    return acc
def pfbeta(labels, predictions, beta=1.):
    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(labels)):
        prediction = min(max(predictions[idx], 0), 1)
        if (labels[idx]):
            y_true_count += 1
            ctp += prediction
        else:
            cfp += prediction

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / max(y_true_count, 1)  # avoid / 0
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0

def optimal_f1(labels, predictions):
    thres = np.linspace(0, 1, 201)
    f1s = [pfbeta(labels, predictions > thr) for thr in thres]
    idx = np.argmax(f1s)
    return f1s[idx], thres[idx]
```

### 2.模型的保存与加载

```python
def save_model(name, model, thres, model_type):
    torch.save({'model': model.state_dict(), 'threshold': thres, 'model_type': model_type}, save_path+f'{name}')
def load_model(path, model=None):
    data = torch.load(path, map_location=DEVICE)
    if model is None:
        model = BreastCancerModel(AUX_TARGET_NCLASSES, data['model_type'])
    model.load_state_dict(data['model'])
    return model, data['threshold'], data['model_type']
```

### 3.内存释放相关

```python
def gc_collect():
    gc.collect()
    torch.cuda.empty_cache()
```

## 13.其他相关说明

### 1.使用AdamW进行参数更新，使用weight_decay抑制过拟合现象

### 2.使用梯度累积以获得较大的batchsize，因为显卡只有一块TITANXp，最大单次batchsize只能为8

### 3.使用半精训练，提高训练速度

### 4.使用OneCycleLR，设置预热占比为0.1-0.15

### 5.使用的timm模型加载预训练参数(经实验如果pretrained=False,图像极难学习到内容，pf1在0.04左右)

### 6.对efficientnetv2s模型使用dropout和drop path,抑制过拟合现象

### 7.使用wandb实时检测训练进度，同时便于可视化模型表现

### 8.考虑使用TTA获取更高的测试分数

### 9.考虑赛题使用pf1作为打分依据，考虑使用二进制化预测值提高验证分数，采取启发式搜索寻找最佳阈值

