'''
train_loader = get_dataloader(rootdir= './CASIA/', batch_size=128)

root_dir
batch_size
수정 필요
'''

import torch
from torch import Tensor, nn, optim
from torch.nn import Parameter
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader 
from typing import Optional, Union, List, Type, Iterable

import torchvision
from torchvision import transforms, datasets

import numpy as np
import random
import math
import time
import os
import threading
import mxnet as mx
import queue as Queue
import numbers


#seed
random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)


#dataload: insightface
def get_dataloader(
    root_dir: str,
    batch_size: int,
    local_rank: int = 0,) -> Iterable:

    train_set = MXFaceDataset(root_dir=root_dir, local_rank=local_rank)
    train_loader = DataLoaderX(
            local_rank=local_rank,
            dataset=train_set,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )
       
    print('Total data : '+str(len(train_set)))    
    return train_loader



class MXFaceDataset(Dataset):
    def __init__(self, root_dir, local_rank):
        super(MXFaceDataset, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]

        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)
    
class DataLoaderX(DataLoader):

    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank, non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch
    
class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


#backbone
class Basicneck(nn.Module):
    expansion: int = 1
    def __init__(
        self,
        num_filters : int,
        out_filters : int, 
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        ):
      
        super(Basicneck, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, out_filters, kernel_size=3, bias=False)
        self.bn1 = nn.BatchNorm2d(out_filters)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_filters, out_filters, kernel_size=3, bias=False)
        self.bn2 = nn.BatchNorm2d(out_filters)        
        
    def forward(self, x: Tensor):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)

        return out

    
class Bottleneck(nn.Module):
    expansion: int = 4
    def __init__(
        self,
        num_filters : int,
        out_filters : int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(num_filters, out_filters, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_filters)        
        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=stride, padding=1, groups=1, bias=False, dilation=1)
        self.bn2 = nn.BatchNorm2d(out_filters)                               
        self.conv3 = nn.Conv2d(out_filters, out_filters*self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_filters*self.expansion)                               
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride                               
                               
    def forward(self, x: Tensor):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)

        return out

    
    
class ResNet(nn.Module):
    fc_scale = 4*4
    def __init__(
        self,
        block: Type[Union[Basicneck, Bottleneck]],
        layers: List[int],
        num_classes: int = 10572, #casia
        dropout = 0
    ):
        super(ResNet, self).__init__()
        self.num_filters = 64

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = self.num_filters, kernel_size = 7, stride = 2, padding = 3, bias=False)    
        self.bn1 = nn.BatchNorm2d(self.num_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)    
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.bn2 = nn.BatchNorm2d(512 * block.expansion)
        self.dropout = nn.Dropout(p = dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_classes)
        self.bn3 = nn.BatchNorm1d(num_classes)         

                    
    def _make_layer(self,
                    block: Type[Union[Basicneck, Bottleneck]],
                    out_filters: int,
                    blocks: int,
                    stride: int = 1, 
                   ) -> nn.Sequential:
        
        downsample = None
            
        if stride != 1 or self.num_filters != out_filters * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels = self.num_filters, out_channels = out_filters* block.expansion, kernel_size = 1,stride = stride, bias=False),
                nn.BatchNorm2d(out_filters* block.expansion)
            )

        layers = []
        layers.append(block(self.num_filters, out_filters, stride = stride, downsample = downsample))        
    
        self.num_filters = out_filters * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.num_filters, out_filters))

        return nn.Sequential(*layers)
    
    def forward(self, x: Tensor):

        x = self.conv1(x)   
        x = self.bn1(x)  
        x = self.relu(x)    
        x = self.maxpool(x)        
        
        x = self.layer1(x) 
        x = self.layer2(x)       
        x = self.layer3(x)       
        x = self.layer4(x)
        
        #BN-Dropout-FC-BN
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x.float())
        x = self.bn3(x)

        return x
    
    
    
def _resnet(block, layers, **kwargs):    
    model = ResNet(block, layers, **kwargs)
    return model

def resnet18(**kwargs):
    return _resnet(Basicneck, [2, 2, 2, 2], **kwargs)

def resnet34(**kwargs):
    return _resnet(Basicneck, [3, 4, 6, 3], **kwargs)

def resnet50(**kwargs):
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


class IBottleneck(nn.Module):
    expansion = 4
    def __init__(
        self,
        num_filters : int,
        out_filters : int, 
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        ):

        super(IBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv1 = nn.Conv2d(num_filters, out_filters, kernel_size=3, bias=False)
        self.bn2 = nn.BatchNorm2d(out_filters)
        self.prelu = nn.PReLU(out_filters)
        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=stride, padding=1, grousp=1, bias=False, dilation=1)
        self.bn3 = nn.BatchNorm2d(out_filters)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        identity = x
        
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        return out


class IResNet(nn.Module):
    fc_scale = 4 * 4
    def __init__(
        self,
        block,
        layers: List[int],
        num_classes: int = 10572, #casia
        dropout = 0
        ):     
        
        super(IResNet, self).__init__()
        self.num_filters = 64
                 
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = self.num_filters, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(self.num_filters)
        self.prelu = nn.PReLU(self.num_filters)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
                 
        self.bn2 = nn.BatchNorm2d(512 * block.expansion)
        self.dropout = nn.Dropout(p = dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_classes)
        self.bn3 = nn.BatchNorm1d(num_classes)                  
                 
    def _make_layer(self,
                    block,
                    out_filters: int,
                    blocks: int,
                    stride: int = 1, 
                   ) -> nn.Sequential:
                 
        downsample = None
                 
        if stride != 1 or self.num_filters != out_filters * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels = self.num_filters, out_channels = out_filters * block.expansion, kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_filters * block.expansion)
            )
                 
        layers = []
        layers.append(block(self.num_filters, out_filters, stride, downsample))
                 
        self.num_filters = out_filters * block.expansion
                 
        for _ in range(1, blocks):
            layers.append(block(self.num_filters, out_filters))

        return nn.Sequential(*layers)                 
                 
    def forward(self, x: Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        #BN-Dropout-FC-BN
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)  
        x = self.fc(x) 
        x = self.bn3(x)
        
        return x
                 
def _iresnet(block, layers, **kwargs):
    model = IResNet(block, layers, **kwargs)
    return model

def iresnet100( **kwargs):
    return _iresnet(IBottleneck, [3, 13, 30, 3], **kwargs)




#loss
class ArcFace(nn.Module):
    def __init__(self, batch_size = 512, num_classes = 10572, s = 64.0, m = 0.5,  *args, **kwargs):
        super(ArcFace, self).__init__()
        self.s = s
        self.m = m
        self.w = Parameter(Tensor(batch_size, num_classes))
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.sin_pi_m = math.sin(math.pi - m)
        
    def forward(self, x: Tensor, labels: Tensor):
        x_norm = F.normalize(x) #batch_size x num_classes
        w_norm = F.normalize(self.w) #batch_size x num_classes
        cos_th = torch.matmul(torch.transpose(w_norm, 0, 1), x_norm)  #num_classes x num_classes
        cos_th_fc = nn.Linear(in_features = 10572, out_features = 1)(cos_th) #num_classes x 1
#        cos_th_fc = F.linear(x_norm , w_norm) #num_classes x 1
        sin_th = torch.sqrt(1.0 - torch.pow(cos_th, 2)) #num_classes x 1     
        
        cos_th_m = torch.where(cos_th > 0, cos_th_fc * self.cos_m - sin_th * self.sin_m, cos_th_fc - m * self.sin_pi_m) #num_classes x 1
 
        one_hot = F.one_hot(labels, 10572) #batch_size x num_classes
        one_hot = one_hot.float()
        logits = torch.matmul(one_hot,cos_th_m) + torch.matmul((1 - one_hot), cos_th_fc) # logits = one_hot * (cos_th_m - cos_x) + cos_th_fc
        logits = logits * self.s

        return logits

    
class CosFace(nn.Module):
    def __init__(self, s = 64.0, m = 0.40):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, x: Tensor, labels: Tensor):
        x_norm = F.normalize(x) #batch_size x num_classes
        w_norm = F.normalize(self.w) #batch_size x num_classes
        cos_th = torch.matmul(torch.transpose(w_norm, 0, 1), x_norm, out = torch.Tensor.to('cuda'))  #num_classes x num_classes
        cos_th_fc = nn.Linear(in_features = 10572, out_features = 1)(cos_th) #num_classes x 1
        
        final_cos = cos_th_fc - self.m
        
        one_hot = F.one_hot(labels, 10572) #batch_size x num_classes
        one_hot = one_hot.float()
        logits = torch.matmul(one_hot,final_cos) + torch.matmul((1 - one_hot), cos_th_fc)        
        logits = logits * self.s
        
        return logits


#train
def train_epoch(model, lossf, optimizer, optimizer_lossf, data_loader, loss_history):
    total_samples = len(data_loader.dataset)
    model.train()
    lossf.train()
 
    for i, (data, target) in enumerate(data_loader):
        data=data.to('cuda')
        target=target.to('cuda')
               
        optimizer.zero_grad()
        optimizer_lossf.zero_grad()

        arcface = lossf(x = model(data), labels = target)
        loss=nn.CrossEntropyLoss()(arcface, target)
        loss.backward()
        
        optimizer.step()
        optimizer_lossf.step()

        if i % 1000000 == 0:
            print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))
            loss_history.append(loss.item())
    scheduler.step()
    
'''  verification.py로 대체          
#test
def evaluate(model, data_loader, loss_history):
    model.eval()
    model=model.to('cuda')
    
    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0

    with torch.no_grad():
        for i, (data, target) in enumerate(data_loader):
            data=data.to("cuda")
            target=target.to("cuda")
            
            loss = ArcFace(model(data), target, reduction='sum') #reduction
            _, pred = torch.max(model(data), dim=1)          
    
            total_loss += loss.item()
        
            correct_pred += pred.eq(target).sum()

        avg_loss = total_loss / total_samples
        acc = '{:4.2f}'.format(100.0 * correct_pred / total_samples)
        
        loss_history.append(avg_loss)
        
        print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
          '\nAccuracy:' + '{:5}'.format(correct_pred) + '/' +'{:5}'.format(total_samples) +
              ' (' + acc + '%)\n')
'''
      
#Parameter
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "1"

root_dir= './CASIA/'
batch_size = 64
num_classes = 10572
train_loader = get_dataloader(root_dir, batch_size)

N_EPOCHS = 16
model = resnet50(num_classes=num_classes)
lossf = ArcFace(batch_size=batch_size, num_classes=num_classes)
model.to('cuda')
lossf.to('cuda')

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
optimizer_lossf = optim.SGD(lossf.parameters(), lr=0.1, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.1)

train_loss_history, test_loss_history = [], []
start_time = time.time()

for epoch in range(1, N_EPOCHS + 1):
    print('Epoch:', epoch)
    train_epoch(model, lossf, optimizer, optimizer_lossf, train_loader, train_loss_history)
#     evaluate(model, val_loader, test_loss_history)

print('\nExecution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')

torch.save(model.state_dict(), './arcface_BR_trained.pth')
print('\nsaved as "arcface_BR_trained.pth"')