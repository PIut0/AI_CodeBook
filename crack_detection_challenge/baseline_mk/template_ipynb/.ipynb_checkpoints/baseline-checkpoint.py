from glob import glob
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import segmentation_models_pytorch as smp
from tqdm import tqdm, trange

from PIL import Image
import numpy as np
from torchvision import transforms
import albumentations as A
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch.optim as optim
import shutil
import zipfile

class SegDataset(Dataset):
    def __init__(self, image_paths, mask_paths=None, resize=(400, 400), mode='train'):
        '''
        image_paths: 이미지 경로들의 리스트
        mask_paths: 마스크 경로들의 리스트
        size: 이미지와 마스크를 리사이즈를 몇으로 할 지 결정하는 변수
        mode: train인지 test인지를 결정하는 변수
        '''
        self.image_paths = image_paths 
        self.mask_paths = mask_paths 
        self.resize = resize # resize variable
        self.mode = mode # mode: ['train', 'test']

        self.image_transform = transforms.Compose([
                transforms.ToTensor(), # 'tensorize'
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
                # normalize
        ]) 
        
        self.mask_transform = A.Compose([
            
        ])

    def __len__(self):
        return len(self.image_paths) 

    def __getitem__(self, idx):
        image_path = self.image_paths[idx] 
        filename = os.path.basename(image_path) 

        image = Image.open(image_path) # Load image in BGR form
        
        image = image.convert('RGB') # to RGB form
        image_size = torch.tensor(image.size) # store image size

        image = image.resize(self.resize) # resize image

        image = self.image_transform(image).float() # image to torch tensor

        if self.mode == 'train' or self.mode == 'val':
            mask_path = self.mask_paths[idx] 

            mask = Image.open(mask_path) # Gray image, takes values in [0, 59]

            # resize mask, interpolating by "NEAREST" ("LINEAR" makes it out-of-range)
            mask = mask.resize(self.resize, resample=Image.NEAREST) 

            # mask to tensor, in LONG (integer) format
            mask = torch.from_numpy(np.array(mask)).long()

        # test    
        else:
            mask = np.zeros(self.resize) # retuns the all-zero tensor

        return image, mask, filename, image_size # original image size
    
    
# --------------------------------------------------
# --------------------------------------------------
# --------------------------------------------------
def visualize_batch(images, masks):
    # 이미지와 마스크를 그리드로 만듭니다.
    image_grid = make_grid(images, nrow=4, normalize=True) # 이미지 그리드 생성
    mask_grid = make_grid(masks.unsqueeze(1).float(), nrow=4, normalize=False, scale_each=False) # 마스크 그리드 생성

    # 이미지와 마스크를 시각화합니다.
    plt.figure(figsize=(15, 10))
    plt.subplot(211)
    plt.imshow(image_grid.permute(1, 2, 0)) # 이미지를 시각화합니다.
    plt.title('Images')
    plt.axis('off')

    plt.subplot(212)
    plt.imshow(mask_grid[0], interpolation='nearest') # 마스크를 시각화합니다.
    plt.title('Masks')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# --------------------------------------------------
# --------------------------------------------------
# --------------------------------------------------
def iou(predict_mask: torch.Tensor, mask: torch.Tensor) -> float:
    SMOOTH = 1e-6
    
    batch_size = predict_mask.size()[0]
    
    intersection = ((predict_mask.int() == 1) & (mask.int() == 1) & (predict_mask.int() == mask.int())).float()
    
    intersection = intersection.view(batch_size, -1).sum(1)
    
    union = ((predict_mask.int() == 1) | (mask.int() == 1)).float()
    union = union.view(batch_size, -1).sum(1)

    iou = (intersection + SMOOTH) / (union + SMOOTH)
    
    return iou.mean()

# --------------------------------------------------
# --------------------------------------------------
# --------------------------------------------------
def get_model():
    # return smp.UnetPlusPlus(classes=2, encoder_name = 'timm-efficientnet-b4', encoder_weights = 'noisy-student')
    return smp.DeepLabV3Plus(classes = 2, encoder_name = 'timm-efficientnet-b4', encoder_weights = 'noisy-student')
    

# --------------------------------------------------
# --------------------------------------------------
# --------------------------------------------------
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

# --------------------------------------------------
# --------------------------------------------------
# --------------------------------------------------
def Baseline():
    data_directory = '/scratch/network/mk8574/crack_detection_challenge/data'

    train_image_directory = os.path.join(data_directory, 'train', 'images') 
    train_mask_directory = os.path.join(data_directory, 'train', 'masks') 
    test_image_directory = os.path.join(data_directory, 'test', 'images')


    train_image_paths = sorted(glob(os.path.join(train_image_directory, '*.jpg')))
    train_mask_paths = sorted(glob(os.path.join(train_mask_directory, '*.png')))
    test_image_paths = sorted(glob(os.path.join(test_image_directory, '*.jpg')))

    # train-val split
    train_image_paths, val_image_paths, train_mask_paths, val_mask_paths = train_test_split(train_image_paths, train_mask_paths, test_size=0.2, random_state=2021)

    # save to dataset
    train_dataset = SegDataset(train_image_paths, train_mask_paths, mode='train') 
    val_dataset = SegDataset(val_image_paths, val_mask_paths, mode='val') 
    test_dataset = SegDataset(test_image_paths, mode='test') 

    # save to dataloader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=10) 
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=10) 
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=10) 

    # set CUDA device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define model
    model = get_model().to(device)

    # Define optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Define loss function
    loss_function = DiceLoss()
        # nn.CrossEntropyLoss()

    # Define # Epochs
    num_epochs = 30 
    best_acc = 0

    # Train
    for epoch in range(num_epochs):
        model.train()

        train_accs = []

        for image, mask, filename, image_size in tqdm(train_loader, position=0, leave=True, desc=f'epoch: {epoch} | train'):

            image = image.to(device) # image: [B, C, H, W] 
            mask = mask.to(device) # [B, H, W]


            pred_mask = model(image) # [B, C, H, W]

            loss = loss_function(pred_mask, mask)

            optimizer.zero_grad() # init grad
            loss.backward() # compute grad
            optimizer.step() # update param

            pred_mask = torch.argmax(pred_mask, dim=1) # [B, C, H, W] -> [B, H, W]

            # compute accuracy
            batch_acc = iou(pred_mask, mask) 
            # (pred_mask == mask).float().mean().cpu().item()
            train_accs.append(batch_acc)

        # validation #
        model.eval() # set to val mode 

        val_accs = []

        for image, mask, filename, image_size in tqdm(val_loader, position=0, leave=True, desc=f'epoch: {epoch} | val'):

                image = image.to(device) # [B, C, H, W].
                mask = mask.to(device) # [B, H, W]

                pred_mask = model(image) # [B, C, H, W]

                pred_mask = torch.argmax(pred_mask, dim=1) # [B, C, H, W] -> [B, H, W]

                batch_acc =  iou(pred_mask, mask)
                # (pred_mask == mask).float().mean().cpu().item()
                val_accs.append(batch_acc)

        train_acc = sum(train_accs) / len(train_accs)
        val_acc = sum(val_accs) / len(val_accs)

        print(f'Epoch : {epoch}, train_acc : {train_acc}, val_acc : {val_acc}')

        # store best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), '/scratch/network/mk8574/crack_detection_challenge/baseline_mk/results/best_model.pt')
            print(f'Best performance at epoch {epoch} : {best_acc}')

            
    # Inference #
    model = get_model().to(device)
    model.load_state_dict(torch.load('/scratch/network/mk8574/crack_detection_challenge/baseline_mk/results/best_model.pt'))
    
    # SAVE FILE
    pred_directory = '/scratch/network/mk8574/crack_detection_challenge/baseline_mk/results/pred'

    # 이미 경로가 있으면 삭제합니다.
    if os.path.isdir(pred_directory):
        shutil.rmtree(pred_directory)

    # 경로를 생성합니다.
    os.makedirs(pred_directory, exist_ok=True)

    for image, _, filename, image_size in tqdm(test_loader, position=0, leave=True, desc=f'Prediction'):
        # 가져온 데이터를 장치에 할당합니다.
        image = image.to(device)

        # 모델의 출력값을 계산합니다.
        pred_mask = model(image)

        # argmax 연산을 통해 확률이 가장 높은 클래스를 예측값으로 선택합니다.
        pred_mask = torch.argmax(pred_mask, dim=1)

        for i, a_pred_mask in enumerate(pred_mask):
            # pred_mask를 PIL image로 변환합니다.
            pred_mask_image = Image.fromarray(np.uint8(a_pred_mask.cpu().numpy()))

            # 원본 이미지의 크기로 resize합니다.
            pred_mask_image = pred_mask_image.resize(image_size[i])

            filename_ = filename[i].replace('.jpg', '.png')
            # 이미지를 저장합니다.
            pred_mask_image.save(f'{pred_directory}/{filename_}')
            
    pred_files = glob(f'{pred_directory}/*.png')

    # 압축을 수행합니다.
    with zipfile.ZipFile('/scratch/network/mk8574/crack_detection_challenge/baseline_mk/results/sample_submission.zip', 'w') as zip:
        for pred_file in pred_files:
            zip.write(pred_file, os.path.basename(pred_file))            
            
if __name__ == '__main__':
    Baseline()