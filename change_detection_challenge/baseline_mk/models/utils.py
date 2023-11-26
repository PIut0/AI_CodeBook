import segmentation_models_pytorch as smp
import torch.nn as nn

# custom Unet 
class Unet_Front_Conv(nn.Module):
    def __init__(self, classes, encoder_name, encoder_weights, activation):
        super(CustomModel, self).__init__()
        
        # Add a convolutional layer
        pretrained_model = smp.Unet(encoder_name = encoder_name, classes = classes, encoder_weights = encoder_weights, activation = activation)

        self.conv_layer = nn.Conv2d(in_channels = 6, out_channels=3, kernel_size = 1)

        # Add the pre-trained model
        self.pretrained_model = pretrained_model

    def forward(self, x):
        w, h, c = x.shape
        print(w, h, c)
        y = np.zeros_like((w, h, c * 2))
        y[:, :w/2, :3] = x[:, :w/2 , :]
        y[:, w/2: , 3:]= x[:, w/2 : , :]
        x = self.scaler(x)
        x = np.transpose(x, (2, 0, 1))

        # Apply the convolutional layer to input x
        x = self.conv_layer(x)
        
        # Forward pass through the pre-trained model
        x = self.pretrained_model(x)

        return x


def get_model(model_str: str):
    """모델 클래스 변수 설정
    Args:
        model_str (str): 모델 클래스명
    Note:
        model_str 이 모델 클래스명으로 정의돼야함
        `model` 변수에 모델 클래스에 해당하는 인스턴스 할당
    """


    if model_str == 'Unet':
        return smp.Unet

    elif model_str == 'Unet_Front_Conv':
        return Unet_Front_Conv
    
    elif model_str == 'FPN':
        return smp.FPN

    elif model_str == 'DeepLabV3Plus':
        return smp.DeepLabV3Plus
    
    elif model_str == 'UnetPlusPlus':
        return smp.UnetPlusPlus

    elif model_str == 'PAN':
        return smp.PAN

    elif model_str == 'MAnet':
        return smp.MAnet

    elif model_str == 'PSPNet':
        return smp.PSPNet