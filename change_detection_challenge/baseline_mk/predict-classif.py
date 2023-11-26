"""
Predict
"""
from datetime import datetime
from tqdm import tqdm
import numpy as np
import random, os, sys, torch, cv2, warnings
from glob import glob
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from segmentation_models_pytorch.base import SegmentationModel

prj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(prj_dir)

from modules.utils import load_yaml, save_yaml, get_logger
from modules.scalers import get_image_scaler
from modules.datasets import SegDataset
from models.utils import get_model

warnings.filterwarnings('ignore')

def decompose(img):
    B, C, W, H = img.shape

    half_height = H // 2
    return img[:, :, :, :half_height], img[:, :, :, half_height:]

class Difference_Model(SegmentationModel):       
    def __init__(self):
        super().__init__()
        config_path = os.path.join('./config', 'train.yaml')
        config = load_yaml(config_path)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu_num'])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = smp.DeepLabV3Plus(classes=4 , # config['n_classes'] 
                encoder_name=config['encoder'],
                encoder_weights=config['encoder_weight'],
                activation=config['activation']).to(device)
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder
        self.segmentation_head = self.model.segmentation_head
        self.classification_head = self.model.classification_head
        
        return
        
    def forward(self, x):
        # # split x into two
        a, b = decompose(x)
        # process as input
        a = self.encoder(a)
        b = self.encoder(b)

        # compute differences
        features = list(a[i] - b[i] for i in range(len(a)))

        # process as output
        result = self.decoder(*features)
        
        masks = self.segmentation_head(result)
        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels
            
        return masks
    
    def state_dict(self):
        return self.model.state_dict()
            
    def load_state_dict(self, *checkpoint):
        self.model.load_state_dict(*checkpoint)
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder
        self.segmentation_head = self.model.segmentation_head
        self.classification_head = self.model.classification_head
        
        return


if __name__ == '__main__':

    #! Load config
    config = load_yaml(os.path.join(prj_dir, 'config', 'predict.yaml'))
    train_config = load_yaml(os.path.join(prj_dir, 'results', 'train', config['train_serial'], 'train.yaml'))
    
    #! Set predict serial
    pred_serial = config['train_serial'] + '_' + datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set random seed, deterministic
    torch.cuda.manual_seed(train_config['seed'])
    torch.manual_seed(train_config['seed'])
    np.random.seed(train_config['seed'])
    random.seed(train_config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set device(GPU/CPU)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu_num'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create train result directory and set logger
    pred_result_dir = os.path.join(prj_dir, 'results', 'pred', pred_serial)
    pred_result_dir_mask = os.path.join(prj_dir, 'results', 'pred', pred_serial, 'mask')
    os.makedirs(pred_result_dir, exist_ok=True)
    os.makedirs(pred_result_dir_mask, exist_ok=True)

    # Set logger
    logging_level = 'debug' if config['verbose'] else 'info'
    logger = get_logger(name='train',
                        file_path=os.path.join(pred_result_dir, 'pred.log'),
                        level=logging_level)

    # Set data directory
    test_dirs = '../data/test' # os.path.join(prj_dir, 'data', 'test')
    test_img_paths = glob(os.path.join(test_dirs, 'x', '*.png'))

    #! Load data & create dataset for train 
    test_dataset = SegDataset(paths=test_img_paths,
                            input_size=[train_config['input_width'], train_config['input_height']],
                            scaler=get_image_scaler(train_config['scaler']),
                            mode='test',
                            logger=logger)

    # Create data loader
    test_dataloader = DataLoader(dataset=test_dataset,
                                batch_size=config['batch_size'],
                                num_workers=config['num_workers'],
                                shuffle=False,
                                drop_last=False)
    logger.info(f"Load test dataset: {len(test_dataset)}")

    # Load architecture
    model = Difference_Model()
    logger.info(f"Load model architecture: {train_config['architecture']}")

    #! Load weight
    check_point_path = os.path.join(prj_dir, 'results', 'train', config['train_serial'], 'model.pt')
    check_point = torch.load(check_point_path)
    model.load_state_dict(check_point['model'])
    logger.info(f"Load model weight, {check_point_path}")

    # Save config
    save_yaml(os.path.join(pred_result_dir, 'train_config.yml'), train_config)
    save_yaml(os.path.join(pred_result_dir, 'predict_config.yml'), config)
    
    # Predict
    logger.info(f"START PREDICTION")

    model.eval()

    with torch.no_grad():

        for batch_id, (x, orig_size, filename) in enumerate(tqdm(test_dataloader)):
            
            x = x.to(device, dtype=torch.float)
            y_pred = model(x)
            softmax = torch.nn.Softmax(dim=1)
            y_pred = softmax(y_pred)

            y_pred_expanded_0 = torch.cat([y_pred[:,0,:,:],y_pred[:,0,:,:]], dim=2)
            y_pred_expanded_1 = torch.cat([torch.zeros_like(y_pred[:,1,:,:]),y_pred[:,1,:,:]], dim=2)
            y_pred_expanded_2 = torch.cat([y_pred[:,2,:,:],torch.zeros_like(y_pred[:,2,:,:])], dim=2)
            y_pred_expanded_3 = torch.cat([torch.zeros_like(y_pred[:,3,:,:]),y_pred[:,3,:,:]], dim=2)

            y_pred_expanded = torch.stack([y_pred_expanded_0,y_pred_expanded_1,y_pred_expanded_2,y_pred_expanded_3], dim=1)

            y_pred_argmax = y_pred.argmax(1).cpu().numpy().astype(np.uint8)
            orig_size = [(orig_size[0].tolist()[i], orig_size[1].tolist()[i]) for i in range(len(orig_size[0]))]
            # Save predict result
            for filename_, orig_size_, y_pred_ in zip(filename, orig_size, y_pred_argmax):
                resized_img = cv2.resize(y_pred_, [orig_size_[1], orig_size_[0]], interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(os.path.join(pred_result_dir_mask, filename_), resized_img)
    logger.info(f"END PREDICTION")