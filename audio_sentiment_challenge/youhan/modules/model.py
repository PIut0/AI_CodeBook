import torch.nn as nn
from transformers import AutoModelForAudioClassification

class MyModel(nn.Module):
    def __init__(self, pretrained_name: str, mode='train'):
        super(MyModel, self).__init__()
        self.model = AutoModelForAudioClassification.from_pretrained(pretrained_name, num_labels=6)
        
        if mode=='train':
            self.model.classifier = nn.Linear(in_features=self.model.projector.out_features, out_features=6)
            nn.init.kaiming_normal_(self.model.classifier.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(self.model.classifier.bias)

    def forward(self, x):
        output = self.model(x)
        return output.logits
    
    def save_pretrained(self, path):
        self.model.save_pretrained(path)