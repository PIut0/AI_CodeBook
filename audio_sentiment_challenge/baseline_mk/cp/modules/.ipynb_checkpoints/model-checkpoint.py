import torch.nn as nn
from transformers import AutoModelForAudioClassification

class MyModel(nn.Module):
    def __init__(self, pretrained_name: str, mode='train'):
        super(MyModel, self).__init__()
        #self.model = AutoModelForAudioClassification.from_pretrained(pretrained_name, num_labels=6,ignore_mismatched_sizes=True) #변경
        self.model = AutoModelForAudioClassification.from_pretrained(pretrained_name)
        
        if mode=='train':
            self.model.fc =  nn.Linear(in_features=7, out_features=6, bias=True) # 변경
            print(self.model)
            #self.model.classifier = nn.Linear(in_features=self.model.projector.out_features, out_features=6)
            nn.init.kaiming_normal_(self.model.fc.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(self.model.fc.bias)
            # self.model.projector.apply(self._init_weight_and_bias)
    def forward(self, x):
        output = self.model(x)
        return output.logits
    
    def save_pretrained(self, path):
        self.model.save_pretrained(path)
        
    def _init_weight_and_bias(self, module):                        
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.model.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)   
    