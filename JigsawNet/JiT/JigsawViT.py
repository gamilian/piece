import torch
import torch.nn as nn
import timm
from torch.cuda.amp import autocast


class JigsawViT(nn.Module):

    def __init__(self, pretrained_cfg_file, num_labels=2):
        super(JigsawViT, self).__init__()
        # pretrained_cfg = timm.models.create_model(
        #     'crossvit_base_240.in1k').default_cfg
        # pretrained_cfg['file'] = pretrained_cfg_file
        # self.model = timm.models.create_model(
        #     'crossvit_base_240.in1k', pretrained=True, num_classes=num_labels, pretrained_cfg=pretrained_cfg)
        pretrained_cfg = timm.models.create_model(
            'pit_s_distilled_224.in1k').default_cfg
        pretrained_cfg['file'] = pretrained_cfg_file
        self.model = timm.models.create_model(
            'pit_s_distilled_224.in1k', pretrained=True, num_classes=num_labels, pretrained_cfg=pretrained_cfg)
        
    # @autocast()
    def forward(self, x):
        output = self.model(x)
        return output


if __name__ == "__main__":
    pass
