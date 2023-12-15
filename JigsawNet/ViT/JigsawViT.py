import torch.nn as nn
from transformers import ViTModel, ViTConfig
from torch.cuda.amp import autocast

class JigsawViT(nn.Module):

  def __init__(self, pretrained_model, config=ViTConfig(), num_labels=2):
        super(JigsawViT, self).__init__()

        self.vit = ViTModel.from_pretrained(pretrained_model, add_pooling_layer=False)
        self.classifier = (
            nn.Linear(config.hidden_size, num_labels) 
        )
  # @autocast()
  def forward(self, x):

    x = self.vit(x)['last_hidden_state']
    # Use the embedding of [CLS] token
    output = self.classifier(x[:, 0, :])

    return output

if __name__ == "__main__":
    pretrained_model = '/work/csl/code/piece/model/vit-base-patch16-224-in21k'
    model = ViTModel.from_pretrained(pretrained_model, add_pooling_layer=False, num_labels=2)
    print(model)
    print("---------------")
    print(model.config)
