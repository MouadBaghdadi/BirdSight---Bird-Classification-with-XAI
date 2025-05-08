import torch.nn as nn
from transformers import SwinForImageClassification

class SwinClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SwinClassifier, self).__init__()
        self.backbone = SwinForImageClassification.from_pretrained(
            "microsoft/swin-large-patch4-window12-384-in22k",
        )
        self.classifier = nn.Linear(self.backbone.classifier.in_features   , num_classes)

    def forward(self, x):
        outputs = self.backbone(pixel_values=x)
        pooled_output = outputs.pooler_output
        return self.classifier(pooled_output)





