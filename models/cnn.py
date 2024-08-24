

import torch
from torch import nn
from torch.nn import functional as TorchFunc

import timm       # for uploading and creating image models

from Configurations import torch_configurations


# Modeling - hybrid model w/ CNN + Metadata
class cnn_metadata(nn.Module):
    def __init__(self, model_name, num_classes=1, pretrained=True):
        super(cnn_metadata, self).__init__()
        self.image_model = timm.create_model(model_name, pretrained=pretrained)
        self.image_out_features = self.image_model.get_classifier().in_features
        self.image_model.reset_classifier(0)  # Remove the original classifier

        # Metadata part
        metadata_input_features = 6
        metadata_output_features = 128

        self.metadata_fc = nn.Sequential(
            nn.Linear(metadata_input_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256 ),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, metadata_output_features),
            nn.BatchNorm1d(metadata_output_features),
            nn.ReLU()
        )

        # Combine features from image model and metadata
        combined_features = self.image_out_features + metadata_output_features
        self.final_fc = nn.Sequential(
            nn.Linear(combined_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )

    def forward(self, image, metadata):
        image_features = self.image_model(image)

        metadata_features = self.metadata_fc(metadata)

        #/////////////////////////////////
        for name, module in self.metadata_fc.named_modules():
            if isinstance(module, nn.BatchNorm1d):
                print(f"BatchNorm1d {name} - Mean: {module.running_mean}, Var: {module.running_var}")
        #/////////////////////////////////

        combined_features = torch.cat((image_features, metadata_features), dim=1)
        output = self.final_fc(combined_features)

        return output
