import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel


class ClipCaptain(nn.Module):
    def __init__(self, clip_name='openai/clip-vit-base-patch32'):
        super(ClipCaptain, self).__init__()
        self.model_name = clip_name
        self.model = CLIPModel.from_pretrained(self.model_name)
        # self.processor = CLIPProcessor.from_pretrained(self.model_name)

    def forward(self, x):
        # x = self.processor(text=None, images=x, return_tensors="pt", padding=True)
        # x = self.model.get_image_features(**x)[0]
        x = self.model.get_image_features(x, output_hidden_states=True)
        return x


if __name__ == '__main__':
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    image = torch.randint(low=-2, high=2, size=(15, 3, 224, 224))

    # image = processor(text=None, images=image, return_tensors="pt", padding=True)

    with torch.no_grad():
        image_features = model.get_image_features(image)
    print(image_features.shape)
