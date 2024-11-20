from CLIP.clip import clip
from PIL import Image

_, preprocess = clip.load("ViT-B/32")

class FashionIndioDataset():
    def __init__(self, list_image_path, list_txt):
        self.image_path = list_image_path
        self.label = list_txt

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        # Preprocess image using CLIP's preprocessing function
        image = preprocess(Image.open(self.image_path[idx]))
        label = self.label[idx]

        return image, label
