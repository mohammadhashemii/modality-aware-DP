import os 
import json
from types import SimpleNamespace
import warnings
warnings.filterwarnings("ignore")
import argparse
import time
from CLIP.clip import clip
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

from tqdm import tqdm

import opacus
from opacus.accountants import RDPAccountant
from opacus.validators import ModuleValidator
from opacus.accountants.utils import get_noise_multiplier

from fastDP import PrivacyEngine

from models import ImageClassifier, CLIPModel, ImageEncoder
from transformers import DistilBertTokenizer

from dataset import FashionIndioDataset
from utils import *


class PrivateModel:
    def __init__(self, configs, template = 'a photo of a {}.'):
        
        self.configs = configs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.template = template

        self.loss_img = nn.CrossEntropyLoss()
        self.loss_txt = nn.CrossEntropyLoss()

        self.losses = []


    def load_model(self, modality="all"):

        self.modality = modality

        if modality == "text":
            print("TODO")

        if modality == "all":
            self.model = CLIPModel().to(self.device)

        elif modality == "image":
            # self.model = ImageClassifier(num_classes=self.num_classes).to(self.device)
            self.model = ImageEncoder(model_name=self.configs.image_encoder,
                                      num_classes=self.num_classes, 
                                      pretrained=self.configs.pretrained, 
                                      trainable=self.configs.trainable).to(self.device)
            
    
        if self.configs.continue_training:
            checkpoint_path = f"saved_ours/resnet_dp_eps{self.configs.epsilon}_epochs10.pth"
            print(f"Loading the model weights from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.configs.epochs += 10


        if self.configs.private:
            self.model = ModuleValidator.fix(self.model)


    def load_data(self, dataset):
        
        
        _, preprocess = clip.load("ViT-B/32")
        cifar10 = datasets.CIFAR10(os.path.expanduser("data/cifar10/"), transform=preprocess, download=False)
        
        if dataset == "cifar10":
            training_data = datasets.CIFAR10(
            root="data/cifar10",
            train=True,
            download=False,
            transform=preprocess)
        
            test_data = datasets.CIFAR10(
                root="data/cifar10",
                train=False,
                download=False,
                transform=preprocess
            )
            self.training_size=len(training_data)
            self.testing_size=len(test_data)
            self.classes = cifar10.classes
            self.num_classes = len(self.classes)

        train_dataloader = DataLoader(training_data, batch_size=self.configs.bs, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=self.configs.bs, shuffle=True)
        

        if dataset == "fashion_indio":
            json_dir = 'fashion_indio/data/'
            image_dir = 'fashion_indio/data/images/'
            unique_labels = set()

        
            for split in ['train', 'test']:
                with open(json_dir + split + '_data.json', 'r') as f:
                    input_data = []
                    for line in f:
                        obj = json.loads(line)
                        input_data.append(obj)


                list_image_path = []
                list_label = []
                list_caption = []
                for item in input_data:
                    img_path = image_dir + split + '/' + item['image_path'].split('/')[-1]
                    label = item['class_label']
                    list_image_path.append(img_path)
                    list_label.append(label)
                    list_caption.append(self.template.format(label))
                    unique_labels.add(label)

                if split == 'train':
                    data = FashionIndioDataset(list_image_path, list_label)
                    train_dataloader = DataLoader(data, batch_size=self.configs.bs, shuffle=True)
                    self.training_size=len(data)

                elif split == 'test':
                    data = FashionIndioDataset(list_image_path, list_label)
                    test_dataloader = DataLoader(data, batch_size=self.configs.bs, shuffle=True)
                    self.testing_size=len(data)


            self.classes = {cls: i for i, cls in enumerate(unique_labels)}
            self.num_classes = len(unique_labels)
    
        self.n_acc_steps = 1000 // args.bs # gradient accumulation steps
        
        return train_dataloader, test_dataloader
    

    def set_optimizer(self, epsilon=10, C=1.0):

        if self.configs.private:
            self.delta = 1/2/self.training_size
            self.C = C
            print("Epsilon: ", self.configs.epsilon)
            print("Delta: ", self.delta)
            print("Clip Param C: ", self.C)

            self.accountant = RDPAccountant()
            self.sample_rate = self.configs.bs / self.training_size
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.configs.lr)

            # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.configs.lr, weight_decay=self.configs.wd, momentum=0.9)
    
            noise_param = get_noise_multiplier(target_epsilon = self.configs.epsilon,
                                                target_delta = 1e-5,
                                                sample_rate = self.sample_rate,
                                                epochs = self.configs.epochs,
                                                accountant="rdp")

            self.noise_scale = noise_param
            print("Noise Scale: ", self.noise_scale)

            self.privacy_engine = PrivacyEngine(self.model,
                                            batch_size=self.configs.bs,
                                            sample_size=self.training_size,
                                            noise_multiplier=self.noise_scale,
                                            epochs=self.configs.epochs,
                                            clipping_fn='automatic', # Abadi
                                            clipping_mode="MixOpt", # ghost
                                            origin_params=None,
                                            clipping_style="all-layer") # layer-wise


            self.privacy_engine.attach(self.optimizer)    

        else:

            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.configs.lr, momentum=0.9, weight_decay=self.configs.wd)

    def compute_noise_multiplier(self, epsilon, delta, num_samples, batch_size, num_epochs):

        noise_multiplier = opacus.accountants.utils.get_noise_multiplier(target_epsilon=epsilon,
                                                                        target_delta=delta,
                                                                        sample_rate=batch_size / num_samples,
                                                                        epochs=num_epochs,
                                                                        accountant="rdp")

        return noise_multiplier
    
    def zeroshot_classifier(self, template):
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(self.classes):
                texts = [t.format(classname) for t in template] # format with class
                texts = clip.tokenize(texts).to(self.device) # tokenize
                class_embeddings = self.model.encode_text(texts) # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(self.device)
        return zeroshot_weights
    
    def test(self, test_dataloader, template=None):
        correct_num = torch.tensor(0).to(self.device)


        if self.modality == "all":
            text_features = self.zeroshot_classifier([template])
            
            for image, label in tqdm(test_dataloader):
                with torch.no_grad():
                    features = self.model.encode_image(image.to(self.device))
                    features /= features.norm(dim=-1, keepdim=True)
                    
                    similarity = (100.0 * features @ text_features)
                    probs = similarity.softmax(dim=-1)

                    _, pred = torch.max(probs, 1)

                    label = torch.tensor([self.classes[name] for name in list(label)])
                    num = torch.sum(pred==label.to(self.device))

                    correct_num = correct_num + num

            print ('Accuracy Rate: {}'.format(correct_num/len(test_dataloader)/self.configs.bs))

        elif self.modality == "image":
            for images, labels in tqdm(test_dataloader):
                with torch.no_grad():
                    outputs = self.model(images.to(self.device))
            
                    _, pred = torch.max(outputs.data, 1)
                    num = torch.sum(pred==labels.to(self.device))

                    correct_num = correct_num + num

            print ('Accuracy Rate: {}'.format(correct_num/len(test_dataloader)/self.configs.bs))
        
    
    def train(self, train_dataloader, test_dataloader, template, loss_report_freq=1000):
        print(f"Num Epochs: {self.configs.epochs}")
        
        
        start_training_time=time.time()
        device = self.device
        self.model.train()

        starting_epoch = 10 if self.configs.continue_training else 0

        for epoch in range(starting_epoch, self.configs.epochs):
            tqdm_object = tqdm(train_dataloader, total=len(train_dataloader))
            batch_ct = 1
            for batch in tqdm_object:
                self.optimizer.zero_grad()
                
                images, labels = batch 
                images = images.to(device)
                labels = labels.to(device)
                if self.modality == "image":
                    outputs = self.model(images) # image encoder
                    total_loss = self.loss_img(outputs, labels)
                    total_loss.backward()
                    if (batch_ct % self.n_acc_steps == 0) or (batch_ct == len(train_dataloader)):
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    
                elif self.modality == "text":
                    print("TODO")
                elif self.modality == "all":
                
                    text_tokens = clip.tokenize([ template.format(desc) for desc in texts]).to(device)
                    logits_per_image, logits_per_text = self.model(images, text_tokens)
                    ground_truth = torch.arange(len(images),dtype=torch.long,device=device) # assigning labels to 
                    total_loss = (self.loss_img(logits_per_image,ground_truth) + self.loss_txt(logits_per_text,ground_truth))/2
                    total_loss.backward()
                    self.optimizer.step()

            
                batch_ct += 1

            if self.configs.private:
                my_dict  = self.privacy_engine.get_privacy_spent(accounting_mode="rdp")
                eps_rdp = my_dict["eps_rdp"]
                alpha_rdp = my_dict["alpha_rdp"]

                print(f"Privacy budget spent: (ε = {eps_rdp:.2f}, δ = {self.delta}) for α = {alpha_rdp}")
        
            if epoch % 2 == 0 :
                print(f"****the {epoch}^th epoch *****")
                print("**** on training set *****")
                self.test(train_dataloader, template)
                print("*************************")
                print("**** on testing set *****")
                self.test(test_dataloader, template)
                print("*************************")

                # save the model weights
                
                if self.configs.private:
                    save_dir = "saved_ours/dp"
                    checkpoint_name = f"{save_dir}/{self.configs.modality}_{self.configs.dataset}_dp_eps{self.configs.epsilon}_epochs{self.configs.epochs}_lr{self.configs.lr}_bs{self.configs.bs}_{self.configs.trainable}.pth"
                else:
                    save_dir = "saved_ours/nondp"
                    checkpoint_name = f"{save_dir}/{self.configs.modality}_{self.configs.dataset}_epochs{self.configs.epochs}_lr{self.configs.lr}_bs{self.configs.bs}_{self.configs.trainable}.pth"

                self.save_model(checkpoint_name)
                

            self.losses.append(total_loss.item())
        
        ending_training_time=time.time()
        print("Training Time: ", ending_training_time-start_training_time)
    

    def save_model(self, checkpoint_name):
        torch.save(
          {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': self.losses
          }, 
          checkpoint_name
        )


def train_wrapper(**kwargs):

    configs = SimpleNamespace(**kwargs)

    print()
    for key, value in kwargs.items():
        print(f"{key}: {value}")
    print()
    
    templates = [
        'a photo of a {}.',
        # 'a blurry photo of a {}.',
        # 'a bad photo of a {}.',
        # 'a good photo of a {}.'
    ]
    
    for template in templates:
        print(f"NEW TEMPLATE: {template}\n")
        
        private_model = PrivateModel(configs=configs, template=template)       
        train_dataloader, test_dataloader = private_model.load_data(configs.dataset)
        private_model.load_model(modality=configs.modality)
        private_model.set_optimizer(epsilon=configs.epsilon, C=1.0)

        print("**********")
        private_model.train(train_dataloader, test_dataloader, template)
        private_model.test(test_dataloader, template)
        print("------------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train wrapper for private model training.")

    # general training arguments
    parser.add_argument('--dataset', type=str, choices=["cifar10", "fashion_indio"])
    parser.add_argument('--bs', type=int, default=64, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--wd', type=float, default=0.0, help="Weight decay")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs")
    parser.add_argument('--trainable', type=str, choices=["last-layer", "all-layers"])
    parser.add_argument('--continue_training', action='store_true', help="load the model weights")

    # differential privacy arguments
    parser.add_argument('--epsilon', type=float, default=10, help="epsilon value for DP")
    parser.add_argument('--modality', type=str, default="image", choices=["image", "text", "all"])
    parser.add_argument('--private', action='store_true', help="Enable DP training")
    

    # extra params
    parser.add_argument('--config', type=str, default="configs/finetune_image_encoder.yaml", help="Path to the configuration file")

    args = parser.parse_args()

    # Load hyperparameters from config file if provided
    if args.config:
        config_hyperparams = load_config(args.config)

        # Update args with the config hyperparameters
        for key, value in config_hyperparams.items():
            if not hasattr(args, key):  # Add new hyperparams not already in args
                setattr(args, key, value)
            else:
                print(f"Overriding {key} from input arguments or defaults.")

    train_wrapper(**vars(args))