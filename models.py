import torch
import torch.nn as nn
import timm
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer


class CLIPModel(nn.Module):
    def __init__(self,temperature, image_embedding, text_embedding,):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()


class ImageClassifier(nn.Module):
    def __init__(self, configs, num_classes):
        super().__init__()
        self.image_encoder = ImageEncoder(num_classes=num_classes, model_name=configs.model_name, pretrained=configs.pretrained, trainable=configs.trainable)
        self.projection_head = ProjectionHead(embedding_dim=configs.image_embedding, projection_dim=configs.projection_dim, dropout=configs.dropout)
        self.classifer = nn.Linear(configs.projection_dim, num_classes)
        
    def forward(self, x):
        return self.classifer(self.projection_head(self.image_encoder(x)))

class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, num_classes, model_name, pretrained, trainable):
        super().__init__()
        
        self.model = timm.create_model(model_name, pretrained, num_classes=num_classes, global_pool="avg")
        for name, p in self.model.named_parameters():
            p.requires_grad = False
            
            if trainable == "all-layers":
                p.requires_grad = True
            elif trainable == "last-layer" and "fc" in name:
                p.requires_grad = True # making the last layer trainable

    def forward(self, x):
        return self.model(x)
    

    
class TextEncoder(nn.Module):
    def __init__(self, model_name, pretrained, trainable):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())

        for name, p in self.model.named_parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim,
        dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim).cuda()
        self.gelu = nn.GELU().cuda()
        self.fc = nn.Linear(projection_dim, projection_dim).cuda()
        self.dropout = nn.Dropout(dropout).cuda()
        self.layer_norm = nn.LayerNorm(projection_dim).cuda()

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

