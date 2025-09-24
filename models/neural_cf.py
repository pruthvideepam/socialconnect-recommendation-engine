import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, hidden_layers=[128, 64, 32]):
        super(NeuralCF, self).__init__()
        
        # User and item embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Neural MF layers
        layers = []
        input_dim = embedding_dim * 2
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        self.mlp_layers = nn.Sequential(*layers)
        
    def forward(self, user_ids, item_ids):
        user_embed = self.user_embedding(user_ids)
        item_embed = self.item_embedding(item_ids)
        
        # Concatenate embeddings
        concat_embed = torch.cat([user_embed, item_embed], dim=1)
        
        # Pass through MLP
        output = self.mlp_layers(concat_embed)
        return torch.sigmoid(output)
