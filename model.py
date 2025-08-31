# model.py
import torch
import torch.nn as nn

# NCF model (NeuMF)
class NCF(nn.Module):
    def __init__(self, num_users, num_items, feedback_type, embed_mlp=32, embed_gmf=8, mlp_layers=[64, 32, 16, 8]):
        super(NCF, self).__init__()
        self.feedback_type = feedback_type
        self.user_embed_mlp = nn.Embedding(num_users, embed_mlp)
        self.item_embed_mlp = nn.Embedding(num_items, embed_mlp)
        self.user_embed_gmf = nn.Embedding(num_users, embed_gmf)
        self.item_embed_gmf = nn.Embedding(num_items, embed_gmf)

        # MLP tower
        mlp_modules = []
        input_size = 2 * embed_mlp
        for dim in mlp_layers:
            mlp_modules.append(nn.Linear(input_size, dim))
            mlp_modules.append(nn.ReLU())
            input_size = dim
        self.mlp = nn.Sequential(*mlp_modules)

        # Prediction layer
        self.predict = nn.Linear(embed_gmf + mlp_layers[-1], 1)

    def forward(self, user, item):
        u_mlp = self.user_embed_mlp(user)
        i_mlp = self.item_embed_mlp(item)
        mlp_in = torch.cat([u_mlp, i_mlp], dim=-1)
        mlp_out = self.mlp(mlp_in)

        u_gmf = self.user_embed_gmf(user)
        i_gmf = self.item_embed_gmf(item)
        gmf = u_gmf * i_gmf

        combined = torch.cat([gmf, mlp_out], dim=-1)
        out = self.predict(combined).squeeze()

        if self.feedback_type == 'implicit':
            out = torch.sigmoid(out)
        else:
            # For explicit, no sigmoid (linear), but clamp for stability 
            out = torch.clamp(out, 0.5, 5.5)
            #out = out
        
        return out

# SimpleMF for proxy/pretraining
class SimpleMF(nn.Module):
    def __init__(self, num_users, num_items, feedback_type, embed_size=32):
        super(SimpleMF, self).__init__()
        self.feedback_type = feedback_type
        self.user_embed = nn.Embedding(num_users, embed_size)
        self.item_embed = nn.Embedding(num_items, embed_size)

    def forward(self, user, item):
        out = (self.user_embed(user) * self.item_embed(item)).sum(1)
        if self.feedback_type == 'implicit':
            out = torch.sigmoid(out)
        else:
            # For explicit, no sigmoid, clamp for stability
            out = torch.clamp(out, 0.5, 5.5)
            #out = out
        return out