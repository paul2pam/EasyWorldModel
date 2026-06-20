import torch.nn as nn
import torch

from encoder import Encoder


# Legend: 
#   e = encoded state of the environment
#   h = recurrent state 
#   z = discrete representation of current state
#   a = action vector

# Representation model
class Posterior(nn.Module):
    def __init__(self, embed_dim, deter_dim):
        super().__init__()
        self.embed_dim = embed_dim  
        self.deter_dim = deter_dim
        self.total_dim = embed_dim + deter_dim
        self.hidden_dim = 512               #TODO: un-hardcode
        self.stoch_dim = 512                #TODO: un-hardcode

        self.mlp = nn.Sequential(
            nn.Linear(self.total_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.stoch_dim)
        )
        

    def forward(self, e, h):
        z = torch.cat((e, h), dim=-1)
        z = self.mlp(z)
        return z
    


# Dynamics predictor
class Prior(nn.Module):
    def __init__(self, deter_dim):
        super().__init__()
        self.deter_dim = deter_dim
        self.hidden_dim = 512          #TODO:un-hardcode
        self.stoch_dim = 512            #TODO:un-hardcode

        self.mlp = nn.Sequential(
            nn.Linear(self.deter_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.stoch_dim)
        )

    def forward(self, h):
        z = self.mlp(h)
        return z
        
class SequenceModel(nn.Module):     #Not true to paper, this uses a regular GRUcell, not the block diagonal version
    def __init__(self, deter_dim, discrete_dim, action_dim):
        super().__init__()
        self.deter_dim = deter_dim
        self.discrete_dim = discrete_dim
        self.action_dim = action_dim
        self.hidden_dim = 512

        self.linear = nn.Linear(self.discrete_dim + self.action_dim, self.hidden_dim)
        self.gru = nn.GRUCell(self.hidden_dim, self.deter_dim)

    def forward(self, h, z, a):
        input = torch.cat((z,a), dim=-1)
        x = self.linear(input)
        h_new = self.gru(x, h)
        return h_new

class RSSM(nn.Module):

    def __init__(self, embed_dim, hidden_dim, deter_dim, discrete_dim, action_dim):
        super().__init__()
        self.sequence        = SequenceModel(deter_dim, discrete_dim, action_dim)
        self.dynamics        = Prior(deter_dim)
        self.representation  = Posterior(embed_dim, deter_dim)
        self.encoder         = Encoder(embed_dim)

        self.deter_dim = deter_dim
        self.discrete_dim = discrete_dim
        self.action_dim = action_dim

    def initial(self, batch_size):
        device = next(self.parameters()).device
        h = torch.zeros(batch_size, self.deter_dim, device=device)
        z = torch.zeros(batch_size, self.discrete_dim, device=device)
        return h, z

    def observation_step(self, h, z, a, e):
        h_new = self.sequence(h, z, a)
        z_new = self.representation(e, h)
        return h_new, z_new

    def imagination_step(self, h, z, a):
        h_new = self.sequence(h, z, a)
        z_new = self.dynamics(h)
        return h_new, z_new