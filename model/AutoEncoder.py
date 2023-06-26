import torch.nn as nn 

class AutoEncoder(nn.Module):
    def __init__(self, timestep=26, hidden_dim=144):
        super().__init__()
        # [b, 784]
        self.encoder = nn.Sequential(
            nn.Linear(timestep*hidden_dim,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,512),
            nn.ReLU(inplace=True),
            nn.Linear(512,64),
            nn.ReLU(inplace=True)
        )   
        self.decoder = nn.Sequential(
            nn.Linear(64,512),
            nn.ReLU(inplace=True),
            nn.Linear(512,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,timestep*hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, x,mask):
        #BS,T,F
        _,T,F = x.size()
        x = x.flatten(1)
        x = self.encoder(x)
        x = self.decoder(x)
        x= x.view(-1,T,F)
        return x


if __name__ == '__main__':
    import torch
    model = AutoEncoder(26,144)
    x = torch.zeros([32,26,144])
    y = model(x)
    print(y.size())
