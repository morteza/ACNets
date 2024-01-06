import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_size, n_embeddings):
        super(Encoder, self).__init__()
        self.encode = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Linear(input_size //2, n_embeddings),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.encode(x)
        return out


class Decoder(nn.Module):
    def __init__(self, n_embeddings, n_outputs):
        super(Decoder, self).__init__()
        self.decode = nn.Sequential(
            nn.Linear(n_embeddings, n_outputs // 2),
            nn.ReLU(),
            nn.Linear(n_outputs // 2, n_outputs),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.decode(x)
        return out


class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_size, n_embeddings):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.encoder = Encoder(input_size, n_embeddings)
        self.decoder = Decoder(n_embeddings, input_size)

    def forward(self, x):

        h = self.encoder(x)
        x_recon = self.decode(h)

        return h, x_recon
