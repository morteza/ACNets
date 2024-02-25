import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_size, n_embeddings):
        super(Encoder, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=n_embeddings,
                           batch_first=True)

    def forward(self, x):
        _, (h, c) = self.rnn(x)
        return h, c


class Decoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(Decoder, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=output_size,
                           batch_first=True)

    def forward(self, x, h, c):
        out, _ = self.rnn(x, (h, c))
        return out


class Seq2SeqAE(nn.Module):
    def __init__(self, input_size, n_embeddings, **kwargs):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.encoder = Encoder(input_size, n_embeddings)
        self.h_fc = nn.Linear(n_embeddings, input_size)
        self.c_fc = nn.Linear(n_embeddings, input_size)
        self.decoder = Decoder(n_embeddings, input_size)

    def forward(self, x):
        """

        Args:
            x (torch.Tensor): of shape (batch_size, n_timepoints, input_size)

        Returns:
            h, x_recon, loss
        """

        batch_size, n_timepoints, _ = x.shape

        # encode
        h, c = self.encoder(x)

        # decoder input
        x_dec = torch.zeros(batch_size, n_timepoints, self.n_embeddings).to(h.device)

        # decoder hidden states
        h = h.view(batch_size, -1)
        c = c.view(batch_size, -1)
        h_dec = self.h_fc(h).unsqueeze(0)
        c_dec = self.c_fc(c).unsqueeze(0)

        # decode
        x_recon = self.decoder(x_dec, h_dec, c_dec)

        loss = F.mse_loss(x_recon, x)

        return h, x_recon, loss
