import torch
import torch.nn as nn
import numpy as np

class SANet(nn.Module):
    def __init__(self, input_size, hidden_size, n_classes, embeddings, n_blocks=1, dropout=0.1, n_heads=1, pos_enc_len=None):
        super(SANet, self).__init__()
        self.word_embedding = nn.Embedding(embeddings.shape[0], embeddings.shape[1], padding_idx=0, sparse=True)
        self.word_embedding.weight.data = torch.from_numpy(embeddings).float()

        self.use_pos_enc = pos_enc_len is not None
        if self.use_pos_enc:
            self.pos_encoding = nn.Embedding(pos_enc_len, embeddings.shape[1], padding_idx=0, sparse=True)
            self.pos_encoding.weight.data = self.pos_enc_init(pos_enc_len, embeddings.shape[1])
            self.pos_encoding.weight.requires_grad = False

        self.first_layer = nn.Linear(input_size, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size,
                                                   nhead=n_heads,
                                                   dim_feedforward=hidden_size,
                                                   dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_classes)
        )

    def forward(self, x, mask):
        enc = self.word_embedding(x)
        if self.use_pos_enc:
            x_pos = self.seq_pos(x)
            enc += self.pos_encoding(x_pos)
        x = self.first_layer(enc)
        x = self.encoder(x, src_key_padding_mask=mask)
        x, _ = torch.max(x, 0)
        return self.classifier(x)

    def pos_enc_init(self, n_pos, pos_dim):
        n_pos = n_pos + 1
        pos_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / pos_dim) for j in range(pos_dim)] if pos != 0 else np.zeros(pos_dim) for pos in range(n_pos)])

        pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])
        pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])
        return torch.from_numpy(pos_enc).float()

    def seq_pos(self, x):
        pos = np.repeat(np.arange(1, x.shape[0]+1).reshape(x.shape[0], 1), x.shape[1], axis=1)
        pos[np.where(x == 0)] = 0
        return x.new(pos).long()
