from torch import nn


class MFBias(nn.Module):
    def __init__(self, num_user, num_movie, emb_size):
        super(MFBias, self).__init__()
        self.user_emb = nn.Embedding(num_user, emb_size)
        self.movie_emb = nn.Embedding(num_movie, emb_size)
        self.user_bias = nn.Embedding(num_user, 1)
        self.movie_bias = nn.Embedding(num_movie, 1)

        self.user_emb.weight.data.uniform_(0, 0.05)
        self.movie_emb.weight.data.uniform_(0, 0.05)
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.movie_bias.weight.data.uniform_(-0.01, 0.01)

    def forward(self, u, v):
        U = self.user_emb(u)
        V = self.movie_emb(v)

        b_u = self.user_bias(u).squeeze()
        b_v = self.movie_bias(v).squeeze()

        return (U * V).sum(1) + b_u + b_v
