import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter


class Layer(nn.Module):
    def __init__(self, d_in, d_out, dropout=0.2):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.LayerNorm(d_out),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.layer(x)


class Encoder(nn.Module):
    def __init__(self, d_v, hidden_states, d_emb, n_layer, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([Layer(d_v, hidden_states[0])] +
                                    [Layer(hidden_states[_], hidden_states[_ + 1]) for _ in range(n_layer - 1)])
        self.mu = nn.Sequential(nn.Linear(hidden_states[-1], d_emb), nn.Dropout(dropout))
        self.logvar = nn.Sequential(nn.Linear(hidden_states[-1], d_emb), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        mu = self.mu(x)
        logvar = self.logvar(x)  # 最后应用 Dropout
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, d_v, hidden_states, d_emb, n_layer, dropout=0.1):
        super().__init__()
        self.first = nn.Linear(d_emb, hidden_states[-1])
        self.mid = nn.ModuleList(
            [Layer(hidden_states[n_layer - 1 - _], hidden_states[n_layer - 2 - _]) for _ in range(n_layer - 1)])
        self.last = nn.Linear(hidden_states[0], d_v)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.first(x)
        for layer in self.mid:
            x = layer(x)
        x = self.dropout(x)
        x = self.last(x)
        return x


class Classifier(nn.Module):
    def __init__(self, d_emb, n_cls, dropout=0.2):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(d_emb, d_emb),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_emb, n_cls),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.layer(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, r_list, d_emb, n_enc_layer, n_dec_layer, dropout=0.1):
        super().__init__()
        n_view = len(r_list)
        enc_hidden_states = []
        dec_hidden_states = []

        for _ in range(n_view):
            temp_hidden_states = []
            temp_hidden_states_ = []

            for i in range(n_enc_layer):
                hd = round(d_emb * 2)
                hd = int(hd)
                temp_hidden_states.append(hd)
            for i in range(n_dec_layer):
                hd = round(d_emb * 2)
                hd = int(hd)
                temp_hidden_states_.append(hd)

            enc_hidden_states.append(temp_hidden_states)
            dec_hidden_states.append(temp_hidden_states_)

        self.encoder_list = nn.ModuleList(
            [Encoder(r_list[v], enc_hidden_states[v], d_emb, n_enc_layer, dropout) for v in range(n_view)])
        self.decoder_list = nn.ModuleList(
            [Decoder(r_list[v], dec_hidden_states[v], d_emb, n_dec_layer, dropout) for v in range(n_view)])

        self.n_view = n_view
        self.r_list = r_list
        self.d_emb = d_emb

    def forward(self, v_list, mask):
        mid_states = []
        for enc_i, enc in enumerate(self.encoder_list):
            mid_states.append(enc(v_list[enc_i]).unsqueeze(1))
        emb = torch.cat(mid_states, dim=1)
        rec_r = []
        for dec_i, dec in enumerate(self.decoder_list):
            rec_r.append(dec(emb))
        return emb, rec_r


class Model(nn.Module):
    def __init__(self, d_list, d_emb, n_enc_layer, n_dec_layer, n_cls, theta, dropout=0.1):
        super().__init__()
        self.ae = AutoEncoder(d_list, d_emb, n_enc_layer, n_dec_layer, dropout)
        self.classifier = Classifier(d_emb, n_cls)
        self.weights = nn.Parameter(torch.softmax(torch.zeros([1, len(d_list), 1]), dim=1))
        self.d_emb = d_emb

    def forward(self, emb, mask_v):
        embs, rec_v = self.ae(emb, mask_v)
        weight = torch.pow(self.weights.expand(embs.shape[0], -1, -1), 1)
        weight = torch.softmax(weight.masked_fill(mask_v.unsqueeze(2) == 0, -1e9), dim=1)
        emb_fus = torch.sum(embs * weight, dim=1)
        pred = self.classifier(emb_fus)
        return pred, rec_v, embs


class MVAE(nn.Module):
    def __init__(self, d_list, d_emb, n_enc_layer, n_dec_layer, n_cls, theta, dropout=0.1):
        super(MVAE, self).__init__()
        n_view = len(d_list)
        enc_hidden_states = []
        dec_hidden_states = []

        for _ in range(n_view):
            temp_hidden_states = []
            temp_hidden_states_ = []

            for i in range(n_enc_layer):
                hd = round(d_emb * 2)
                hd = int(hd)
                temp_hidden_states.append(hd)
            for i in range(n_dec_layer):
                hd = round(d_emb * 2)
                hd = int(hd)
                temp_hidden_states_.append(hd)

            enc_hidden_states.append(temp_hidden_states)
            dec_hidden_states.append(temp_hidden_states_)
        self.encoders = nn.ModuleList([Encoder(d_list[v], enc_hidden_states[v], d_emb, n_enc_layer, dropout) for v in range(n_view)])
        self.decoders = nn.ModuleList([Decoder(d_list[v], dec_hidden_states[v], d_emb, n_dec_layer, dropout) for v in range(n_view)])
        self.benn = BetaENNLayer(d_emb, n_cls)
        self.experts = MixtureOfExperts(n_view, False)
        self.d_emb = d_emb

    def infer(self, data_v, mask_v):
        batch_size = data_v[0].size(0)
        mu, logvar = [], []
        for enc_i, enc in enumerate(self.encoders):
            mu_i, logvar_i = enc(data_v[enc_i])
            mu.append(mu_i.unsqueeze(1))
            logvar.append(logvar_i.unsqueeze(1))
        mu = torch.cat(mu, dim=1)
        logvar = torch.cat(logvar, dim=1)

        return mu, logvar

    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, data_v, mask_v):
        mu, logvar = self.infer(data_v, mask_v)
        data_rec = []
        zs = self.reparametrize(mu, logvar)
        for dec_i, dec in enumerate(self.decoders):
            data_rec.append(dec(zs))
        z = self.reparametrize(*self.experts(mu, logvar, mask_v))
        pred = self.benn(z)
        
        return pred, data_rec, mu, logvar, zs


class MixtureOfExperts(nn.Module):
    def __init__(self, num_experts, avg=True):
        super().__init__()
        self.weights = nn.Parameter(torch.softmax(torch.zeros([1, num_experts, 1]), dim=1))  # 门控网络
        self.num_experts = num_experts
        self.avg = avg

    def forward(self, mu, logvar, mask_v, eps=1e-8):
        batch_size, M, D = mu.shape
        if self.avg:
            alpha = torch.softmax(torch.ones(mu.shape).to(mu.device).masked_fill(mask_v.unsqueeze(2) == 0, -1e9), dim=1)
        else:
            weights = torch.pow(self.weights.expand(mu.shape[0], -1, -1), 1)
            alpha = torch.softmax(weights.masked_fill(mask_v.unsqueeze(2) == 0, -1e9), dim=1)

        var = torch.exp(logvar) + eps
        weighted_mu = torch.sum(alpha * mu, dim=1)  # [B, D]
        weighted_var = torch.sum(alpha * var, dim=1)  # [B, D]
        weighted_logvar = torch.log(weighted_var + eps)

        return weighted_mu, weighted_logvar


class BetaENNLayer(nn.Module):
    def __init__(self, input_dim, n_cls, dropout=0.2):
        super().__init__()
        self.fc_alpha = nn.Sequential(
            nn.Linear(input_dim, round(input_dim)),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(round(input_dim), n_cls),
        )
        self.fc_beta = nn.Sequential(
            nn.Linear(input_dim, round(input_dim)),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(round(input_dim), n_cls),
        )
        self.n_cls = n_cls
        self.W = 2.0
        self.register_buffer('BASE_RATE', torch.zeros(n_cls))
        
    def update_base_rates(self, label_counts, num_counts):
        if torch.sum(label_counts) > 0:
            base_rates = label_counts / num_counts
            self.BASE_RATE.copy_(base_rates)
        
    def forward(self, features):
        alpha_logits = self.fc_alpha(features)
        beta_logits = self.fc_beta(features)

        alpha = F.softplus(alpha_logits) + 1
        beta = F.softplus(beta_logits) + 1

        S = alpha + beta
        belief = (alpha) / (S + self.W)
        disbelief = (beta) / (S + self.W)
        uncertainty = self.W / (S + self.W)

        probability = alpha / (S)

        return {
            "alpha": alpha,
            "beta": beta,
            "belief": belief,
            "disbelief": disbelief,
            "uncertainty": uncertainty,
            "probability": probability,
        }


def prior_expert(size, device="cuda"):
    mu = Variable(torch.zeros(size)).to(device)
    logvar = Variable(torch.zeros(size)).to(device)
    return mu, logvar


class Swish(nn.Module):
    """https://arxiv.org/abs/1710.05941"""
    def forward(self, x):
        return x * F.sigmoid(x)