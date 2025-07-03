import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# Text Encoder for free-text fields
# ----------------------------------------------------------------------------
class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 128, hid_dim: int = 128):
        super(TextEncoder, self).__init__()
        self.word_emb = nn.Embedding(vocab_size, emb_dim)
        self.rnn      = nn.GRU(emb_dim, hid_dim, batch_first=True)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: [batch, seq_len]
        emb, _ = self.rnn(self.word_emb(token_ids))  # [batch, seq_len, hid_dim]
        return emb[:, -1, :]  # [batch, hid_dim]


# ----------------------------------------------------------------------------
# Conditional VAE for mixed beer survey data
# ----------------------------------------------------------------------------
class CVAE(nn.Module):
    def __init__(
        self,
        n_numeric: int,
        cat_cardinalities: list,
        text_vocab_size: int,
        cond_dim: int,
        latent_dim: int = 16,
        hidden_dim: int = 128,
        cat_emb_dim: int = 16,
        text_emb_dim: int = 128,
        text_hid_dim: int = 128
    ):
        """
        n_numeric: number of continuous numeric features
        cat_cardinalities: list of ints for each categorical feature
        text_vocab_size: size of text vocabulary for token embeddings
        cond_dim: dimension of condition vector (e.g. age bucket one-hot)
        """
        super(CVAE, self).__init__()

        # -- Embedding layers --
        self.cat_embs = nn.ModuleList([
            nn.Embedding(card, cat_emb_dim)
            for card in cat_cardinalities
        ])
        self.d_cat = len(cat_cardinalities) * cat_emb_dim

        self.text_enc = TextEncoder(text_vocab_size, emb_dim=text_emb_dim, hid_dim=text_hid_dim)
        self.d_text = text_hid_dim

        # Combined feature dimension
        self.input_dim = n_numeric + self.d_cat + self.d_text
        self.cond_dim  = cond_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # -- Encoder --
        self.enc_fc1   = nn.Linear(self.input_dim + cond_dim, hidden_dim)
        self.enc_fc_mu    = nn.Linear(hidden_dim, latent_dim)
        self.enc_fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # -- Decoder --
        self.dec_fc1 = nn.Linear(latent_dim + cond_dim, hidden_dim)
        self.dec_fc2 = nn.Linear(hidden_dim, self.input_dim)

        # -- Regression head (score predictor) --
        self.reg_fc1 = nn.Linear(latent_dim + cond_dim, hidden_dim)
        self.reg_fc2 = nn.Linear(hidden_dim, 1)

    def encode(
        self,
        x_num: torch.Tensor,
        x_cat_idxs: torch.Tensor,
        x_text_ids: torch.Tensor,
        c: torch.Tensor
    ) -> tuple:
        # Embed categoricals
        cat_vecs = [emb(x_cat_idxs[:, i]) for i, emb in enumerate(self.cat_embs)]
        x_cat = torch.cat(cat_vecs, dim=-1)  # [batch, d_cat]

        # Encode text
        x_txt = self.text_enc(x_text_ids)   # [batch, d_text]

        # Concatenate all features
        x_full = torch.cat([x_num, x_cat, x_txt], dim=-1)  # [batch, input_dim]

        # Include condition
        xc = torch.cat([x_full, c], dim=-1)
        h  = F.relu(self.enc_fc1(xc))
        mu = self.enc_fc_mu(h)
        logvar = self.enc_fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(
        self,
        z: torch.Tensor,
        c: torch.Tensor
    ) -> torch.Tensor:
        zc = torch.cat([z, c], dim=-1)
        h  = F.relu(self.dec_fc1(zc))
        x_recon = self.dec_fc2(h)  # [batch, input_dim]
        return x_recon

    def predict_score(
        self,
        z: torch.Tensor,
        c: torch.Tensor
    ) -> torch.Tensor:
        zc = torch.cat([z, c], dim=-1)
        h  = F.relu(self.reg_fc1(zc))
        score = self.reg_fc2(h)      # [batch, 1]
        return score.squeeze(-1)

    def forward(
        self,
        x_num: torch.Tensor,
        x_cat_idxs: torch.Tensor,
        x_text_ids: torch.Tensor,
        c: torch.Tensor
    ) -> tuple:
        mu, logvar    = self.encode(x_num, x_cat_idxs, x_text_ids, c)
        z             = self.reparameterize(mu, logvar)
        x_recon       = self.decode(z, c)
        score_pred    = self.predict_score(z, c)
        return x_recon, mu, logvar, score_pred


# ----------------------------------------------------------------------------
# Loss function
# ----------------------------------------------------------------------------
def cvae_loss(
    x_recon: torch.Tensor,
    x_full: torch.Tensor,
    score_pred: torch.Tensor,
    score_true: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    gamma: float = 1.0
) -> tuple:
    """
    x_recon: [batch, input_dim]
    x_full:  [batch, input_dim]  # concatenated true features
    score_pred: [batch]
    score_true: [batch]
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(x_recon, x_full, reduction='mean')
    # KL divergence
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    # Regression loss (MSE)
    reg_loss = F.mse_loss(score_pred, score_true, reduction='mean')

    total = recon_loss + beta * kld + gamma * reg_loss
    return total, recon_loss, kld, reg_loss
