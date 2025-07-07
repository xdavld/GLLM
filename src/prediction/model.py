import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

# load a global, frozen SBERT encoder
SENT_ENCODER = SentenceTransformer('all-mpnet-base-v2', device='cuda' if torch.cuda.is_available() else 'cpu')

class CVAE(nn.Module):
    def __init__(
        self,
        n_numeric: int,
        cat_cardinalities: list,
        cond_dim: int,
        n_classes: int,
        latent_dim: int = 16,
        hidden_dim: int = 128,
        cat_emb_dim: int = 16,
        text_emb_dim: int = 768,     # SBERT output dim
        text_proj_dim: int = 128     # your internal text feature size
    ):
        super().__init__()
        # categorical embeddings
        self.cat_embs = nn.ModuleList([
            nn.Embedding(card, cat_emb_dim)
            for card in cat_cardinalities
        ])
        self.d_cat = len(cat_cardinalities) * cat_emb_dim

        # SBERT→projection
        self.text_proj = nn.Linear(text_emb_dim, text_proj_dim)
        self.d_text   = text_proj_dim

        # dimensions
        self.input_dim  = n_numeric + self.d_cat + self.d_text
        self.cond_dim   = cond_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # encoder
        self.enc_fc1     = nn.Linear(self.input_dim + cond_dim, hidden_dim)
        self.enc_fc_mu     = nn.Linear(hidden_dim, latent_dim)
        self.enc_fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # decoder
        self.dec_fc1 = nn.Linear(latent_dim + cond_dim, hidden_dim)
        self.dec_fc2 = nn.Linear(hidden_dim, self.input_dim)

        # recommendation regression head
        self.reg_fc1 = nn.Linear(latent_dim + cond_dim, hidden_dim)
        self.reg_fc2 = nn.Linear(hidden_dim, 1)

        # satisfaction classification head
        self.cls_fc1 = nn.Linear(latent_dim + cond_dim, hidden_dim)
        self.cls_fc2 = nn.Linear(hidden_dim, n_classes)

    def encode(
        self,
        x_num: torch.Tensor,
        x_cat_idxs: torch.Tensor,
        x_text_emb: torch.Tensor,
        c: torch.Tensor
    ):
        # categorical
        cat_vecs = [emb(x_cat_idxs[:, i]) for i, emb in enumerate(self.cat_embs)]
        x_cat = torch.cat(cat_vecs, dim=-1)

        # text: project SBERT embeddings
        x_txt = F.relu(self.text_proj(x_text_emb))

        # full concatenation
        x_full = torch.cat([x_num, x_cat, x_txt], dim=-1)

        # condition
        xc = torch.cat([x_full, c], dim=-1)
        h  = F.relu(self.enc_fc1(xc))
        mu     = self.enc_fc_mu(h)
        logvar = self.enc_fc_logvar(h)
        return mu, logvar, x_full

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        zc = torch.cat([z, c], dim=-1)
        h  = F.relu(self.dec_fc1(zc))
        return self.dec_fc2(h)

    def predict_score(self, z, c):
        zc = torch.cat([z, c], dim=-1)
        h  = F.relu(self.reg_fc1(zc))
        return self.reg_fc2(h).squeeze(-1)

    def classify(self, z, c):
        zc = torch.cat([z, c], dim=-1)
        h  = F.relu(self.cls_fc1(zc))
        return self.cls_fc2(h)

    def forward(self, x_num, x_cat_idxs, x_text_emb, c):
        mu, logvar, x_full = self.encode(x_num, x_cat_idxs, x_text_emb, c)
        z       = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, c)
        rec     = self.predict_score(z, c)
        logits  = self.classify(z, c)
        return x_recon, x_full, mu, logvar, rec, logits

def cvae_loss(x_recon, x_full, rec_pred, rec_true, logits, cls_true, mu, logvar,
              beta=1.0, gamma=0.5):
    recon_loss = F.mse_loss(x_recon, x_full, reduction='mean')
    kld        = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    reg_loss   = F.mse_loss(rec_pred, rec_true, reduction='mean')
    cls_loss   = F.cross_entropy(logits, cls_true.long())
    total      = recon_loss + beta*kld + gamma*reg_loss + gamma*cls_loss
    return total, recon_loss, kld, reg_loss, cls_loss
