# model_training.py

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from torch.optim import Adam

from model import CVAE, cvae_loss
from data_preprocessing import DataProcessor

# SBERT encoder (CPU/GPU as available)
SBERT = SentenceTransformer(
    'all-mpnet-base-v2',
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

class BeerDataset(Dataset):
    def __init__(self, x_numeric, x_categorical, x_text_emb,
                 conditions, rec_scores, class_labels):
        self.x_numeric     = x_numeric
        self.x_categorical = x_categorical
        self.x_text_emb    = x_text_emb
        self.conditions    = conditions
        self.rec_scores    = rec_scores
        self.class_labels  = class_labels

    def __len__(self):
        return len(self.rec_scores)

    def __getitem__(self, idx):
        return {
            'x_numeric':     self.x_numeric[idx],
            'x_categorical': self.x_categorical[idx],
            'x_text_emb':    self.x_text_emb[idx],
            'conditions':    self.conditions[idx],
            'rec_score':     self.rec_scores[idx],
            'class_label':   self.class_labels[idx],
        }

def train_model(csv_path, num_epochs=100, batch_size=32, lr=1e-3):
    # Load and clean data
    processor = DataProcessor()
    df = processor.load_and_clean_data(csv_path)

    # Prepare features and targets
    data = processor.prepare_training_data(csv_path)
    x_numeric     = data['x_numeric']                               # [N, num_feats]
    x_categorical = torch.tensor(data['x_categorical'], dtype=torch.long)  # [N, num_cats]
    conditions    = data['conditions']                              # [N, cond_dim]

    # Classification labels: overall_satisfaction (mapped to 1-5)
    class_labels  = torch.tensor(data['targets'], dtype=torch.long)
    # Zero-base to [0..n_classes-1]
    class_labels -= class_labels.min()
    n_classes     = int(class_labels.max().item() + 1)

    # Regression target: normalized recommendation_score from numeric features
    rec_idx     = data['numeric_cols'].index('recommendation_score')
    rec_scores  = x_numeric[:, rec_idx]

    # Precompute SBERT embeddings for all text fields
    text_cols = [
        'taste_aromas',
        'prominent_taste',
        'liked_aromas',
        'disliked_aromas',
        'improvement_suggestions'
    ]
    texts      = df[text_cols].fillna('').agg(' '.join, axis=1).tolist()
    x_text_emb = SBERT.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_tensor=True
    )  # Tensor [N, 768]

    # Split into train/test
    N          = x_numeric.size(0)
    idxs       = torch.randperm(N)
    train_idx  = idxs[:int(0.8 * N)]
    test_idx   = idxs[int(0.8 * N):]

    # Create datasets and loaders
    train_ds = BeerDataset(
        x_numeric[train_idx],
        x_categorical[train_idx],
        x_text_emb[train_idx],
        conditions[train_idx],
        rec_scores[train_idx],
        class_labels[train_idx]
    )
    test_ds = BeerDataset(
        x_numeric[test_idx],
        x_categorical[test_idx],
        x_text_emb[test_idx],
        conditions[test_idx],
        rec_scores[test_idx],
        class_labels[test_idx]
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    # Instantiate CVAE with SBERT text projection
    model = CVAE(
        n_numeric         = x_numeric.size(1),
        cat_cardinalities = data['cardinalities'],
        cond_dim          = conditions.size(1),
        n_classes         = n_classes,
        latent_dim        = 16,
        hidden_dim        = 128,
        cat_emb_dim       = 8,
        text_emb_dim      = 768,
        text_proj_dim     = 128
    )

    optimizer = Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(1, num_epochs+1):
        model.train()
        Ls = Rs = Ks = RG = CL = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            x_num   = batch['x_numeric']
            x_cat   = batch['x_categorical']
            x_txt   = batch['x_text_emb']
            c       = batch['conditions']
            rec_t   = batch['rec_score']
            cls_t   = batch['class_label']

            x_recon, x_full, mu, logvar, rec_pred, logits = model(
                x_num, x_cat, x_txt, c
            )
            loss, recon, kld, reg, cls_loss = cvae_loss(
                x_recon, x_full, rec_pred, rec_t,
                logits, cls_t, mu, logvar,
                beta=0.1, gamma=0.5
            )
            loss.backward()
            optimizer.step()

            Ls  += loss.item()
            Rs  += recon.item()
            Ks  += kld.item()
            RG  += reg.item()
            CL  += cls_loss.item()

        # Log every epoch
        print(f"Epoch {epoch}/{num_epochs} | "
              f"Loss: {Ls/len(train_loader):.4f} "
              f"(Recon {Rs/len(train_loader):.4f}, "
              f"KLD {Ks/len(train_loader):.4f}, "
              f"Reg {RG/len(train_loader):.4f}, "
              f"Cls {CL/len(train_loader):.4f})")

    # Save model and preprocessing info
    torch.save({
        'model_state_dict': model.state_dict(),
        'processor':        processor,
        'data_info': {
            'cardinalities':  data['cardinalities'],
            'n_numeric':      x_numeric.size(1),
            'cond_dim':       conditions.size(1),
            'n_classes':      n_classes,
            'latent_dim':     model.latent_dim,
            'hidden_dim':     model.hidden_dim,
            'cat_emb_dim':    model.cat_embs[0].embedding_dim,
            'text_emb_dim':   768,
            'text_proj_dim':  128
        }
    }, 'cvae_model.pth')

    return model, processor, data

if __name__ == '__main__':
    train_model('your_beer_survey.csv', num_epochs=100, batch_size=32, lr=1e-3)
