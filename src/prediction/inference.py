import torch
import torch.nn.functional as F
from model import CVAE
from beer_data_preprocessing import BeerDataProcessor

# load the trained model, processor, and metadata
def load_artifacts(model_path):
    ckpt = torch.load(model_path, map_location='cpu')
    processor = ckpt['processor']
    info = ckpt['data_info']
    model = CVAE(
        n_numeric         = info['n_numeric'],
        cat_cardinalities = info['cardinalities'],
        text_vocab_size   = info['vocab_size'],
        cond_dim          = info['cond_dim'],
        latent_dim        = info['latent_dim'],
        hidden_dim        = info['hidden_dim'],
        cat_emb_dim       = info.get('cat_emb_dim', 16),
        text_emb_dim      = info.get('text_emb_dim', 128),
        text_hid_dim      = info.get('text_hid_dim', 128),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, processor, info

# build a one‑hot condition vector for age_bucket
def build_condition(age_bucket: int, cond_dim: int) -> torch.Tensor:
    return F.one_hot(torch.tensor([age_bucket]), num_classes=cond_dim).float()

# sample many latent codes, decode them, predict scores, and return the best
def sample_and_rank(model, c, N=1000):
    z = torch.randn(N, model.latent_dim)
    c_exp = c.expand(N, -1)
    x_cands = model.decode(z, c_exp)
    scores = model.predict_score(z, c_exp)
    best_i = scores.argmax()
    return x_cands[best_i:best_i+1], scores[best_i].item()

# split the reconstructed vector into numeric, categorical‑embedding, and text‑embedding parts
def slice_reconstructed(x_recon: torch.Tensor, info: dict):
    n_num = info['n_numeric']
    n_cat_feats = len(info['cardinalities'])
    cat_emb_dim = info.get('cat_emb_dim', 16)
    d_cat_total = n_cat_feats * cat_emb_dim
    d_text = info.get('text_hid_dim', 128)
    vec = x_recon.squeeze(0)
    num_vec = vec[:n_num]
    cat_vec = vec[n_num : n_num + d_cat_total]
    text_vec = vec[n_num + d_cat_total : n_num + d_cat_total + d_text]
    return num_vec, cat_vec, text_vec

# decode the slices back to human‑readable features
def decode_reconstructed(num_vec, cat_vec, text_vec, processor, info, model):
    # invert scaling on numeric features (detach before converting)
    num_orig = processor.scaler.inverse_transform(
        num_vec.detach().cpu().numpy().reshape(1, -1)
    )

    # decode each categorical by nearest embedding
    decoded_cats = {}
    offset = 0
    for col_name, emb, card in zip(processor.categorical_cols, model.cat_embs, info['cardinalities']):
        emb_dim = emb.embedding_dim
        slice_i = cat_vec[offset : offset + emb_dim]
        weights = emb.weight.data
        sims = F.cosine_similarity(
            slice_i.unsqueeze(0).repeat(card, 1),
            weights,
            dim=1
        )
        idx = sims.argmax().item()
        decoded_cats[col_name] = processor.label_encoders[col_name].classes_[idx]
        offset += emb_dim

    # include raw text embedding for downstream nearest‑neighbor if desired
    decoded_cats['text_embedding'] = text_vec.detach().cpu().numpy()

    return num_orig.flatten(), decoded_cats


if __name__ == '__main__':
    # load artifacts
    model, processor, info = load_artifacts('beer_cvae_model.pth')
    # build a condition vector for age‑bucket 0 (e.g. 18–25)
    c_star = build_condition(age_bucket=0, cond_dim=info['cond_dim'])
    # generate and score candidates, pick the best
    x_best, best_score = sample_and_rank(model, c_star, N=2000)
    print(f"Best predicted score = {best_score:.3f}")
    # slice the best vector back into parts
    num_vec, cat_vec, text_vec = slice_reconstructed(x_best, info)
    # decode to original features
    numeric_vals, categorical_vals = decode_reconstructed(
        num_vec, cat_vec, text_vec, processor, info, model
    )
    # print numeric features with original column names
    print("Numeric features (original scale):")
    for name, val in zip(processor.numeric_cols, numeric_vals):
        print(f"  {name}: {val:.3f}")
    # print decoded categorical choices
    print("Categorical predictions:")
    for col, choice in categorical_vals.items():
        print(f"  {col}: {choice}")
