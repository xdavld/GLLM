import torch
import torch.nn.functional as F
from torch.optim import Adam
from model import CVAE
from data_preprocessing import DataProcessor

# load the trained model, processor, and metadata
def load_artifacts(model_path):
    ckpt      = torch.load(model_path, map_location='cpu')
    processor = ckpt['processor']
    info      = ckpt['data_info']

    model = CVAE(
        n_numeric         = info['n_numeric'],
        cat_cardinalities = info['cardinalities'],
        cond_dim          = info['cond_dim'],
        n_classes         = info['n_classes'],
        latent_dim        = info['latent_dim'],
        hidden_dim        = info['hidden_dim'],
        cat_emb_dim       = info.get('cat_emb_dim', 16),
        text_emb_dim      = info.get('text_emb_dim', 768),
        text_proj_dim     = info.get('text_proj_dim', 128)
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, processor, info

# build a one-hot condition vector for age_bucket
def build_condition(age_bucket: int, cond_dim: int) -> torch.Tensor:
    return F.one_hot(torch.tensor([age_bucket]), num_classes=cond_dim).float()

# hybrid search: random-restart + gradient ascent with range penalty
def hybrid_optimize(model, c, info, M=50, K=5, steps=50, lr=1e-2, pen=100.0):
    # 1) sample M random latent codes and score
    Z = torch.randn(M, model.latent_dim)
    C = c.expand(M, -1)
    rec_scores = model.predict_score(Z, C)
    cls_logits = model.classify(Z, C)
    cls_conf   = F.softmax(cls_logits, dim=1)[:, info['n_classes']-1]
    scores     = rec_scores + cls_conf
    topk_idx   = scores.topk(K).indices

    best_score = -float('inf')
    best_z     = None
    # 2) refine each top seed via constrained gradient ascent
    for idx in topk_idx:
        z = Z[idx:idx+1].clone().detach().requires_grad_(True)
        opt_z = Adam([z], lr=lr)
        for _ in range(steps):
            opt_z.zero_grad()
            # heads
            rec = model.predict_score(z, c)                # [1]
            logits = model.classify(z, c)                  # [1, n_classes]
            prob = F.softmax(logits, dim=1)[0, info['n_classes']-1]
            # decode normalized features
            x_norm   = model.decode(z, c)[0]
            num_norm = x_norm[: info['n_numeric']]
            # penalty for any feature outside [0,1]
            viol = F.relu(num_norm - 1.0) + F.relu(-num_norm)
            penalty = pen * viol.sum()
            # combined objective with penalty
            loss = -(rec + prob) + penalty
            loss.backward()
            opt_z.step()
        # evaluate combined score
        with torch.no_grad():
            comb = (model.predict_score(z, c) + 
                    F.softmax(model.classify(z, c), dim=1)[0, info['n_classes']-1])
            val = comb.item()
            if val > best_score:
                best_score = val
                best_z     = z.detach().clone()
    # decode best latent code
    with torch.no_grad():
        x_best   = model.decode(best_z, c)
        rec_best = model.predict_score(best_z, c).item()
        prob_best= F.softmax(model.classify(best_z, c), dim=1)[0, info['n_classes']-1].item()
    return x_best, rec_best, prob_best

# split reconstructed vector into parts
def slice_reconstructed(x_recon: torch.Tensor, info: dict):
    n_num = info['n_numeric']
    n_cat = len(info['cardinalities'])
    emb_c = info.get('cat_emb_dim', 16)
    total_cat = n_cat * emb_c
    d_text = info.get('text_hid_dim', 128)
    vec = x_recon.squeeze(0)
    num_vec = vec[:n_num]
    cat_vec = vec[n_num:n_num+total_cat]
    text_vec= vec[n_num+total_cat:n_num+total_cat+d_text]
    return num_vec, cat_vec, text_vec

# decode the slices back to human-readable features
def decode_reconstructed(num_vec, cat_vec, text_vec, processor, info, model):
    # clamp all normalized numeric features to [0,1]
    num_vec = num_vec.clamp(0.0, 1.0)
    # invert MinMax scaling
    num_orig = processor.scaler.inverse_transform(
        num_vec.detach().cpu().numpy().reshape(1, -1)
    )
    # decode categoricals
    decoded = {}
    offset = 0
    for name, emb, card in zip(processor.categorical_cols, model.cat_embs, info['cardinalities']):
        dim = emb.embedding_dim
        slice_i = cat_vec[offset:offset+dim]
        sims = F.cosine_similarity(
            slice_i.unsqueeze(0).repeat(card,1), emb.weight.data, dim=1
        )
        idx = sims.argmax().item()
        decoded[name] = processor.label_encoders[name].classes_[idx]
        offset += dim
    decoded['text_emb'] = text_vec.detach().cpu().numpy()
    return num_orig.flatten(), decoded

if __name__ == '__main__':
    model, processor, info = load_artifacts('cvae_model.pth')
    c_star = build_condition(age_bucket=0, cond_dim=info['cond_dim'])

    # run hybrid constrained optimization
    x_best, rec, sat = hybrid_optimize(
        model, c_star, info,
        M=50, K=5, steps=50, lr=1e-2, pen=100.0
    )
    print(f"Final score = {rec + sat:.3f} (rec={rec:.3f}, sat={sat:.3f})")

    num_vec, cat_vec, text_vec = slice_reconstructed(x_best, info)
    numeric_vals, categorical_vals = decode_reconstructed(
        num_vec, cat_vec, text_vec, processor, info, model
    )

    print("Numeric features (orig scale):")
    for name, val in zip(processor.numeric_cols, numeric_vals):
        print(f"  {name}: {val:.3f}")

    print("Categorical predictions:")
    for k, v in categorical_vals.items():
        print(f"  {k}: {v}")
