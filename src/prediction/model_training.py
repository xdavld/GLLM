import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from beer_data_preprocessing import BeerDataProcessor  # Import your preprocessing class
import matplotlib.pyplot as plt

class BeerDataset(Dataset):
    def __init__(self, x_numeric, x_categorical, x_text, conditions, targets):
        self.x_numeric = x_numeric
        self.x_categorical = x_categorical
        self.x_text = x_text
        self.conditions = conditions
        self.targets = targets
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return {
            'x_numeric': self.x_numeric[idx],
            'x_categorical': self.x_categorical[idx],
            'x_text': self.x_text[idx],
            'conditions': self.conditions[idx],
            'targets': self.targets[idx]
        }

def train_beer_cvae(csv_path, num_epochs=100, batch_size=32, learning_rate=1e-3):
    """Train CVAE on beer survey data"""
    
    # Process data
    processor = BeerDataProcessor()
    data = processor.prepare_training_data(csv_path)
    
    # Split data
    indices = torch.randperm(len(data['targets']))
    train_size = int(0.8 * len(indices))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    # Create datasets
    train_dataset = BeerDataset(
        data['x_numeric'][train_indices],
        data['x_categorical'][train_indices],
        data['x_text'][train_indices],
        data['conditions'][train_indices],
        data['targets'][train_indices]
    )
    
    test_dataset = BeerDataset(
        data['x_numeric'][test_indices],
        data['x_categorical'][test_indices],
        data['x_text'][test_indices],
        data['conditions'][test_indices],
        data['targets'][test_indices]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = CVAE(
        n_numeric=data['x_numeric'].shape[1],
        cat_cardinalities=data['cardinalities'],
        text_vocab_size=data['vocab_size'],
        cond_dim=data['conditions'].shape[1],
        latent_dim=16,  # Reduced from 32
        hidden_dim=128,  # Reduced from 256
        cat_emb_dim=8,   # Reduced from 16
        text_emb_dim=64, # Reduced from 128
        text_hid_dim=64  # Reduced from 128
    )
    
    print(f"Model initialized with:")
    print(f"  n_numeric: {data['x_numeric'].shape[1]}")
    print(f"  cardinalities: {data['cardinalities']}")
    print(f"  vocab_size: {data['vocab_size']}")
    print(f"  cond_dim: {data['conditions'].shape[1]}")
    
    # Check for NaN in input data
    print(f"Input data check:")
    print(f"  x_numeric has NaN: {torch.isnan(data['x_numeric']).any()}")
    print(f"  x_categorical has invalid values: {(data['x_categorical'] < 0).any()}")
    print(f"  x_text has invalid values: {(data['x_text'] < 0).any()}")
    print(f"  conditions has NaN: {torch.isnan(data['conditions']).any()}")
    print(f"  targets has NaN: {torch.isnan(data['targets']).any()}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    train_losses = []
    test_losses = []
    
    print("Starting training...")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss_epoch = 0
        train_recon_epoch = 0
        train_kld_epoch = 0
        train_reg_epoch = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            x_recon, mu, logvar, score_pred = model(
                batch['x_numeric'],
                batch['x_categorical'],
                batch['x_text'],
                batch['conditions']
            )
            
            # Prepare target for reconstruction loss
            cat_embeddings = []
            for i, emb in enumerate(model.cat_embs):
                cat_embeddings.append(emb(batch['x_categorical'][:, i]))
            
            text_features = model.text_enc(batch['x_text'])
            x_full = torch.cat([batch['x_numeric']] + cat_embeddings + [text_features], dim=-1)
            
            # Compute loss
            loss, recon_loss, kld, reg_loss = cvae_loss(
                x_recon, x_full, score_pred, batch['targets'], mu, logvar,
                beta=0.1, gamma=0.1  # Reduce these to prevent instability
            )
            
            # Check for NaN values
            if torch.isnan(loss):
                print("NaN detected in loss!")
                print(f"x_recon contains NaN: {torch.isnan(x_recon).any()}")
                print(f"x_full contains NaN: {torch.isnan(x_full).any()}")
                print(f"score_pred contains NaN: {torch.isnan(score_pred).any()}")
                print(f"mu contains NaN: {torch.isnan(mu).any()}")
                print(f"logvar contains NaN: {torch.isnan(logvar).any()}")
                print(f"targets contains NaN: {torch.isnan(batch['targets']).any()}")
                break
            
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Accumulate losses
            train_loss_epoch += loss.item()
            train_recon_epoch += recon_loss.item()
            train_kld_epoch += kld.item()
            train_reg_epoch += reg_loss.item()
        
        # Validation
        model.eval()
        test_loss_epoch = 0
        
        with torch.no_grad():
            for batch in test_loader:
                x_recon, mu, logvar, score_pred = model(
                    batch['x_numeric'],
                    batch['x_categorical'],
                    batch['x_text'],
                    batch['conditions']
                )
                
                # Prepare target for reconstruction loss
                cat_embeddings = []
                for i, emb in enumerate(model.cat_embs):
                    cat_embeddings.append(emb(batch['x_categorical'][:, i]))
                
                text_features = model.text_enc(batch['x_text'])
                x_full = torch.cat([batch['x_numeric']] + cat_embeddings + [text_features], dim=-1)
                
                loss, _, _, _ = cvae_loss(
                    x_recon, x_full, score_pred, batch['targets'], mu, logvar,
                    beta=0.5, gamma=1.0
                )
                
                test_loss_epoch += loss.item()
        
        # Average losses
        train_loss_avg = train_loss_epoch / len(train_loader)
        test_loss_avg = test_loss_epoch / len(test_loader)
        train_recon_avg = train_recon_epoch / len(train_loader)
        train_kld_avg = train_kld_epoch / len(train_loader)
        train_reg_avg = train_reg_epoch / len(train_loader)
        
        train_losses.append(train_loss_avg)
        test_losses.append(test_loss_avg)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss_avg:.4f} (Recon: {train_recon_avg:.4f}, KLD: {train_kld_avg:.4f}, Reg: {train_reg_avg:.4f})")
            print(f"  Test Loss: {test_loss_avg:.4f}")
            print()
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Progress')
    
    plt.subplot(1, 2, 2)
    # Generate some samples to visualize latent space
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(test_loader))
        _, mu, _, _ = model(
            sample_batch['x_numeric'][:10],
            sample_batch['x_categorical'][:10],
            sample_batch['x_text'][:10],
            sample_batch['conditions'][:10]
        )
        
        plt.scatter(mu[:, 0].cpu(), mu[:, 1].cpu(), c=sample_batch['targets'][:10], cmap='viridis')
        plt.xlabel('Latent Dim 1')
        plt.ylabel('Latent Dim 2')
        plt.title('Latent Space (First 2 Dims)')
        plt.colorbar(label='Satisfaction Score')
    
    plt.tight_layout()
    plt.show()
    
    return model, processor, data

# Example usage
if __name__ == '__main__':
    # Make sure to import your CVAE class and cvae_loss function
    from model import CVAE, cvae_loss  # Adjust import path
    
    # Train the model
    model, processor, data = train_beer_cvae('your_beer_survey.csv')
    
    # Save the trained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'processor': processor,
        'data_info': {
            'cardinalities':      data['cardinalities'],
            'vocab_size':         data['vocab_size'],
            'n_numeric':          data['x_numeric'].shape[1],
            'cond_dim':           data['conditions'].shape[1],
            'latent_dim':         model.latent_dim,
            'hidden_dim':         model.hidden_dim,
            'cat_emb_dim':        model.cat_embs[0].embedding_dim,
            'text_emb_dim':       model.text_enc.word_emb.embedding_dim,
            'text_hid_dim':       model.text_enc.rnn.hidden_size
        }
    }, 'beer_cvae_model.pth')

    
    print("Model saved to 'beer_cvae_model.pth'")