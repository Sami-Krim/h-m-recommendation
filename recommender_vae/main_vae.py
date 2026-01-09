import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from scipy.sparse import csr_matrix
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import random
from pathlib import Path

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()

# Define paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VAE_DIR = BASE_DIR / "recommender_vae"
CHECKPOINT_DIR = VAE_DIR / "checkpoints"
PRETRAINED_DIR = VAE_DIR / "pretrained"

# Ensure directories exist
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)

# Set device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Model definition
class HybridVAE(nn.Module):
    def __init__(self, num_items, embeddings, latent_dim=128):
        super().__init__()
        self.item_repr = nn.Embedding.from_pretrained(embeddings, freeze=True)
        
        self.en_cf = nn.Sequential(
            nn.Linear(num_items, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        fusion_dim = 256 + embeddings.shape[1]
        self.fc_mu = nn.Linear(fusion_dim, latent_dim)
        self.fc_logvar = nn.Linear(fusion_dim, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_items)
        )

    def forward(self, x):
        h_cf = self.en_cf(x)
        weights = x / (x.sum(dim=1, keepdim=True) + 1e-8)
        h_sem = torch.matmul(weights, self.item_repr.weight)
        combined = torch.cat([h_cf, h_sem], dim=1)
        mu, logvar = self.fc_mu(combined), self.fc_logvar(combined)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(mu) * std
        return self.decoder(z), mu, logvar

# --- Utility functions ---

def get_qwen_embeddings(article_ids, articles_df, filename="qwen_embeddings_full.pt"):
    """
    Handles embeddings specifically within the pretrained directory.
    """

    save_path = PRETRAINED_DIR / filename
    
    if save_path.exists():
        print(f"Loading cached Qwen embeddings from {save_path} ...")
        return torch.load(save_path, map_location=device)

    print("Cached file not found. Generating embeddings ...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    model = AutoModel.from_pretrained("Qwen/Qwen2.5-0.5B").to(device)
    model.eval()

    articles_df = articles_df.set_index('article_id')
    embeddings = []

    with torch.no_grad():
        for a_id in tqdm(article_ids, desc="Generating embeddings"):
            row = articles_df.loc[a_id]
            text = (f"Category: {row['product_group_name']} | "
                    f"Details: {str(row['detail_desc'])[:300]}")
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
            outputs = model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu()
            embeddings.append(emb)

    final_embs = torch.stack(embeddings)
    torch.save(final_embs, save_path)
    return final_embs.to(device)

def load_and_split_data(trans_path, articles_path):
    df = pd.read_csv(trans_path)
    df['t_dat'] = pd.to_datetime(df['t_dat'])
    split_date = df['t_dat'].max() - pd.Timedelta(days=7)
    
    train_df = df[df['t_dat'] < split_date].copy()
    test_df = df[df['t_dat'] >= split_date].copy()
    
    articles_df = pd.read_csv(articles_path)
    all_article_ids = articles_df['article_id'].unique().tolist()
    i_map = {id: i for i, id in enumerate(all_article_ids)}

    common_users = set(train_df['customer_id']).intersection(set(test_df['customer_id']))
    u_map = {id: i for i, id in enumerate(common_users)}
    
    train_df = train_df[train_df['customer_id'].isin(common_users)]
    test_df = test_df[test_df['customer_id'].isin(common_users)]
    
    def to_sparse(sub_df):
        rows = sub_df['customer_id'].map(u_map).values
        cols = sub_df['article_id'].map(i_map).values
        return csr_matrix((np.ones(len(sub_df)), (rows, cols)), shape=(len(common_users), len(all_article_ids)))

    return to_sparse(train_df), to_sparse(test_df), i_map, articles_df

class HMSparseDataset(Dataset):
    def __init__(self, train, test):
        self.train, self.test = train, test
    def __len__(self): return self.train.shape[0]
    def __getitem__(self, i): 
        return torch.from_numpy(self.train[i].toarray().squeeze()).float(), \
               torch.from_numpy(self.test[i].toarray().squeeze()).float()

def calculate_metrics(recon, target, exclude_input=None):
    recon = recon.clone()
    if exclude_input is not None:
        recon[exclude_input > 0] = -1e9
    _, topk = torch.topk(recon, 12, dim=1)
    hits = torch.gather(target, 1, topk)
    actual_pos = torch.clamp(target.sum(dim=1), min=1)
    recall = (hits.sum(dim=1) / actual_pos).mean().item()
    precisions = torch.cumsum(hits, dim=1) / torch.arange(1, 13, device=device)
    ap = ((precisions * hits).sum(dim=1) / torch.clamp(hits.sum(dim=1), min=1)).mean().item()
    return ap, recall

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="HybridVAE training and validation script")
    parser.add_argument('mode', choices=['train', 'valid'], help="Mode: 'train' or 'valid'")
    parser.add_argument('--trans_path', type=str, default=str(DATA_DIR / "transactions_train.csv"))
    parser.add_argument('--articles_path', type=str, default=str(DATA_DIR / "articles.csv"))
    parser.add_argument('--model_name', type=str, default="best_hybrid_vae.pt")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    args = parser.parse_args()

    # Determine model load/save path based on mode
    # If training, we save to checkpoints. If validating, we look in pretrained first, then checkpoints.
    pretrained_model = PRETRAINED_DIR / args.model_name
    checkpoint_model = CHECKPOINT_DIR / args.model_name
    
    # 1. Load Data
    train_m, test_m, i_map, articles_df = load_and_split_data(args.trans_path, args.articles_path)
    
    # 2. Manage Embeddings
    qwen_embs = get_qwen_embeddings(list(i_map.keys()), articles_df)
    
    # 3. Initialize Model
    model = HybridVAE(len(i_map), qwen_embs).to(device)

    train_loader = DataLoader(HMSparseDataset(train_m, train_m), batch_size=128, shuffle=True)
    val_loader = DataLoader(HMSparseDataset(train_m, test_m), batch_size=128, shuffle=False)

    if args.mode == 'train':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_map = 0.0
        
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss = 0
            for x_train, _ in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
                x_train = x_train.to(device)
                optimizer.zero_grad()
                recon, mu, logvar = model(x_train)
                neg_ll = -torch.mean(torch.sum(F.log_softmax(recon, dim=1) * x_train, dim=1))
                kld = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
                loss = neg_ll + 0.1 * kld
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation with Recall restored
            model.eval()
            val_map, val_recall = 0, 0
            with torch.no_grad():
                for xt, yt in tqdm(val_loader, desc=f"Epoch {epoch} [Valid]"):
                    xt, yt = xt.to(device), yt.to(device)
                    h_cf = model.en_cf(xt)
                    weights = xt / (xt.sum(dim=1, keepdim=True) + 1e-8)
                    h_sem = torch.matmul(weights, model.item_repr.weight)
                    mu = model.fc_mu(torch.cat([h_cf, h_sem], dim=1))
                    r = model.decoder(mu)
                    
                    m12, r12 = calculate_metrics(r, yt, exclude_input=xt)
                    val_map += m12
                    val_recall += r12
            
            avg_map = val_map / len(val_loader)
            avg_recall = val_recall / len(val_loader)
            print(f"Epoch {epoch} | Loss: {train_loss/len(train_loader):.4f} | MAP@12: {avg_map:.6f} | Recall@12: {avg_recall:.6f}")

            if avg_map > best_map:
                best_map = avg_map
                torch.save(model.state_dict(), checkpoint_model)
                print(f" ==> New best model saved to {checkpoint_model}")

    elif args.mode == 'valid':
        load_path = pretrained_model if pretrained_model.exists() else checkpoint_model
        
        if not load_path.exists():
            print(f"Error: Model not found at {load_path}")
            return
        
        print(f"Loading model for final validation: {load_path}")
        model.load_state_dict(torch.load(load_path, map_location=device))
        model.eval()
        
        total_ap, total_recall = 0, 0
        with torch.no_grad():
            for xt, yt in tqdm(val_loader, desc="Final Eval"):
                xt, yt = xt.to(device), yt.to(device)
                h_cf = model.en_cf(xt)
                weights = xt / (xt.sum(dim=1, keepdim=True) + 1e-8)
                h_sem = torch.matmul(weights, model.item_repr.weight)
                mu = model.fc_mu(torch.cat([h_cf, h_sem], dim=1))
                r = model.decoder(mu)
                
                ap, rec = calculate_metrics(r, yt, exclude_input=xt)
                total_ap += ap
                total_recall += rec
        
        print(f"\nFinal results")
        print(f"MAP@12:    {total_ap/len(val_loader):.6f}")
        print(f"Recall@12: {total_recall/len(val_loader):.6f}")

if __name__ == "__main__":
    main()