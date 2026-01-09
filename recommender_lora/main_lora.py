import argparse
import os
import sys
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from torch.optim import AdamW
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 1. HARDWARE DETECTION
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DTYPE = torch.float16
    USE_QUANTIZATION = True
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    DTYPE = torch.float32
    USE_QUANTIZATION = False
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
else:
    DEVICE = torch.device("cpu")
    DTYPE = torch.float32
    USE_QUANTIZATION = False

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
MAX_SEQ_LEN = 512
BATCH_SIZE = 16
ACC_STEPS = 2
LR = 1e-5
EPOCHS = 3
TEMP = 0.07

# --- 2. METRICS ---
def get_metrics(actual, predicted, k=12):
    """Calculates Mean Average Precision and Mean Recall at K."""
    ap_scores = []
    recall_scores = []

    for a, p in zip(actual, predicted):
        a = set(a)
        p = p[:k]

        # Calculate Average Precision (AP)
        score = 0.0
        num_hits = 0.0
        for i, item in enumerate(p):
            if item in a:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        ap_scores.append(score / min(len(a), k) if len(a) > 0 else 0)

        # Calculate Recall
        hits = len(a.intersection(set(p)))
        recall_scores.append(hits / len(a) if len(a) > 0 else 0)

    return np.mean(ap_scores), np.mean(recall_scores)

# --- 3. DATASET CLASSES ---
class HMRefinerDataset(Dataset):
    def __init__(self, user_histories, articles_df, max_history=10):
        self.samples = []
        # Create a fast lookup for descriptions
        desc_lookup = articles_df.set_index('article_id')['text'].to_dict()

        for user, history in user_histories.items():
            if len(history) < 2:
                continue

            # Target = last item bought | Context = preceding items
            target_id = history[-1]
            context_ids = history[-max_history-1 : -1]

            # Construct strings
            target_text = desc_lookup.get(target_id, "")
            context_text = " [SEP] ".join([desc_lookup.get(idx, "") for idx in context_ids])

            if target_text and context_text:
                self.samples.append({'user_text': context_text, 'item_text': target_text})

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

class PreTokenizedDataset(Dataset):
    def __init__(self, tokenized_list):
        self.data = tokenized_list
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# 4. LOSS FUNCTION
class InfoNCELoss(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, user_emb, item_emb):
        # L2 Normalize for Cosine Similarity
        user_emb = F.normalize(user_emb, p=2, dim=1)
        item_emb = F.normalize(item_emb, p=2, dim=1)

        # Logits = (Batch x Batch) similarity matrix
        logits = torch.matmul(user_emb, item_emb.t()) / self.temperature

        # Ground truth is the diagonal (each user matches their own target item)
        labels = torch.arange(logits.size(0)).to(DEVICE)
        return F.cross_entropy(logits, labels)

# 5. PRE-TOKENIZATION
def pre_tokenize(raw_dataset, tokenizer):
    print("Pre-tokenizing samples... This removes the training bottleneck.")
    tokenized_samples = []
    for sample in tqdm(raw_dataset, desc="Tokenizing"):
        u_tok = tokenizer(sample['user_text'], truncation=True, max_length=MAX_SEQ_LEN, padding='max_length')
        i_tok = tokenizer(sample['item_text'], truncation=True, max_length=256, padding='max_length')
        tokenized_samples.append({
            'u_input_ids': torch.tensor(u_tok['input_ids'], dtype=torch.long),
            'u_mask': torch.tensor(u_tok['attention_mask'], dtype=torch.bool),
            'i_input_ids': torch.tensor(i_tok['input_ids'], dtype=torch.long),
            'i_mask': torch.tensor(i_tok['attention_mask'], dtype=torch.bool)
        })
    return tokenized_samples

# 6. TRAINING FUNCTION
def train_one_epoch(loader, model, optimizer, loss_fn, accumulation_steps):
    model.train()
    total_loss = 0
    u_embs_list, i_embs_list = [], []

    optimizer.zero_grad()
    pbar = tqdm(loader, desc="Training")
    
    # Use CUDA scaler if available 
    if DEVICE.type == "cuda":
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = None

    for i, batch in enumerate(pbar):
        # Move tensors to device
        u_ids, u_mask = batch['u_input_ids'].to(DEVICE), batch['u_mask'].to(DEVICE)
        i_ids, i_mask = batch['i_input_ids'].to(DEVICE), batch['i_mask'].to(DEVICE)

        # Forward pass with AMP
        if DEVICE.type == "cuda" and scaler is not None:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                u_out = model(input_ids=u_ids, attention_mask=u_mask).last_hidden_state.mean(dim=1)
                i_out = model(input_ids=i_ids, attention_mask=i_mask).last_hidden_state.mean(dim=1)
                
                u_embs_list.append(u_out)
                i_embs_list.append(i_out)

                if (i + 1) % accumulation_steps == 0:
                    # Combine accumulated physical batches into one logical batch
                    loss = loss_fn(torch.cat(u_embs_list), torch.cat(i_embs_list))

                    # Backprop with Scaler 
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                    u_embs_list, i_embs_list = [], []
                    total_loss += loss.item()
                    pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        else:
            # Fallback for non-CUDA devices
            u_out = model(input_ids=u_ids, attention_mask=u_mask).last_hidden_state.mean(dim=1)
            i_out = model(input_ids=i_ids, attention_mask=i_mask).last_hidden_state.mean(dim=1)

            u_embs_list.append(u_out)
            i_embs_list.append(i_out)

            if (i + 1) % accumulation_steps == 0:
                # Combine accumulated physical batches into one logical batch
                loss = loss_fn(torch.cat(u_embs_list), torch.cat(i_embs_list))

                # Backprop
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                u_embs_list, i_embs_list = [], []
                total_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    return total_loss / (len(loader) // accumulation_steps) if len(loader) > 0 else 0

# 7. EMBEDDING FUNCTIONS
@torch.no_grad()
def get_article_embeddings(articles_df, model, tokenizer, batch_size=128, mode='default'):
    model.eval()
    texts = articles_df['text'].tolist()
    all_embs = []

    mode_str = "fine-tuned LoRA" if mode == 'finetuned' else "base model"
    print(f"Encoding {len(texts)} articles with {mode_str}...")
    for i in tqdm(range(0, len(texts), batch_size), desc="Article Encoding"):
        batch_texts = texts[i : i + batch_size]
        tok = tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)

        # Use same pooling as training (Mean Pooling)
        out = model(input_ids=tok['input_ids'], attention_mask=tok['attention_mask']).last_hidden_state.mean(dim=1)
        all_embs.append(out.cpu().float().numpy())

    return np.vstack(all_embs)

@torch.no_grad()
def get_user_embeddings(user_texts, model, tokenizer, batch_size=64):
    model.eval()
    all_embs = []

    print(f"Encoding {len(user_texts)} user histories...")
    for i in tqdm(range(0, len(user_texts), batch_size), desc="User Encoding"):
        batch_texts = user_texts[i : i + batch_size]
        tok = tokenizer(batch_texts, padding=True, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt").to(DEVICE)

        out = model(input_ids=tok['input_ids'], attention_mask=tok['attention_mask']).last_hidden_state.mean(dim=1)
        all_embs.append(out.cpu().float().numpy())

    return np.vstack(all_embs)

# 8. MAIN PIPELINE
def main():
    parser = argparse.ArgumentParser(description="H&M Semantic Recommender")
    parser.add_argument('--mode', type=str, choices=['default', 'finetuned'], required=True)
    parser.add_argument('--phase', type=str, choices=['train', 'eval'], required=True)
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size for training')
    parser.add_argument('--acc_steps', type=int, default=ACC_STEPS, help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=LR, help='Learning rate')
    parser.add_argument('--max_history', type=int, default=10, help='Max history items for user context')
    parser.add_argument('--test_users', type=int, default=5000, help='Number of test users for evaluation')
    args = parser.parse_args()

    # Get script directory and set paths relative to it
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    data_dir = os.path.join(project_root, args.data_dir.replace('../', ''))
    articles_csv = os.path.join(data_dir, "articles.csv")
    trans_csv = os.path.join(data_dir, "transactions_train.csv")
    checkpoint_dir = os.path.join(script_dir, "checkpoints")
    pretrained_dir = os.path.join(script_dir, "pretrained")
    # Check for pre-downloaded files in pretrained directory
    tokenized_data_file = os.path.join(pretrained_dir, "hm_pretokenized_data.pt")
    refined_embeddings_file = os.path.join(pretrained_dir, "refined_article_embeddings.npy")
    lora_epoch_4_dir = os.path.join(pretrained_dir, "lora_epoch_4")
    
    # Determine cache file use refined embeddings for finetuned mode if available
    # Also allow using refined embeddings for default mode as fallback (they're compatible)
    if args.mode == 'finetuned' and os.path.exists(refined_embeddings_file):
        cache_file = refined_embeddings_file
        print(f"Found pre-downloaded refined embeddings: {refined_embeddings_file}")
    elif args.mode == 'default' and os.path.exists(refined_embeddings_file):
        # Allow using refined embeddings for default mode too (they work as baseline)
        cache_file = refined_embeddings_file
        print(f"Using refined embeddings for default mode (compatible): {refined_embeddings_file}")
    else:
        cache_file = os.path.join(pretrained_dir, f"cache_{args.mode}.npy")
    
    # Use pretrained lora_epoch_4 checkpoint if available for eval
    original_checkpoint_dir = checkpoint_dir
    if args.mode == 'finetuned' and args.phase == 'eval':
        if os.path.exists(lora_epoch_4_dir) and os.path.exists(os.path.join(lora_epoch_4_dir, "adapter_config.json")):
            checkpoint_dir = lora_epoch_4_dir
            print(f"Found pre-downloaded checkpoint: {lora_epoch_4_dir}")
        elif os.path.exists(original_checkpoint_dir) and os.path.exists(os.path.join(original_checkpoint_dir, "adapter_config.json")):
            checkpoint_dir = original_checkpoint_dir
            print(f"Found checkpoint: {original_checkpoint_dir}")
    
    # Check for pre-tokenized data
    if os.path.exists(tokenized_data_file):
        print(f"Found pre-downloaded tokenized data: {tokenized_data_file}")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(pretrained_dir, exist_ok=True)

    print(f"\nINITIALIZING: {args.mode.upper()} mode on {DEVICE}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize base model
    if USE_QUANTIZATION and DEVICE.type == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        base_model = AutoModel.from_pretrained(MODEL_NAME, quantization_config=bnb_config, trust_remote_code=True)
    else:
        base_model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=DTYPE, trust_remote_code=True).to(DEVICE)

    # LoRA Configuration
    LORA_CONFIG = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none"
    )

    if args.mode == 'finetuned':
        if args.phase == 'eval' and os.path.exists(checkpoint_dir) and os.path.exists(os.path.join(checkpoint_dir, "adapter_config.json")):
            print(f"Loading pre-trained LoRA checkpoint from {checkpoint_dir}...")
            model = PeftModel.from_pretrained(base_model, checkpoint_dir)
            if DEVICE.type != "cuda" or not USE_QUANTIZATION:
                model = model.to(DEVICE)
            print(f"Checkpoint loaded successfully")
        else:
            if args.phase == 'train':
                print("Initializing new LoRA model for training...")
            else:
                print("No checkpoint found, using untrained LoRA model...")
            model = get_peft_model(base_model, LORA_CONFIG)
            if DEVICE.type != "cuda" or not USE_QUANTIZATION:
                model = model.to(DEVICE)
    else:
        model = base_model

    # Enable gradient checkpointing and input gradients for training
    if args.phase == 'train':
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        if hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()
        # Optional: torch.compile for faster training (matches notebook)
        # Note: Only works on CUDA, can cause issues on MPS/CPU
        if DEVICE.type == "cuda" and hasattr(torch, 'compile'):
            try:
                model = torch.compile(model)
                print("Model compiled with torch.compile for faster training")
            except Exception as e:
                print(f"torch.compile failed: {e}, continuing without compilation")

    if args.phase == 'train':
        if not (os.path.exists(articles_csv) and os.path.exists(trans_csv)):
            print(f"Error: Missing CSVs in {data_dir}")
            return

        print("\nPhase: Training")
        print("Loading and preprocessing data ...")
        
        # Load data
        articles = pd.read_csv(articles_csv, dtype={'article_id': str})
        transactions = pd.read_csv(trans_csv, dtype={'article_id': str}, parse_dates=['t_dat'])

        articles['text'] = (
            "Product: " + articles['prod_name'].fillna('') + " | " +
            "Category: " + articles['product_group_name'].fillna('') + " (" + articles['product_type_name'].fillna('') + ") | " +
            "Color: " + articles['colour_group_name'].fillna('') + " | " +
            "Details: " + articles['detail_desc'].fillna('No description provided.')
        )

        # Prepare user histories
        print("Building user histories ...")
        user_histories = transactions.sort_values(['customer_id', 't_dat']).groupby('customer_id')['article_id'].apply(list).to_dict()

        # Create dataset
        print("Creating training dataset...")
        raw_dataset = HMRefinerDataset(user_histories, articles, max_history=args.max_history)
        print(f"Created {len(raw_dataset)} training samples")

        # Pre-tokenize - use pre-downloaded file if available
        if os.path.exists(tokenized_data_file):
            print(f"Using pre-downloaded tokenized data from {tokenized_data_file}...")
            tokenized_data = torch.load(tokenized_data_file)
            print(f"   Loaded {len(tokenized_data)} pre-tokenized samples")
        else:
            print(f"Pre-tokenized data not found, generating from scratch ...")
            tokenized_data = pre_tokenize(raw_dataset, tokenizer)
            print(f"Saving pre-tokenized data to {tokenized_data_file} ...")
            torch.save(tokenized_data, tokenized_data_file)

        # Create data loader
        fast_loader = DataLoader(PreTokenizedDataset(tokenized_data), batch_size=args.batch_size, shuffle=True)

        # Setup training
        optimizer = AdamW(model.parameters(), lr=args.lr)
        loss_fn = InfoNCELoss(temperature=TEMP)

        # Training loop
        print(f"\nStarting training for {args.epochs} epochs ...")
        for epoch in range(args.epochs):
            avg_loss = train_one_epoch(fast_loader, model, optimizer, loss_fn, args.acc_steps)
            print(f"Epoch {epoch+1}/{args.epochs} | Avg Loss: {avg_loss:.4f}")
            
            # Save checkpoint
            epoch_checkpoint = os.path.join(checkpoint_dir, f"epoch_{epoch+1}")
            model.save_pretrained(epoch_checkpoint)
            print(f"Checkpoint saved to {epoch_checkpoint}")

        # Save final model
        model.save_pretrained(checkpoint_dir)
        print(f" Final model saved to {checkpoint_dir}")

    elif args.phase == 'eval':
        if not (os.path.exists(articles_csv) and os.path.exists(trans_csv)):
            print(f"Error: Missing CSVs in {data_dir}")
            return

        print("\nPhase: Evaluation")
        print("Loading and preprocessing data ...")

        # Load data
        articles = pd.read_csv(articles_csv, dtype={'article_id': str})
        transactions = pd.read_csv(trans_csv, dtype={'article_id': str}, parse_dates=['t_dat'])

        articles['text'] = (
            "Product: " + articles['prod_name'].fillna('') + " | " +
            "Category: " + articles['product_group_name'].fillna('') + " (" + articles['product_type_name'].fillna('') + ") | " +
            "Color: " + articles['colour_group_name'].fillna('') + " | " +
            "Details: " + articles['detail_desc'].fillna('No description provided.')
        )

        # Split train/val
        val_start_date = transactions['t_dat'].max() - pd.Timedelta(days=7)
        train_df = transactions[transactions['t_dat'] < val_start_date]
        test_df = transactions[transactions['t_dat'] >= val_start_date]

        # Build user histories
        user_histories = train_df.sort_values(['customer_id', 't_dat']).groupby('customer_id')['article_id'].apply(list).to_dict()

        # Prepare test users
        test_users = test_df['customer_id'].unique()[:args.test_users]
        user_queries = []
        valid_users = []
        ground_truth = []

        print(f"Building user profiles for {len(test_users)} users...")
        for user in tqdm(test_users, desc="Processing users"):
            history = user_histories.get(user, [])
            if not history:
                continue

            # Get ground truth items for validation
            actual_items = test_df[test_df['customer_id'] == user]['article_id'].tolist()
            if not actual_items:
                continue

            # Take last 10 items for the context window 
            last_items = history[-args.max_history:]
            history_texts = articles[articles['article_id'].isin(last_items)]['text'].tolist()

            # Format for Qwen Long-Context
            user_doc = "User's recent style preferences: " + " [SEP] ".join(history_texts)
            user_queries.append(user_doc)
            valid_users.append(user)
            ground_truth.append(actual_items)

        print(f"Valid test users: {len(user_queries)}")

        # Get embeddings - use pre-downloaded file if available
        if os.path.exists(cache_file):
            print(f"Using pre-downloaded article embeddings from {cache_file}...")
            article_embs = np.load(cache_file)
            print(f"Loaded embeddings shape: {article_embs.shape}")
        else:
            print("Pre-downloaded embeddings not found, generating from scratch ...")
            print(f"   This may take several minutes for {len(articles)} articles ...")
            article_embs = get_article_embeddings(articles, model, tokenizer, mode=args.mode)
            np.save(cache_file, article_embs)
            print(f"Article embeddings cached to {cache_file}")

        print("Generating user embeddings ...")
        user_embs = get_user_embeddings(user_queries, model, tokenizer)

        # Calculate similarity and predictions 
        print("Calculating semantic similarity ...")
        sim_matrix = cosine_similarity(user_embs, article_embs)
        top_12_indices = np.argsort(-sim_matrix, axis=1)[:, :12]

        article_ids = articles['article_id'].values
        predictions = [[article_ids[idx] for idx in user_row] for user_row in top_12_indices]

        # Calculate metrics
        map12, recall12 = get_metrics(ground_truth, predictions, k=12)

        print(f"\nComputed results ({args.mode.upper()}):")
        print(f"MAP@12 :    {map12:.5f}")
        print(f"Recall@12 : {recall12:.5f}")

if __name__ == "__main__":
    main()
