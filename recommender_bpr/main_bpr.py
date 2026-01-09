import cornac
import pandas as pd
import os
import time
from cornac.eval_methods import StratifiedSplit
from cornac.metrics import Precision, Recall, MAP
import os
import warnings

def train_validate(path):
    print("Loading model") 
    model_bpr12 = cornac.models.BPR(
        name="BPR@12", 
        k=12, 
        max_iter=50, 
        learning_rate=0.01, 
        lambda_reg=0.01,         # Added explicit regularization
        num_threads=0,           # 0 utilizes all available CPU cores (best for Mac M1/M2/M3)
        verbose=True,
        seed=None                # IMPORTANT: Setting a seed forces num_threads=1, making it very slow
    )
    try:
        model_bpr12.load(os.path.join("checkpoint", "BPR@12"))
    except FileNotFoundError:
        pass

    print("Loading dataset")
    train_review_path = path
    df_train = pd.read_csv(train_review_path, usecols=['customer_id', 'article_id', 't_dat'])
    user_counts = df_train.groupby('customer_id').size()
    active_users = user_counts[user_counts >= 5].index
    df_train = df_train[df_train['customer_id'].isin(active_users)]


    df_train['rating'] = 1.0
    df_train['t_dat'] = pd.to_datetime(df_train['t_dat']).astype('int64') // 10**9

    data = list(df_train[['customer_id', 'article_id', 'rating', 't_dat']].itertuples(index=False, name=None))
    del df_train # Clean up memory immediately

    ss = StratifiedSplit(
        data=data,
        test_size=0.01, # For massive data, a smaller test % still yields thousands of users
        chrono=True,
        fmt='UIRT',
        exclude_unknowns=True,
        seed=1,
        verbose=True
    )
    
    print("Training and Validating a BPR model...")
    
    start = time.time()
    metrics = [Precision(k=12), Recall(k=12), MAP()]
    exp = cornac.Experiment(
        eval_method=ss,
        models=[model_bpr12],
        metrics=metrics,
        user_based=True,
    )
    exp.run()#(model_bpr12, metrics, ss.test_set, True, True)

    cornac_time = time.time() - start
    print(f"Full validation time: {cornac_time:.2f}s")
    model_bpr12.save("checkpoints")
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    train_validate(os.path.join("data", "transactions_train.csv"))
