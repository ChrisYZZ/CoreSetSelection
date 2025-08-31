# data.py
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random

# Parameters (global defaults, can be overridden per method)
DATA_PATH = 'ml-1m/ratings.dat'
MIN_INTERACTIONS = 5
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

class MovieLensDataset(Dataset):
    def __init__(self, data):
        self.users = torch.tensor(data['user_id'].values, dtype=torch.long)
        self.items = torch.tensor(data['item_id'].values, dtype=torch.long)
        self.labels = torch.tensor(data['rating'].values, dtype=torch.float)  # 如果implicit则设置1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

def load_movielens(path=DATA_PATH, min_interactions=MIN_INTERACTIONS, feedback_type='explicit'):
    data = pd.read_csv(path, sep='::', names=['user_id', 'item_id', 'rating', 'timestamp'], engine='python')
    if feedback_type == 'implicit':
        data['rating'] = 1.0  # Implicit feedback

    # Remove users/items with few interactions
    while True:
        user_counts = data['user_id'].value_counts()
        item_counts = data['item_id'].value_counts()
        data = data[data['user_id'].isin(user_counts[user_counts >= min_interactions].index)]
        data = data[data['item_id'].isin(item_counts[item_counts >= min_interactions].index)]
        if len(data) == len(data):  # No more removal
            break

    # User interacted sets (for implicit evaluation)
    user_interacted = data.groupby('user_id')['item_id'].apply(set).to_dict()
    num_users = data['user_id'].max() + 1
    num_items = data['item_id'].max() + 1

    # Leave-one-out split
    train_data, val_data, test_data = [], [], []
    grouped = data.groupby('user_id')
    for u, group in grouped:
        group = group.sort_values('timestamp')
        test_row = group.iloc[-1]
        remaining = group.iloc[:-1]
        if len(remaining) > 0:
            val_idx = np.random.choice(len(remaining))
            val_row = remaining.iloc[val_idx]
            train_rows = remaining.drop(val_row.name)
        else:
            continue
        train_data.append(train_rows)
        val_data.append(pd.DataFrame([val_row]))
        test_data.append(pd.DataFrame([test_row]))

    train_df = pd.concat(train_data)
    val_df = pd.concat(val_data)
    test_df = pd.concat(test_data)

    train_ds = MovieLensDataset(train_df)
    val_ds = MovieLensDataset(val_df)
    test_ds = MovieLensDataset(test_df)

    return train_ds, val_ds, test_ds, num_users, num_items, user_interacted