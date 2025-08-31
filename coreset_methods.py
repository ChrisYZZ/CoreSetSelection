# coreset_methods.py
import torch
import torch.nn as nn
import time
import numpy as np
import pandas as pd  # Added import to fix NameError
from tqdm import tqdm  # Progress bar
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from torch.utils.data import DataLoader, Subset
from model import SimpleMF
from train import train_model

# Method-specific parameters
RANDOM_PARAMS = {'ratio': 0.2}
CLUSTERING_PARAMS = {'ratio': 0.2, 'num_clusters': 10, 'mf_epochs': 5, 'embed_size': 32}  # mf_epochs=1 for faster test
GRADIENT_PARAMS = {'ratio': 0.2, 'mf_epochs': 5, 'embed_size': 32, 'subsample': None}  # mf_epochs=1
DYNAMIC_PARAMS = {'ratio': 0.2, 'mf_epochs': 5, 'embed_size': 32, 'subsample': None}  # Added mf_epochs and subsample for k-greedy

def random_coreset(dataset, num_users, num_items, **kwargs):
    params = {**RANDOM_PARAMS, **kwargs}
    start = time.time()
    indices = np.random.choice(len(dataset), int(len(dataset) * params['ratio']), replace=False)
    subset_time = time.time() - start
    return Subset(dataset, indices), subset_time

def clustering_coreset(dataset, num_users, num_items, **kwargs):
    params = {**CLUSTERING_PARAMS, **kwargs}
    start = time.time()

    # Pretrain SimpleMF
    mf = SimpleMF(num_users, num_items, params.get('feedback_type', 'explicit'), params['embed_size'])  # 传入feedback_type
    mf_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    mf, _ = train_model(mf, mf_loader, num_items, epochs=params['mf_epochs'], lr=0.001, num_neg=4, loss_type=params['loss_type'])

    # Get user embeddings 
    user_embeds = mf.user_embed.weight.detach().cpu().numpy()

    # KMeans
    kmeans = KMeans(n_clusters=params['num_clusters'])
    labels = kmeans.fit_predict(user_embeds)

    # Select per cluster using ANN
    selected = []
    for c in range(params['num_clusters']):
        cluster_indices = np.where(labels == c)[0]
        if len(cluster_indices) == 0:
            continue
        centroid = kmeans.cluster_centers_[c]
        nn = NearestNeighbors(n_neighbors=int(len(cluster_indices) * params['ratio']))
        nn.fit(user_embeds[cluster_indices])
        _, indices_nn = nn.kneighbors([centroid])
        selected.extend(cluster_indices[indices_nn.flatten()])

    # Representativeness （dist_matrix提取）
    distances = []
    dist_matrix = None  # 如果需要全dist_matrix，计算一次；否则None
    for c in range(params['num_clusters']):
        cluster_embeds = user_embeds[labels == c]
        if len(cluster_embeds) > 1:
            cluster_dist = pairwise_distances(cluster_embeds)
            dist = cluster_dist.mean()
            distances.append(dist)
            # 目前只有per-cluster mean
    rep_score = np.mean(distances) if distances else 0
    print(f"Clustering Representativeness Score: {rep_score}")

    subset_time = time.time() - start
    return Subset(dataset, selected), subset_time, user_embeds, None, dist_matrix  

def clustering_no_pretrain_coreset(dataset, num_users, num_items, **kwargs):
    params = {**CLUSTERING_PARAMS, **kwargs}
    start = time.time()

    # No pretrain, random init SimpleMF
    mf = SimpleMF(num_users, num_items, params.get('feedback_type', 'explicit'), params['embed_size'])  # 传入feedback_type

    # Get user embeddings (random)
    user_embeds = mf.user_embed.weight.detach().cpu().numpy()

    # KMeans
    kmeans = KMeans(n_clusters=params['num_clusters'])
    labels = kmeans.fit_predict(user_embeds)

    # Select per cluster using ANN
    selected = []
    for c in range(params['num_clusters']):
        cluster_indices = np.where(labels == c)[0]
        if len(cluster_indices) == 0:
            continue
        centroid = kmeans.cluster_centers_[c]
        nn = NearestNeighbors(n_neighbors=int(len(cluster_indices) * params['ratio']))
        nn.fit(user_embeds[cluster_indices])
        _, indices_nn = nn.kneighbors([centroid])
        selected.extend(cluster_indices[indices_nn.flatten()])

    # Representativeness (same as original)
    distances = []
    dist_matrix = None
    for c in range(params['num_clusters']):
        cluster_embeds = user_embeds[labels == c]
        if len(cluster_embeds) > 1:
            dist = pairwise_distances(cluster_embeds).mean()
            distances.append(dist)
    rep_score = np.mean(distances) if distances else 0
    print(f"Clustering No Pretrain Representativeness Score: {rep_score}")

    subset_time = time.time() - start
    return Subset(dataset, selected), subset_time, user_embeds, None, dist_matrix

def gradient_coreset(dataset, num_users, num_items, **kwargs):
    params = {**GRADIENT_PARAMS, **kwargs}
    start = time.time()

    # Pretrain SimpleMF
    mf = SimpleMF(num_users, num_items, params.get('feedback_type', 'explicit'), params['embed_size'])
    mf_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    mf, _ = train_model(mf, mf_loader, num_items, epochs=params['mf_epochs'], lr=0.001, num_neg=4, loss_type=params['loss_type'])

    # Compute gradients
    gradients = []
    user_embeds = mf.user_embed.weight.detach().cpu().numpy()
    for u, i, l in DataLoader(dataset, batch_size=1):
        pred = mf(u, i)
        if params['loss_type'] == 'MSE':
            loss = nn.MSELoss()(pred, l)
        else:
            loss = nn.BCELoss()(pred, l)
        loss.backward()
        grad_norm = torch.norm(mf.user_embed.weight.grad[u.item()]).item()  # Example: user embed grad
        gradients.append(grad_norm)
        mf.zero_grad()

    gradients = np.array(gradients)
    indices = np.argsort(gradients)[-int(len(dataset) * params['ratio']):]  # Top gradients

    subset_time = time.time() - start
    return Subset(dataset, indices), subset_time, user_embeds, gradients, None

def gradient_no_pretrain_coreset(dataset, num_users, num_items, **kwargs):
    params = {**GRADIENT_PARAMS, **kwargs}
    start = time.time()

    # No pretrain, random init
    mf = SimpleMF(num_users, num_items, params.get('feedback_type', 'explicit'), params['embed_size'])

    # Compute gradients (on random model)
    gradients = []
    user_embeds = mf.user_embed.weight.detach().cpu().numpy()
    for u, i, l in DataLoader(dataset, batch_size=1):
        pred = mf(u, i)
        if params['loss_type'] == 'MSE':
            loss = nn.MSELoss()(pred, l)
        else:
            loss = nn.BCELoss()(pred, l)
        loss.backward()
        grad_norm = torch.norm(mf.user_embed.weight.grad[u.item()]).item()
        gradients.append(grad_norm)
        mf.zero_grad()

    gradients = np.array(gradients)
    indices = np.argsort(gradients)[-int(len(dataset) * params['ratio']):]

    subset_time = time.time() - start
    return Subset(dataset, indices), subset_time, user_embeds, gradients, None

def greedy_coreset(dataset, num_users, num_items, **kwargs):
    params = {**DYNAMIC_PARAMS, **kwargs}
    start = time.time()

    # Pretrain SimpleMF to get embeddings (similar to clustering/gradient)
    mf = SimpleMF(num_users, num_items, params.get('feedback_type', 'explicit'), params['embed_size'])
    mf_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    mf, _ = train_model(mf, mf_loader, num_items, epochs=params['mf_epochs'], lr=0.001, num_neg=4, loss_type=params['loss_type'])

    # Get user embeddings (focus on users for recsys) 
    user_embeds = mf.user_embed.weight.detach().cpu().numpy()  

    # Subsample if needed to avoid memory issues with pairwise_distances
    subsample = params['subsample'] or len(user_embeds)
    sub_indices = np.random.choice(len(user_embeds), subsample, replace=False) if params['subsample'] else np.arange(len(user_embeds))
    sub_embeds = user_embeds[sub_indices]

    # Compute pairwise distances 
    dist_matrix = pairwise_distances(sub_embeds)

    # k-center greedy: select k points to minimize max min-distance
    k = int(len(dataset) * params['ratio'])  # Adjust k to match coreset size (approx, since based on users)
    selected = []
    # Start with random point
    first_idx = np.random.choice(len(sub_embeds))
    selected.append(first_idx)
    min_dists = dist_matrix[first_idx]

    for _ in tqdm(range(1, k), desc="k-center greedy selection", ascii=True, leave=False):
        # Find point farthest from current selected
        farthest_idx = np.argmax(min_dists)
        selected.append(farthest_idx)
        # Update min_dists to min over new selected
        min_dists = np.minimum(min_dists, dist_matrix[farthest_idx])

    # Map back to original indices (user-based, but select corresponding dataset indices)
    selected_user_ids = sub_indices[selected]
    # For simplicity, select all interactions of selected users (to match coreset size approx)
    all_indices = np.arange(len(dataset))
    user_mask = np.isin(dataset.users.numpy(), selected_user_ids)
    selected_indices = all_indices[user_mask][:k]  # Truncate to exact k if needed

    subset_time = time.time() - start
    return Subset(dataset, selected_indices), subset_time, user_embeds, None, dist_matrix  

def greedy_no_pretrain_coreset(dataset, num_users, num_items, **kwargs):
    params = {**DYNAMIC_PARAMS, **kwargs}
    start = time.time()

    # No pretrain, random init SimpleMF
    mf = SimpleMF(num_users, num_items, params.get('feedback_type', 'explicit'), params['embed_size'])

    # Get user embeddings (random) （已有，改名）
    user_embeds = mf.user_embed.weight.detach().cpu().numpy()

    # Subsample if needed
    subsample = params['subsample'] or len(user_embeds)
    sub_indices = np.random.choice(len(user_embeds), subsample, replace=False) if params['subsample'] else np.arange(len(user_embeds))
    sub_embeds = user_embeds[sub_indices]

    # Compute pairwise distances （已有）
    dist_matrix = pairwise_distances(sub_embeds)

    # k-center greedy
    k = int(len(dataset) * params['ratio'])
    selected = []
    first_idx = np.random.choice(len(sub_embeds))
    selected.append(first_idx)
    min_dists = dist_matrix[first_idx]

    for _ in tqdm(range(1, k), desc="k-center greedy selection (no pretrain)"):
        farthest_idx = np.argmax(min_dists)
        selected.append(farthest_idx)
        min_dists = np.minimum(min_dists, dist_matrix[farthest_idx])

    # Map back to original
    selected_user_ids = sub_indices[selected]
    all_indices = np.arange(len(dataset))
    user_mask = np.isin(dataset.users.numpy(), selected_user_ids)
    selected_indices = all_indices[user_mask][:k]  # Truncate if needed

    subset_time = time.time() - start
    return Subset(dataset, selected_indices), subset_time, user_embeds, None, dist_matrix  # 新返回，同上