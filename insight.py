# insight.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, kendalltau
from sklearn.metrics import pairwise_distances  

def analyze_insights(coresets_extra, train_ds, num_users, num_items, model_type, feedback_type):
    insights_file = f'insights_{model_type}_{feedback_type}.txt'
    with open(insights_file, 'w') as f:
        f.write("Q1-Q5 Insights Analysis\n")
    
    # Full dataset baselines (subsample for efficiency)
    full_indices = np.arange(len(train_ds))
    full_users = train_ds.users.numpy()
    full_items = train_ds.items.numpy()
    full_labels = train_ds.labels.numpy()
    
    # Q1: Interaction value (gradient norms, rarity)
    for method in ['gradient', 'gradient_no_pretrain']:
        if method in coresets_extra:
            coreset, user_embeds, gradients, _ = coresets_extra[method]
            if gradients is not None:
                # Vis: Histogram of gradient norms
                plt.figure()
                plt.hist(gradients, bins=50, alpha=0.5, label=method)
                plt.title(f'Q1: Gradient Norms Distribution ({method})')
                plt.legend()
                plt.savefig(f'q1_grad_norms_{method}.png')
                
                # Quant: Mean norm, rarity (long-tail items: low freq items ratio)
                mean_norm = np.mean(gradients)
                coreset_items = coreset.dataset.items.numpy()[coreset.indices] if hasattr(coreset, 'indices') else coreset.items.numpy()
                item_freq = np.bincount(full_items, minlength=num_items)
                rarity_score = np.mean(1 / (item_freq[coreset_items] + 1e-5))  # Inverse freq
                print(f"Q1 ({method}): Mean Gradient Norm = {mean_norm:.4f}, Rarity Score = {rarity_score:.4f}")
                with open(insights_file, 'a') as f:
                    f.write(f"Q1 ({method}): Mean Gradient Norm = {mean_norm:.4f}, Rarity Score = {rarity_score:.4f}\n")

    # Q2/Q3: Representativeness / Key users/items (coverage, activity hist)
    methods = list(coresets_extra.keys())  # 所有非full
    for method in methods:
        if method != 'random':  # Focus on advanced
            coreset, user_embeds, _, dist_matrix = coresets_extra[method]
            coreset_users = coreset.dataset.users.numpy()[coreset.indices] if hasattr(coreset, 'indices') else coreset.users.numpy()
            coreset_items = coreset.dataset.items.numpy()[coreset.indices] if hasattr(coreset, 'indices') else coreset.items.numpy()
            
            # Quant: Coverage
            user_coverage = len(np.unique(coreset_users)) / num_users
            item_coverage = len(np.unique(coreset_items)) / num_items
            print(f"Q2/Q3 ({method}): User Coverage = {user_coverage:.4f}, Item Coverage = {item_coverage:.4f}")
            with open(insights_file, 'a') as f:
                f.write(f"Q2/Q3 ({method}): User Coverage = {user_coverage:.4f}, Item Coverage = {item_coverage:.4f}\n")
            
            # Vis: Activity hist (user interaction counts)
            full_user_counts = np.bincount(full_users, minlength=num_users)
            coreset_user_counts = np.bincount(coreset_users, minlength=num_users)
            plt.figure()
            plt.hist(full_user_counts[full_user_counts > 0], bins=50, alpha=0.5, label='Full')
            plt.hist(coreset_user_counts[coreset_user_counts > 0], bins=50, alpha=0.5, label=method)
            plt.title(f'Q3: User Activity Distribution ({method} vs Full)')
            plt.legend()
            plt.savefig(f'q3_activity_{method}.png')
            
            # Q3 corr
            activity_corr, _ = pearsonr(full_user_counts, coreset_user_counts)
            print(f"Q3 ({method}): Activity Correlation = {activity_corr:.4f}")
            with open(insights_file, 'a') as f:
                f.write(f"Q3 ({method}): Activity Correlation = {activity_corr:.4f}\n")

    # Q4: Diversity/redundancy (pairwise dist, embed scatter)
    for method in ['clustering', 'clustering_no_pretrain', 'greedy', 'greedy_no_pretrain']:
        if method in coresets_extra:
            coreset, user_embeds, _, dist_matrix = coresets_extra[method]
            if user_embeds is not None:
                # Vis: Embed scatter (PCA 2D)
                pca = PCA(n_components=2)
                embeds_2d = pca.fit_transform(user_embeds)
                plt.figure()
                plt.scatter(embeds_2d[:, 0], embeds_2d[:, 1], alpha=0.5)
                plt.title(f'Q4: User Embeddings Scatter ({method})')
                plt.savefig(f'q4_embeds_scatter_{method}.png')
                
                # Quant: Redundancy (mean pairwise sim, assume dist is Euclidean -> sim=1/(1+dist))
                if dist_matrix is not None:
                    redundancy = np.mean(dist_matrix)  # Higher mean dist = lower redundancy
                else:
                    redundancy = np.mean(pairwise_distances(user_embeds))
                print(f"Q4 ({method}): Redundancy (Mean Dist) = {redundancy:.4f}")
                with open(insights_file, 'a') as f:
                    f.write(f"Q4 ({method}): Redundancy (Mean Dist) = {redundancy:.4f}\n")

    # Q5: Heuristic vs model-driven (pretrain vs no_pretrain corr)
    pairs = [('gradient', 'gradient_no_pretrain'), ('clustering', 'clustering_no_pretrain'), ('greedy', 'greedy_no_pretrain')]
    for pre, no_pre in pairs:
        if pre in coresets_extra and no_pre in coresets_extra:
            _, pre_embeds, pre_grads, _ = coresets_extra[pre]
            _, no_embeds, no_grads, _ = coresets_extra[no_pre]
            
            # Use embeds/grads as scores
            if pre_embeds is not None and no_embeds is not None:
                # Flatten or norm for corr
                pre_scores = np.linalg.norm(pre_embeds, axis=1)
                no_scores = np.linalg.norm(no_embeds, axis=1)
                correlation, _ = pearsonr(pre_scores, no_scores)
                tau, _ = kendalltau(pre_scores, no_scores)
                print(f"Q5 ({pre} vs {no_pre}): Correlation = {correlation:.4f}, Kendall Tau = {tau:.4f}")
                with open(insights_file, 'a') as f:
                    f.write(f"Q5 ({pre} vs {no_pre}): Correlation = {correlation:.4f}, Kendall Tau = {tau:.4f}\n")
            
            # Vis: Scores hist
            plt.figure()
            plt.hist(pre_scores, bins=50, alpha=0.5, label=pre)
            plt.hist(no_scores, bins=50, alpha=0.5, label=no_pre)
            plt.title(f'Q5: Embed Norms ({pre} vs {no_pre})')
            plt.legend()
            plt.savefig(f'q5_scores_{pre}.png')