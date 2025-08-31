# evaluate.py
import torch
import numpy as np
import random

def evaluate(model, test_ds, num_items, user_interacted, k=10, num_neg_eval=99, feedback_type='explicit'):
    model.eval()
    if feedback_type == 'implicit':
        hr_scores, ndcg_scores = [], []
        with torch.no_grad():
            for idx in range(len(test_ds)):
                u, i, _ = test_ds[idx]
                u_id = u.item()
                pos_i = i.item()

                negatives = []
                while len(negatives) < num_neg_eval:
                    cand = random.randint(0, num_items - 1)
                    if cand not in user_interacted.get(u_id, set()):
                        negatives.append(cand)
                items_to_rank = [pos_i] + negatives
                u_tensor = torch.full((len(items_to_rank),), u_id, dtype=torch.long)
                i_tensor = torch.tensor(items_to_rank, dtype=torch.long)
                scores = model(u_tensor, i_tensor).numpy()

                ranked_indices = np.argsort(-scores)
                rank = np.where(ranked_indices == 0)[0][0] + 1

                hr = 1 if rank <= k else 0
                hr_scores.append(hr)

                if rank <= k:
                    dcg = 1 / np.log2(rank + 1)
                else:
                    dcg = 0
                idcg = 1 / np.log2(2)
                ndcg = dcg / idcg
                ndcg_scores.append(ndcg)

        return np.mean(hr_scores), np.mean(ndcg_scores)
    
    elif feedback_type == 'explicit':
        mse = 0.0
        with torch.no_grad():
            for idx in range(len(test_ds)):
                u, i, l = test_ds[idx]
                pred = model(u.unsqueeze(0), i.unsqueeze(0))
                mse += (pred - l) ** 2
        rmse = np.sqrt(mse / len(test_ds))
        return rmse.item()