# main.py
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from data import load_movielens
from model import NCF, SimpleMF
from coreset_methods import random_coreset, clustering_coreset, clustering_no_pretrain_coreset, gradient_coreset, gradient_no_pretrain_coreset, greedy_coreset, greedy_no_pretrain_coreset
from train import train_model
from evaluate import evaluate
import os  
import torch  
import insight  

parser = argparse.ArgumentParser()
parser.add_argument('--methods', type=str, default='all', help='Comma-separated: full,random,clustering,gradient,gradient_no_pretrain,greedy,greedy_no_pretrain or all')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--k', type=int, default=10)
parser.add_argument('--model_type', type=str, default='NCF', choices=['NCF', 'MF'], help='Model type: NCF or MF')
parser.add_argument('--feedback_type', type=str, default='explicit', choices=['implicit', 'explicit'], help='Feedback type')
parser.add_argument('--debug', action='store_true', help='Print debug shapes')
parser.add_argument('--insights', action='store_true', help='Run insights analysis for Q1-Q5')  
args = parser.parse_args()

# Load data
train_ds, val_ds, test_ds, num_users, num_items, user_interacted = load_movielens(feedback_type=args.feedback_type)

methods = args.methods.split(',') if args.methods != 'all' else ['full', 'random', 'clustering', 'gradient', 'gradient_no_pretrain', 'greedy', 'greedy_no_pretrain']

loss_type = 'BCE' if args.feedback_type == 'implicit' else 'MSE'

csv_file = f'results_{args.model_type}_{args.feedback_type}.csv'
if os.path.exists(csv_file):
    os.remove(csv_file)  # 清空旧CSV

# 存储coreset额外数据 for insights
coresets_extra = {}  # method: (coreset, user_embeds, gradients, dist_matrix)

for method in methods:
    print(f"Running {method} with {args.model_type} and {args.feedback_type}...")
    if args.model_type == 'NCF':
        model = NCF(num_users, num_items, args.feedback_type)
    elif args.model_type == 'MF':
        model = SimpleMF(num_users, num_items, args.feedback_type, embed_size=32)  # 使用SimpleMF作为主模型
    
    if method == 'full':
        select_time = 0
        loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        extra = (None, None, None)  # No extra for full
    else:
        kwargs = {'debug': args.debug, 'loss_type': loss_type, 'feedback_type': args.feedback_type}  # 传feedback_type到coreset
        if method == 'random':
            coreset, select_time = random_coreset(train_ds, num_users, num_items, **kwargs)
            extra = (None, None, None)  # Random无extra
        elif method == 'clustering':
            coreset, select_time, user_embeds, _, dist_matrix = clustering_coreset(train_ds, num_users, num_items, **kwargs)  
            extra = (user_embeds, None, dist_matrix)
        elif method == 'clustering_no_pretrain':
            coreset, select_time, user_embeds, _, dist_matrix = clustering_no_pretrain_coreset(train_ds, num_users, num_items, **kwargs)
            extra = (user_embeds, None, dist_matrix)
        elif method == 'gradient':
            coreset, select_time, user_embeds, gradients, _ = gradient_coreset(train_ds, num_users, num_items, **kwargs) 
            extra = (user_embeds, gradients, None)
        elif method == 'gradient_no_pretrain':
            coreset, select_time, user_embeds, gradients, _ = gradient_no_pretrain_coreset(train_ds, num_users, num_items, **kwargs)
            extra = (user_embeds, gradients, None)
        elif method == 'greedy':
            coreset, select_time, user_embeds, _, dist_matrix = greedy_coreset(train_ds, num_users, num_items, **kwargs) 
            extra = (user_embeds, None, dist_matrix)
        elif method == 'greedy_no_pretrain':
            coreset, select_time, user_embeds, _, dist_matrix = greedy_no_pretrain_coreset(train_ds, num_users, num_items, **kwargs)
            extra = (user_embeds, None, dist_matrix)
        else:
            continue
        loader = DataLoader(coreset, batch_size=args.batch_size, shuffle=True)
    
    model, train_time = train_model(model, loader, num_items, epochs=args.epochs, lr=args.lr, loss_type=loss_type)
    metrics = evaluate(model, test_ds, num_items, user_interacted, k=args.k, feedback_type=args.feedback_type)
    total_time = select_time + train_time
    
    # 立即打印
    if args.feedback_type == 'implicit':
        recall, ndcg = metrics
        print(f"{method.capitalize()}: Recall@K = {recall:.4f}, NDCG@K = {ndcg:.4f}, Select Time = {select_time:.2f}, Train Time = {train_time:.2f}, Total Time = {total_time:.2f}")
        result_dict = {
            'Method': method.capitalize(),
            'Feedback Type': args.feedback_type,
            'Model Type': args.model_type,
            'Recall@K': recall,
            'NDCG@K': ndcg,
            'Select Time': select_time,
            'Train Time': train_time,
            'Total Time': total_time
        }
    else:
        rmse = metrics
        print(f"{method.capitalize()}: RMSE = {rmse:.4f}, Select Time = {select_time:.2f}, Train Time = {train_time:.2f}, Total Time = {total_time:.2f}")
        result_dict = {
            'Method': method.capitalize(),
            'Feedback Type': args.feedback_type,
            'Model Type': args.model_type,
            'RMSE': rmse,
            'Select Time': select_time,
            'Train Time': train_time,
            'Total Time': total_time
        }
    
    # 追加到CSV
    result_df = pd.DataFrame([result_dict])
    header = not os.path.exists(csv_file)
    result_df.to_csv(csv_file, mode='a', header=header, index=False)
    
    # 保存coreset for insights
    if method != 'full':
        coresets_extra[method] = (coreset, *extra)
    
    # 释放内存
    del model
    del loader
    if 'coreset' in locals():
        del coreset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# 最终汇总打印
df = pd.read_csv(csv_file)
print(df)

# Insights调用 (if --insights)
if args.insights:
    insight.analyze_insights(coresets_extra, train_ds, num_users, num_items, args.model_type, args.feedback_type)