# train.py
import torch
import torch.optim as optim
import torch.nn as nn
import time

def train_model(model, dataloader, num_items, epochs=20, lr=0.001, num_neg=4, loss_type='MSE'):
    start = time.time()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if loss_type == 'MSE':
        criterion = nn.MSELoss()
    elif loss_type == 'BCE':
        criterion = nn.BCELoss()
    else:
        raise ValueError("Unsupported loss_type")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for u, i, l in dataloader:
            pred = model(u, i)
            if loss_type == 'MSE':
                loss = criterion(pred, l)
            elif loss_type == 'BCE':
                # Positive
                pred_pos = pred
                loss_pos = criterion(pred_pos, l)
                # Negatives
                num_neg_samples = len(u) * num_neg
                neg_i = torch.randint(0, num_items, (num_neg_samples,))
                neg_u = u.repeat_interleave(num_neg)
                neg_l = torch.zeros(num_neg_samples)
                pred_neg = model(neg_u, neg_i)
                loss_neg = criterion(pred_neg, neg_l)
                loss = loss_pos + loss_neg
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch}: Loss {total_loss / len(dataloader)}')
    train_time = time.time() - start
    return model, train_time