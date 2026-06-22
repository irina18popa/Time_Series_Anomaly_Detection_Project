import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support

# ==========================================
# 0. CONFIGURARE PATH-URI
# ==========================================
DATA_PATH = './ServerMachineDataset' # Modifică dacă folderul are alt nume
CHECKPOINT_PATH = 'anomaly_transformer_smd.pth'
WINDOW_SIZE = 100
BATCH_SIZE = 32

# ==========================================
# 1. ARHITECTURA MODELULUI
# ==========================================
class AnomalyAttention(nn.Module):
    def __init__(self, d_model=512, h=8, window_size=100):
        super(AnomalyAttention, self).__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_sigma = nn.Linear(d_model, h) 
        self.out_proj = nn.Linear(d_model, d_model)
        
        idx = torch.arange(window_size, dtype=torch.float32)
        self.register_buffer('D', (idx.unsqueeze(1) - idx.unsqueeze(0)) ** 2)

    def forward(self, x):
        B, N, C = x.size() 
        Q = self.W_q(x).view(B, N, self.h, self.d_k).transpose(1, 2) 
        K = self.W_k(x).view(B, N, self.h, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.h, self.d_k).transpose(1, 2)
        
        sigma = self.W_sigma(x) 
        sigma = F.softplus(sigma) + 1e-5 
        sigma = sigma.transpose(1, 2).unsqueeze(-1) 
        
        D_expanded = self.D.unsqueeze(0).unsqueeze(0) 
        prior_assoc = (1.0 / (math.sqrt(2 * math.pi) * sigma)) * torch.exp(-D_expanded / (2 * sigma ** 2))
        prior_assoc = prior_assoc / (prior_assoc.sum(dim=-1, keepdim=True) + 1e-8) 
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        series_assoc = F.softmax(scores, dim=-1)
        
        Z = torch.matmul(series_assoc, V) 
        Z = Z.transpose(1, 2).contiguous().view(B, N, self.d_model)
        out = self.out_proj(Z)
        
        return out, prior_assoc, series_assoc

def compute_association_discrepancy(prior_list, series_list):
    P_all = torch.stack(prior_list, dim=1)
    S_all = torch.stack(series_list, dim=1)
    
    P_mean = P_all.mean(dim=2) + 1e-8
    S_mean = S_all.mean(dim=2) + 1e-8
    
    P_mean = P_mean / P_mean.sum(dim=-1, keepdim=True)
    S_mean = S_mean / S_mean.sum(dim=-1, keepdim=True)
    
    kl_P_S = F.kl_div(S_mean.log(), P_mean, reduction='none').sum(dim=-1) 
    kl_S_P = F.kl_div(P_mean.log(), S_mean, reduction='none').sum(dim=-1) 
    
    sym_kl = kl_P_S + kl_S_P 
    return sym_kl.mean(dim=1)

class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, h=8, d_ff=2048, dropout=0.1, window_size=100):
        super(EncoderLayer, self).__init__()
        self.attention = AnomalyAttention(d_model=d_model, h=h, window_size=window_size)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, prior, series = self.attention(x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.norm2(x)
        return x, prior, series

class AnomalyTransformer(nn.Module):
    def __init__(self, c_in=38, d_model=512, L=3, h=8, d_ff=2048, dropout=0.1, window_size=100):
        super(AnomalyTransformer, self).__init__()
        self.embedding = nn.Linear(c_in, d_model) 
        self.layers = nn.ModuleList([
            EncoderLayer(d_model=d_model, h=h, d_ff=d_ff, dropout=dropout, window_size=window_size) 
            for _ in range(L)
        ])
        self.projection = nn.Linear(d_model, c_in)

    def forward(self, x):
        prior_list, series_list = [], []
        x = self.embedding(x)
        for layer in self.layers:
            x, prior, series = layer(x)
            prior_list.append(prior)
            series_list.append(series)
        reconstruction = self.projection(x)
        return reconstruction, prior_list, series_list

# ==========================================
# 2. PROCESARE DATE ȘI EVALUARE
# ==========================================
class SMDDataset(Dataset):
    def __init__(self, data_array, window_size=100, is_train=True, scaler=None):
        raw_data = np.array(data_array, dtype=np.float32)
        if is_train:
            self.scaler = StandardScaler()
            self.data = self.scaler.fit_transform(raw_data)
        else:
            self.scaler = scaler
            self.data = self.scaler.transform(raw_data)
        self.window_size = window_size
        self.num_windows = len(self.data) // self.window_size
        self.data = self.data[:self.num_windows * self.window_size]
        self.windows = self.data.reshape(self.num_windows, self.window_size, -1)

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        return torch.tensor(self.windows[idx])

def load_smd_data(data_path):
    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')
    label_path = os.path.join(data_path, 'test_label')
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Nu am găsit folderul {train_path}. Verifică DATA_PATH!")
        
    machine_files = sorted([f for f in os.listdir(train_path) if f.endswith('.txt')])
    train_dfs, test_dfs, label_dfs = [], [], []
    
    for file_name in machine_files:
        train_dfs.append(pd.read_csv(os.path.join(train_path, file_name), header=None))
        test_dfs.append(pd.read_csv(os.path.join(test_path, file_name), header=None))
        label_dfs.append(pd.read_csv(os.path.join(label_path, file_name), header=None))

    return pd.concat(train_dfs, ignore_index=True), pd.concat(test_dfs, ignore_index=True), pd.concat(label_dfs, ignore_index=True)

def get_anomaly_scores(model, dataloader, device):
    model.eval()
    all_scores = []
    with torch.no_grad():
        for batch_x in dataloader:
            batch_x = batch_x.to(device)
            reconstruction, prior_list, series_list = model(batch_x)
            ass_dis = compute_association_discrepancy(prior_list, series_list) 
            rec_error = torch.mean((batch_x - reconstruction) ** 2, dim=-1) 
            score = F.softmax(-ass_dis, dim=1) * rec_error
            all_scores.append(score.cpu().numpy().flatten())
    return np.concatenate(all_scores)

def apply_point_adjust(predictions, labels):
    adjusted_preds = predictions.copy()
    in_anomaly_segment = False
    start_idx = 0
    segments = []
    for i in range(len(labels)):
        if labels[i] == 1 and not in_anomaly_segment:
            start_idx = i
            in_anomaly_segment = True
        elif labels[i] == 0 and in_anomaly_segment:
            segments.append((start_idx, i))
            in_anomaly_segment = False
    if in_anomaly_segment:
        segments.append((start_idx, len(labels)))
        
    for (start, end) in segments:
        if np.sum(adjusted_preds[start:end]) > 0:
            adjusted_preds[start:end] = 1 
    return adjusted_preds

# ==========================================
# 3. EXECUȚIA PRINCIPALĂ (INFERENȚĂ)
# ==========================================
if __name__ == "__main__":
    print("=== Inițializare Pipeline de Testare Anomaly Transformer ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispozitiv de calcul: {device}")

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Eroare: Nu am găsit fișierul checkpoint '{CHECKPOINT_PATH}'. Asigură-te că modelul a fost salvat.")
        exit(1)

    print("\n1. Se încarcă datele (SMD)...")
    full_train_df, test_df, test_labels_df = load_smd_data(DATA_PATH)
    
    # Refacem split-ul pentru validare (necesar pentru calculul threshold-ului)
    val_split_index = int(len(full_train_df) * 0.8)
    train_df = full_train_df.iloc[:val_split_index].copy()
    val_df = full_train_df.iloc[val_split_index:].copy()
    
    print("2. Normalizare date și creare DataLoaders...")
    train_dataset = SMDDataset(train_df, window_size=WINDOW_SIZE, is_train=True)
    val_dataset = SMDDataset(val_df, window_size=WINDOW_SIZE, is_train=False, scaler=train_dataset.scaler)
    test_dataset = SMDDataset(test_df, window_size=WINDOW_SIZE, is_train=False, scaler=train_dataset.scaler)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("\n3. Se încarcă Arhitectura și Ponderile Modelului...")
    model = AnomalyTransformer(c_in=38, d_model=512, L=3, h=8, window_size=WINDOW_SIZE).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    model.eval()

    print("\n4. Extragere Scoruri de Anomalie...")
    print(" -> Rulăm inferența pe setul de validare...")
    val_scores = get_anomaly_scores(model, val_loader, device)

    print(" -> Rulăm inferența pe setul de testare...")
    test_scores = get_anomaly_scores(model, test_loader, device)

    # Stabilirea Threshold-ului pe validare
    r_ratio = 0.005 
    threshold = np.percentile(val_scores, 100 * (1 - r_ratio))
    print(f"\nThreshold calculat (top 0.5% din Validare): {threshold:.6f}")

    print("\n5. Calculare Metrici Finale...")
    raw_test_preds = (test_scores >= threshold).astype(int)
    raw_labels = test_labels_df.values.flatten()
    test_labels_array = raw_labels[:len(raw_test_preds)]

    adjusted_preds = apply_point_adjust(raw_test_preds, test_labels_array)

    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels_array, adjusted_preds, average='binary', zero_division=0
    )

    print("=======================================")
    print("   REZULTATE INFERENȚĂ SMD (TEST SET)  ")
    print("=======================================")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%")
    print("=======================================")