import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio
from torch.utils.data import Dataset, DataLoader
from loss import SmoothedL1Loss  
from corn import CORN 
import os
import pandas as pd
import librosa
import csv
from torch.utils.data import random_split
import scipy.stats
from sklearn import metrics
from tqdm import tqdm

def load_audio(file_path, sr=16000):

    if not os.path.isfile(file_path):
        print(f"文件路径无效: {file_path}")
        raise FileNotFoundError(f"文件路径无效: {file_path}")

    audio, original_sr = librosa.load(file_path, sr=None) 

    if original_sr != sr:
        audio = librosa.resample(audio, orig_sr=original_sr, target_sr=sr)  

    target_samples = sr * 3

    if len(audio) < target_samples:
        padding = target_samples - len(audio)
        audio = torch.cat([torch.tensor(audio), torch.zeros(padding)], dim=0)
    elif len(audio) > target_samples:
        audio = audio[:target_samples]

    audio_tensor = torch.tensor(audio).float().unsqueeze(0) 
    return audio_tensor

def add_road(corn_model):
    if corn_model is not None and corn_model.training:
        folder_deg = r"train_data\deg" 
        folder_ref = r"train_data\ref" 

    else: 
        folder_deg = r"eval_data\deg" 
        folder_ref = r"eval_data\ref" 
    return folder_deg, folder_ref

class SpeechQualityDataset(Dataset):
    def __init__(self, clean_files, noise_files, target_sdr_file, is_training=True):

        self.noise_files = sorted([os.path.join(noise_files, f) for f in os.listdir(noise_files) if f.endswith('.wav')])
        self.clean_files = sorted([os.path.join(clean_files, f) for f in os.listdir(clean_files) if f.endswith('.wav')])

        self.target_mos = self.read_target_mos(target_sdr_file)  

        self.noise_to_clean_map = self.create_noise_to_clean_mapping(target_sdr_file)
        
    def __len__(self):
        return len(self.noise_files)
    
    def __getitem__(self, idx):
        folder_deg, folder_ref= add_road(corn_model)
        noise_file = self.noise_files[idx]
        noise_file_name = os.path.basename(noise_file)
        noise_file_name_all = os.path.join(folder_deg, noise_file_name)
        clean_file = self.noise_to_clean_map.get(noise_file_name)
        if clean_file is None:
            raise ValueError(f"未找到对应的干净音频文件: {noise_file_name}")

        noise_file_path = os.path.join(folder_deg, noise_file)
        clean_file_path = os.path.join(folder_ref, clean_file)
        
        noise_audio = load_audio(noise_file_path)
        clean_audio = load_audio(clean_file_path)
        
        target_mos = self.target_mos.get(noise_file_name_all)

        if target_mos is None:
            raise ValueError(f"未找到对应的MOS评分: {noise_file_name_all}")

        return noise_audio, clean_audio, target_mos

    def read_target_mos(self, target_sdr_file):
        target_mos = {}
        folder_deg, folder_ref = add_road(corn_model)
        with open(target_sdr_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                try:
                    noise_file_name = os.path.join(folder_deg, row[4]) 
                    clean_file_name = os.path.join(folder_ref, row[5])
                    target_score = float(row[9])  
                    target_mos[noise_file_name] = target_score

                except (IndexError, ValueError) as e:
                    continue  
        return target_mos

    def create_noise_to_clean_mapping(self, target_sdr_file):
        noise_to_clean_map = {}
        with open(target_sdr_file, 'r') as file:
            reader = csv.reader(file)
            next(reader) 
            for row in reader:
                try:
                    noise_file_name = row[4] 
                    clean_file_name = row[5] 
                    target_score = float(row[9]) 
                    noise_to_clean_map[noise_file_name] = clean_file_name

                except (IndexError, ValueError) as e:
                    print(f"Error processing row: {row} -> {e}")
                    continue 
        return noise_to_clean_map    

def save_scores_to_csv(file_names, fr_scores, nr_scores, output_file='scores.csv'):
    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:  
            writer.writerow(["File_Name", "FR_Score", "NR_Score"]) 

        for file_name, fr, nr in zip(file_names, fr_scores, nr_scores):
            writer.writerow([file_name, fr, nr])  

def train_corn(model, train_loader, test_loader, optimizer, epochs=1000, beta=1.0, device='cuda'):
    model.train()
    smoothed_l1 = SmoothedL1Loss(beta=beta) 
    best_mse = float('inf') 
    best_model_state = None 
    
    for epoch in range(epochs):
        total_loss = 0
        for x_i, r_j, target_mos in tqdm(train_loader):
            x_i, r_j, target_mos = x_i.to(device), r_j.to(device), target_mos.to(device)
            
            f_ij, n_i, _, _ = model(x_i, r_j)
            
            fr_loss = smoothed_l1(f_ij, target_mos)
            nr_loss = smoothed_l1(n_i, target_mos)

            loss = fr_loss + nr_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 1 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}')

        model.eval()
        fr_mse, nr_mse = eval_corn(model, test_loader, device)

        if fr_mse < best_mse:
            best_mse = fr_mse
            best_model_state = model.state_dict() 
    
    if best_model_state is not None:
        torch.save(best_model_state, "best_corn_model.pth")
        print(f"最好的模型保存至 'best_corn_model.pth'，MSE: {best_mse:.4f}")
    
    return model

def eval_corn(model, test_loader, device='cuda'):
    model.eval()
    fr_mse = 0
    nr_mse = 0
    all_fr_scores = []
    all_nr_scores = []
    all_target_mos = []
    all_filenames = [] 

    with torch.no_grad():
        for x_i, r_j, target_mos in test_loader:
            x_i, r_j, target_mos = x_i.to(device), r_j.to(device), target_mos.to(device)

            f_ij, n_i, _, _ = model(x_i, r_j)

            all_fr_scores.append(f_ij.cpu().numpy()) 
            all_nr_scores.append(n_i.cpu().numpy())
            all_target_mos.append(target_mos.cpu().numpy())

            all_filenames.extend(x_i.cpu().numpy()) 

    fr_mse = metrics.mean_squared_error(np.concatenate(all_fr_scores), np.concatenate(all_target_mos)) ** 0.5
    nr_mse = metrics.mean_squared_error(np.concatenate(all_nr_scores), np.concatenate(all_target_mos)) ** 0.5

    fr_plcc, _ = scipy.stats.pearsonr(np.concatenate(all_fr_scores), np.concatenate(all_target_mos))
    nr_plcc, _ = scipy.stats.pearsonr(np.concatenate(all_nr_scores), np.concatenate(all_target_mos))
    
    fr_srcc, _ = scipy.stats.spearmanr(np.concatenate(all_fr_scores), np.concatenate(all_target_mos))
    nr_srcc, _ = scipy.stats.spearmanr(np.concatenate(all_nr_scores), np.concatenate(all_target_mos))

    print(f'FR RMSE: {fr_mse}, NR RMSE: {nr_mse}')
    print(f'FR PLCC: {fr_plcc}, NR PLCC: {nr_plcc}')
    print(f'FR SRCC: {fr_srcc}, NR SRCC: {nr_srcc}')

    save_scores_to_csv(all_filenames, all_fr_scores, all_nr_scores, 'scores.csv')
    model.train()
    return fr_mse, nr_mse

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    corn_model = CORN().to(device)

    corn_model.train()

    clean_train_files = r"train_data\ref"
    noise_train_files = r"train_data\deg"

    clean_test_files = r"eval_data\ref"
    noise_test_files = r"eval_data\deg"
    
    target_sdr_file_train = r"train_data\TRAIN_file.csv" 
    target_sdr_file_test = r"eval_data\VAL_file.csv" 
    
    train_dataset = SpeechQualityDataset(clean_train_files, noise_train_files, target_sdr_file_train, is_training=True)

    corn_model.eval()
    test_dataset = SpeechQualityDataset(clean_test_files, noise_test_files, target_sdr_file_test, is_training=True)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    corn_model.train()

    optimizer = torch.optim.Adam(corn_model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)

    trained_model = train_corn(corn_model, train_loader, test_loader, optimizer, epochs=1000)
    
    torch.save(trained_model.state_dict(), "corn_model.pth")

    corn_model.eval()

    eval_corn(trained_model, test_loader, device)
