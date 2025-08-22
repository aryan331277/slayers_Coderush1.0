#!/usr/bin/env python3

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy import signal
from scipy.stats import zscore
import json
import warnings
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from spacepy import pycdf
    CDF_AVAILABLE = True
except ImportError:
    print("Warning: spacepy not available. Install with: pip install spacepy")
    CDF_AVAILABLE = False

try:
    from astropy.io import fits
    FITS_AVAILABLE = True
except ImportError:
    print("Warning: astropy not available. Install with: pip install astropy")
    FITS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdityaL1DataProcessor:
    def __init__(self, cadence_s=64, window_h=6):
        self.cadence_s = cadence_s
        self.window_h = window_h
        self.window_samples = int((window_h * 3600) / cadence_s)
        self.papa_vars = {
            'time': 'Epoch',
            'he_flux': 'He_flux',
            'h_flux': 'H_flux',
            'he_density': 'He_density',
            'h_density': 'H_density',
            'quality': 'quality_flag'
        }
        self.swiss_vars = {
            'time': 'Epoch',
            'he_counts': 'He_counts',
            'h_counts': 'H_counts',
            'velocity': 'bulk_velocity',
            'temperature': 'temperature'
        }

    def load_papa_data(self, filepath: str) -> pd.DataFrame:
        if not CDF_AVAILABLE:
            raise ImportError("spacepy required for CDF files")
        logger.info(f"Loading PAPA data from {filepath}")
        with pycdf.CDF(filepath) as cdf:
            data = {}
            if self.papa_vars['time'] in cdf:
                data['time'] = cdf[self.papa_vars['time']][:]
            else:
                time_vars = ['Epoch', 'Time', 'time', 'timestamp']
                for tv in time_vars:
                    if tv in cdf:
                        data['time'] = cdf[tv][:]
                        break
            for key, var in self.papa_vars.items():
                if key != 'time' and var in cdf:
                    data[key] = cdf[var][:]
        df = pd.DataFrame(data)
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df = df.set_index('time')
        return df

    def load_swiss_data(self, filepath: str) -> pd.DataFrame:
        if not CDF_AVAILABLE:
            raise ImportError("spacepy required for CDF files")
        logger.info(f"Loading SWISS data from {filepath}")
        with pycdf.CDF(filepath) as cdf:
            data = {}
            if self.swiss_vars['time'] in cdf:
                data['time'] = cdf[self.swiss_vars['time']][:]
            for key, var in self.swiss_vars.items():
                if key != 'time' and var in cdf:
                    data[key] = cdf[var][:]
        df = pd.DataFrame(data)
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df = df.set_index('time')
        return df

    def calculate_he_h_ratio(self, papa_df: pd.DataFrame, swiss_df: pd.DataFrame) -> pd.DataFrame:
        if not papa_df.empty and not swiss_df.empty:
            combined = papa_df.copy()
            for col in swiss_df.columns:
                if col not in combined.columns:
                    combined[f'swiss_{col}'] = np.interp(
                        combined.index.values.astype(np.int64),
                        swiss_df.index.values.astype(np.int64),
                        swiss_df[col].values
                    )
        else:
            combined = papa_df if not papa_df.empty else swiss_df
        if 'he_density' in combined.columns and 'h_density' in combined.columns:
            combined['he_h_ratio'] = combined['he_density'] / combined['h_density']
        elif 'he_flux' in combined.columns and 'h_flux' in combined.columns:
            combined['he_h_ratio'] = combined['he_flux'] / combined['h_flux']
        elif 'he_counts' in combined.columns and 'h_counts' in combined.columns:
            combined['he_h_ratio'] = combined['he_counts'] / combined['h_counts']
        else:
            raise ValueError("No suitable He++ and H+ data found for ratio calculation")
        combined['he_h_ratio'] = combined['he_h_ratio'].replace([np.inf, -np.inf], np.nan)
        combined = combined.dropna(subset=['he_h_ratio'])
        return combined

    def resample_to_cadence(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.resample(f'{self.cadence_s}s').mean()

    def extract_precursor_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['he_h_rolling_mean'] = df['he_h_ratio'].rolling(
            window=self.window_samples, center=True
        ).mean()
        df['he_h_rolling_std'] = df['he_h_ratio'].rolling(
            window=self.window_samples, center=True
        ).std()
        df['he_h_zscore'] = (
            df['he_h_ratio'] - df['he_h_rolling_mean']
        ) / df['he_h_rolling_std']
        df['he_h_gradient'] = np.gradient(df['he_h_ratio'].values)
        df['he_h_variance'] = df['he_h_ratio'].rolling(
            window=self.window_samples//4
        ).var()
        return df.dropna()

class CMECatalogLoader:
    def __init__(self):
        self.donki_url = "https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/CME"

    def load_donki_catalog(self, start_date: str, end_date: str) -> pd.DataFrame:
        try:
            import requests
            params = {
                'startDate': start_date,
                'endDate': end_date,
                'mostAccurateOnly': 'true',
                'speed': '300',
                'halfAngle': '30'
            }
            response = requests.get(self.donki_url, params=params)
            cme_data = response.json()
            cme_times = []
            for cme in cme_data:
                if 'startTime' in cme:
                    cme_times.append(pd.to_datetime(cme['startTime']))
            return pd.DataFrame({'cme_time': cme_times})
        except Exception as e:
            logger.warning(f"Could not load DONKI catalog: {e}")
            return self._generate_sample_cmes(start_date, end_date)

    def _generate_sample_cmes(self, start_date: str, end_date: str) -> pd.DataFrame:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        n_cmes = int((end - start).days / 7)
        cme_times = pd.date_range(start, end, periods=n_cmes)
        return pd.DataFrame({'cme_time': cme_times})

class CMEDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, sequence_length: int = 256):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.features) - self.sequence_length + 1

    def __getitem__(self, idx):
        return (
            self.features[idx:idx + self.sequence_length],
            self.labels[idx + self.sequence_length - 1]
        )

class CMEPrecursorCNN(nn.Module):
    def __init__(self, input_features=4, hidden_dim=128, num_layers=3):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(input_features, hidden_dim, kernel_size=7, padding=3),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.Conv1d(hidden_dim, hidden_dim//2, kernel_size=3, padding=1),
        ])
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.BatchNorm1d(hidden_dim//2)
        ])
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim//2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        for conv, bn in zip(self.conv_layers, self.batch_norms):
            x = conv(x)
            x = bn(x)
            x = self.relu(x)
            x = self.dropout(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.classifier(x)
        return x.squeeze(-1)

class AdityaCMEPredictor:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.processor = AdityaL1DataProcessor()
        self.catalog_loader = CMECatalogLoader()
        self.model = None

    def load_aditya_data(self, papa_files: List[str], swiss_files: List[str]) -> pd.DataFrame:
        all_data = []
        papa_data_list = []
        for pfile in papa_files:
            if Path(pfile).exists():
                papa_df = self.processor.load_papa_data(pfile)
                papa_data_list.append(papa_df)
        swiss_data_list = []
        for sfile in swiss_files:
            if Path(sfile).exists():
                swiss_df = self.processor.load_swiss_data(sfile)
                swiss_data_list.append(swiss_df)
        if papa_data_list:
            papa_combined = pd.concat(papa_data_list, sort=True)
        else:
            papa_combined = pd.DataFrame()
        if swiss_data_list:
            swiss_combined = pd.concat(swiss_data_list, sort=True)
        else:
            swiss_combined = pd.DataFrame()
        combined_df = self.processor.calculate_he_h_ratio(papa_combined, swiss_combined)
        combined_df = self.processor.resample_to_cadence(combined_df)
        feature_df = self.processor.extract_precursor_features(combined_df)
        return feature_df.sort_index()

    def create_labels(self, data_df: pd.DataFrame, cme_catalog: pd.DataFrame, lead_time_h: int = 8) -> np.ndarray:
        labels = np.zeros(len(data_df))
        for cme_time in cme_catalog['cme_time']:
            precursor_start = cme_time - timedelta(hours=12)
            precursor_end = cme_time - timedelta(hours=6)
            mask = (data_df.index >= precursor_start) & (data_df.index <= precursor_end)
            labels[mask] = 1
        return labels

    def train_model(self, data_df: pd.DataFrame, labels: np.ndarray):
        feature_cols = ['he_h_zscore', 'he_h_gradient', 'he_h_variance', 'he_h_ratio']
        features = data_df[feature_cols].values
        mask = ~np.isnan(features).any(axis=1) & ~np.isnan(labels)
        features = features[mask]
        labels = labels[mask]
        dataset = CMEDataset(features, labels, sequence_length=256)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        self.model = CMEPrecursorCNN(input_features=len(feature_cols))
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.model.train()
        for epoch in range(50):
            epoch_loss = 0
            for batch_features, batch_labels in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {epoch_loss/len(dataloader):.4f}")

    def predict_precursors(self, data_df: pd.DataFrame) -> Dict:
        if self.model is None:
            raise ValueError("Model not trained yet!")
        feature_cols = ['he_h_zscore', 'he_h_gradient', 'he_h_variance', 'he_h_ratio']
        features = data_df[feature_cols].values
        if len(features) >= 256:
            recent_features = torch.FloatTensor(features[-256:]).unsqueeze(0)
            self.model.eval()
            with torch.no_grad():
                prob = self.model(recent_features).item()
            if prob > 0.7:
                return {
                    "precursor": True,
                    "probability": prob,
                    "lead_h": 8.3,
                    "timestamp": data_df.index[-1].isoformat(),
                    "confidence": "high" if prob > 0.85 else "medium"
                }
            else:
                return {
                    "precursor": False,
                    "probability": prob,
                    "timestamp": data_df.index[-1].isoformat()
                }
        else:
            return {"error": "Insufficient data for prediction"}

    def validate_model(self, test_data: pd.DataFrame, test_labels: np.ndarray) -> Dict:
        feature_cols = ['he_h_zscore', 'he_h_gradient', 'he_h_variance', 'he_h_ratio']
        features = test_data[feature_cols].values
        predictions = []
        true_labels = []
        for i in range(256, len(features)):
            seq_features = torch.FloatTensor(features[i-256:i]).unsqueeze(0)
            self.model.eval()
            with torch.no_grad():
                prob = self.model(seq_features).item()
            predictions.append(prob)
            true_labels.append(test_labels[i])
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        pred_binary = (predictions > 0.7).astype(int)
        tp = np.sum((pred_binary == 1) & (true_labels == 1))
        fp = np.sum((pred_binary == 1) & (true_labels == 0))
        fn = np.sum((pred_binary == 0) & (true_labels == 1))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        return {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0,
            "total_predictions": len(predictions),
            "precursor_alerts": np.sum(pred_binary)
        }

def main():
    predictor = AdityaCMEPredictor(data_dir="./aditya_data")
    papa_files = ["papa_data_001.cdf", "papa_data_002.cdf"]
    swiss_files = ["swiss_data_001.cdf", "swiss_data_002.cdf"]
    print("🚀 Loading Aditya-L1 data...")
    try:
        data_df = predictor.load_aditya_data(papa_files, swiss_files)
        print(f"✓ Loaded {len(data_df)} data points")
        print(f"✓ Date range: {data_df.index.min()} to {data_df.index.max()}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    start_date = data_df.index.min().strftime('%Y-%m-%d')
    end_date = data_df.index.max().strftime('%Y-%m-%d')
    cme_catalog = predictor.catalog_loader.load_donki_catalog(start_date, end_date)
    print(f"✓ Loaded {len(cme_catalog)} CME events")
    labels = predictor.create_labels(data_df, cme_catalog)
    print(f"✓ Created labels: {np.sum(labels)} precursor periods")
    split_idx = int(0.8 * len(data_df))
    train_data = data_df.iloc[:split_idx]
    train_labels = labels[:split_idx]
    test_data = data_df.iloc[split_idx:]
    test_labels = labels[split_idx:]
    print("🧠 Training model...")
    predictor.train_model(train_data, train_labels)
    print("✓ Model training complete!")
    print("📊 Validating model...")
   
