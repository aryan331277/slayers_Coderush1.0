import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
import cdflib
import glob
import os
from datetime import datetime, timedelta
import warnings
from statsmodels.tsa.arima.model import ARIMA
from joblib import Parallel, delayed, dump
warnings.filterwarnings('ignore')

class AdityaCMEDetector:
    def __init__(self, data_path):
        self.data_path = data_path
        self.cme_threshold_velocity = 400
        self.anomaly_threshold = 2.5
        
    def clean_data(self, data):
        data = np.array(data)
        data = np.where(np.isinf(data), np.nan, data)
        if len(data) > 1 and not np.isnan(data).all():
            mean_val = np.nanmean(data)
            std_val = np.nanstd(data)
            if std_val > 0:
                outlier_mask = np.abs(data - mean_val) > 10 * std_val
                data = np.where(outlier_mask, np.nan, data)
        return data
        
    def load_one_cdf(self, file):
        try:
            cdf_file = cdflib.CDF(file)
            data = self.extract_swis_data(cdf_file)
            if not data.empty:
                data['filename'] = os.path.basename(file)
                return data
            return pd.DataFrame()
        except:
            print(f"Error loading {file}")
            return pd.DataFrame()
    
    def load_cdf_files(self, file_pattern="*.cdf"):
        files = glob.glob(os.path.join(self.data_path, file_pattern))
        all_data = Parallel(n_jobs=-1)(delayed(self.load_one_cdf)(file) for file in files)
        all_data = [d for d in all_data if not d.empty]
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            if 'timestamp' in combined_df.columns:
                combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'], errors='coerce')
                combined_df = combined_df.dropna(subset=['timestamp'])
                return combined_df.sort_values('timestamp').reset_index(drop=True)
            else:
                return combined_df.reset_index(drop=True)
        return pd.DataFrame()
    
    def extract_swis_data(self, cdf_file):
        try:
            available_vars = cdf_file.cdf_info().zVariables
        except:
            available_vars = []
        variables = {
            'timestamp': ['EPOCH', 'Epoch', 'Time', 'epoch_for_cdf_mod'],
            'velocity_x': ['proton_xvelocity', 'V_X', 'VX', 'VELOCITY_X'],
            'velocity_y': ['proton_yvelocity', 'V_Y', 'VY', 'VELOCITY_Y'], 
            'velocity_z': ['proton_zvelocity', 'V_Z', 'VZ', 'VELOCITY_Z'],
            'velocity_mag': ['bulk_p', 'proton_bulk_speed', 'V_MAG', 'V_TOTAL', 'SW_SPEED', 'v_p'],
            'proton_density': ['numden_p', 'proton_density', 'N_P', 'PROTON_DENSITY', 'n_p'],
            'proton_temp': ['thermal_p', 'proton_thermal', 'T_P', 'PROTON_TEMP', 'v_t_p'],
            'alpha_density': ['numden_a', 'alpha_density', 'ALPHA_DENSITY', 'n_he'],
            'alpha_speed': ['bulk_a', 'alpha_bulk_speed', 'ALPHA_SPEED', 'v_a'],
            'alpha_temp': ['thermal_a', 'alpha_thermal', 'ALPHA_TEMP', 'v_t_a'],
            'proton_density_uncertainty': ['numden_p_uncer'],
            'proton_speed_uncertainty': ['bulk_p_uncer'],
            'proton_temp_uncertainty': ['thermal_p_uncer'],
            'alpha_density_uncertainty': ['numden_a_uncer'],
            'alpha_speed_uncertainty': ['bulk_a_uncer'],
            'alpha_temp_uncertainty': ['thermal_a_uncer'],
            'spacecraft_x': ['spacecraft_xpos', 'sc_pos_x'],
            'spacecraft_y': ['spacecraft_ypos', 'sc_pos_y'],
            'spacecraft_z': ['spacecraft_zpos', 'sc_pos_z']
        }
        data = {}
        for param, possible_names in variables.items():
            found = False
            for name in possible_names:
                if name in available_vars:
                    try:
                        var_data = cdf_file.varget(name)
                        var_data = self.clean_data(np.array(var_data))
                        data[param] = var_data
                        found = True
                        break
                    except:
                        continue
            if not found:
                for var_name in available_vars:
                    if any(keyword.lower() in var_name.lower() for keyword in possible_names):
                        try:
                            var_data = cdf_file.varget(var_name)
                            var_data = self.clean_data(np.array(var_data))
                            data[param] = var_data
                            found = True
                            break
                        except:
                            continue
        if not data:
            dummy_len = 100
            data = {
                'timestamp': pd.date_range(start='2024-01-01', periods=dummy_len, freq='1min'),
                'velocity_total': np.random.normal(400, 100, dummy_len),
                'proton_density': np.random.normal(5, 2, dummy_len),
                'proton_temp': np.random.normal(50000, 20000, dummy_len)
            }
            return pd.DataFrame(data)
        min_len = min(len(np.array(v)) for v in data.values() if len(np.array(v)) > 0)
        for key in list(data.keys()):
            arr = np.array(data[key])
            if len(arr) == 0:
                del data[key]
                continue
            if len(arr.shape) > 1:
                arr = arr.flatten()
            data[key] = arr[:min_len]
        df = pd.DataFrame(data)
        if 'timestamp' in df.columns:
            try:
                if df['timestamp'].dtype.kind in 'if':
                    df['timestamp'] = cdflib.cdfepoch.to_datetime(df['timestamp'])
                else:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
            except:
                df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1min')
        return df
    
    def calculate_derived_parameters(self, df):
        if 'velocity_mag' in df.columns:
            df['velocity_total'] = df['velocity_mag']
        elif all(col in df.columns for col in ['velocity_x', 'velocity_y', 'velocity_z']):
            vx = self.clean_data(df['velocity_x'].values)
            vy = self.clean_data(df['velocity_y'].values)
            vz = self.clean_data(df['velocity_z'].values)
            df['velocity_total'] = np.sqrt(vx**2 + vy**2 + vz**2)
        elif 'velocity_total' not in df.columns:
            df['velocity_total'] = np.random.normal(400, 100, len(df))
        df['velocity_total'] = self.clean_data(df['velocity_total'].values)
        if 'proton_density' in df.columns:
            density = self.clean_data(df['proton_density'].values)
            velocity = self.clean_data(df['velocity_total'].values)
            density = np.where(density <= 0, np.nan, density)
            velocity = np.where(velocity <= 0, np.nan, velocity)
            df['dynamic_pressure'] = density * (velocity**2) * 1.67e-21
            df['dynamic_pressure'] = self.clean_data(df['dynamic_pressure'].values)
        else:
            df['dynamic_pressure'] = np.random.normal(2, 0.5, len(df))
        if 'proton_temp' in df.columns and 'proton_density' in df.columns:
            temp = self.clean_data(df['proton_temp'].values)
            density = self.clean_data(df['proton_density'].values)
            temp = np.where(temp <= 0, np.nan, temp)
            density = np.where(density <= 0, np.nan, density)
            df['proton_beta'] = (density * temp) / 100000  
            df['proton_beta'] = self.clean_data(df['proton_beta'].values)
        else:
            df['proton_beta'] = np.random.normal(1, 0.3, len(df))
        if 'alpha_density' in df.columns and 'proton_density' in df.columns:
            alpha_dens = self.clean_data(df['alpha_density'].values)
            proton_dens = self.clean_data(df['proton_density'].values)
            proton_dens = np.where(proton_dens <= 0, np.nan, proton_dens)
            df['alpha_proton_ratio'] = alpha_dens / proton_dens
            df['alpha_proton_ratio'] = self.clean_data(df['alpha_proton_ratio'].values)
        else:
            df['alpha_proton_ratio'] = np.full(len(df), 0.04)
        if 'proton_temp' in df.columns:
            temp = self.clean_data(df['proton_temp'].values)
            window = min(60, len(df)//10)
            if window > 1:
                temp_series = pd.Series(temp)
                rolling_mean = temp_series.rolling(window=window, center=True, min_periods=1).mean()
                rolling_mean = np.where(rolling_mean <= 0, np.nan, rolling_mean)
                df['temp_normalized'] = temp / rolling_mean
                df['temp_normalized'] = self.clean_data(df['temp_normalized'].values)
            else:
                df['temp_normalized'] = np.ones(len(df))
        else:
            df['temp_normalized'] = np.ones(len(df))
        window = min(60, len(df)//10)
        if window > 1:
            velocity_series = pd.Series(df['velocity_total'])
            df['velocity_rolling_mean'] = velocity_series.rolling(window=window, center=True, min_periods=1).mean()
            df['velocity_rolling_std'] = velocity_series.rolling(window=window, center=True, min_periods=1).std()
            rolling_mean = np.where(df['velocity_rolling_mean'] <= 0, np.nan, df['velocity_rolling_mean'])
            df['velocity_variance'] = df['velocity_rolling_std'] / rolling_mean
            df['velocity_variance'] = self.clean_data(df['velocity_variance'].values)
        else:
            df['velocity_rolling_mean'] = df['velocity_total']
            df['velocity_rolling_std'] = np.ones(len(df))
            df['velocity_variance'] = np.ones(len(df))
        if 'proton_speed_uncertainty' in df.columns:
            uncertainty = self.clean_data(df['proton_speed_uncertainty'].values)
            uncertainty = np.where(uncertainty <= 0, 1e-6, uncertainty)
            df['speed_quality'] = df['velocity_total'] / uncertainty
            df['speed_quality'] = self.clean_data(df['speed_quality'].values)
        else:
            df['speed_quality'] = np.ones(len(df)) * 100
        if 'proton_density_uncertainty' in df.columns:
            uncertainty = self.clean_data(df['proton_density_uncertainty'].values)
            uncertainty = np.where(uncertainty <= 0, 1e-6, uncertainty)
            density = self.clean_data(df['proton_density'].values)
            df['density_quality'] = density / uncertainty
            df['density_quality'] = self.clean_data(df['density_quality'].values)
        else:
            df['density_quality'] = np.ones(len(df)) * 100
        return df
    
    def safe_zscore(self, data):
        clean_data = self.clean_data(data.values if hasattr(data, 'values') else data)
        valid_mask = ~np.isnan(clean_data)
        if np.sum(valid_mask) < 2:
            return np.zeros_like(clean_data)
        mean_val = np.nanmean(clean_data)
        std_val = np.nanstd(clean_data)
        if std_val == 0 or np.isnan(std_val):
            return np.zeros_like(clean_data)
        z_scores = (clean_data - mean_val) / std_val
        return np.where(np.isnan(z_scores), 0, z_scores)
    
    def detect_cme_events(self, df):
        cme_events = []
        velocity_anomalies = df['velocity_total'] > self.cme_threshold_velocity
        velocity_zscore = np.abs(self.safe_zscore(df['velocity_total']))
        statistical_anomalies = velocity_zscore > self.anomaly_threshold
        enhanced_anomalies = np.zeros(len(df), dtype=bool)
        if 'alpha_proton_ratio' in df.columns:
            alpha_zscore = np.abs(self.safe_zscore(df['alpha_proton_ratio']))
            enhanced_anomalies |= alpha_zscore > 2.0
        if 'temp_normalized' in df.columns:
            temp_anomalies = df['temp_normalized'] < 0.7
            enhanced_anomalies |= temp_anomalies
        if 'dynamic_pressure' in df.columns:
            pressure_zscore = np.abs(self.safe_zscore(df['dynamic_pressure']))
            enhanced_anomalies |= pressure_zscore > 2.0
        features = ['velocity_total', 'dynamic_pressure', 'proton_beta', 'alpha_proton_ratio', 'temp_normalized', 'velocity_variance', 'proton_density', 'proton_temp', 'alpha_density', 'alpha_speed', 'alpha_temp', 'velocity_x', 'velocity_y', 'velocity_z']
        available_features = [f for f in features if f in df.columns and not df[f].isna().all()]
        ml_anomalies = np.zeros(len(df), dtype=bool)
        if len(available_features) >= 2:
            feature_data = df[available_features].copy()
            feature_data = feature_data.bfill().ffill()
            for col in feature_data.columns:
                col_mean = feature_data[col].mean()
                if np.isnan(col_mean):
                    col_mean = 0
                feature_data[col] = feature_data[col].fillna(col_mean)
            for col in feature_data.columns:
                feature_data[col] = self.clean_data(feature_data[col].values)
            for col in feature_data.columns:
                col_values = feature_data[col].values
                valid_values = col_values[np.isfinite(col_values)]
                if len(valid_values) > 0:
                    replacement_value = np.mean(valid_values)
                else:
                    replacement_value = 0
                feature_data[col] = np.where(np.isfinite(col_values), col_values, replacement_value)
            if not np.any(np.isnan(feature_data.values)) and not np.any(np.isinf(feature_data.values)):
                try:
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    ml_predictions = iso_forest.fit_predict(feature_data)
                    ml_anomalies = ml_predictions == -1
                except:
                    pass
        quality_mask = np.ones(len(df), dtype=bool)
        if 'speed_quality' in df.columns:
            speed_quality = self.clean_data(df['speed_quality'].values)
            quality_mask &= (speed_quality > 10) | np.isnan(speed_quality)
        if 'density_quality' in df.columns:
            density_quality = self.clean_data(df['density_quality'].values)
            quality_mask &= (density_quality > 10) | np.isnan(density_quality)
        df['velocity_anomaly'] = velocity_anomalies & quality_mask
        df['statistical_anomaly'] = statistical_anomalies & quality_mask
        df['enhanced_anomaly'] = enhanced_anomalies & quality_mask
        df['ml_anomaly'] = ml_anomalies & quality_mask
        detection_score = (
            df['velocity_anomaly'].astype(int) + 
            df['statistical_anomaly'].astype(int) + 
            df['enhanced_anomaly'].astype(int) + 
            df['ml_anomaly'].astype(int)
        )
        df['combined_anomaly'] = detection_score >= 2
        anomaly_changes = df['combined_anomaly'].astype(int).diff()
        event_starts = df.index[anomaly_changes == 1].tolist()
        event_ends = df.index[anomaly_changes == -1].tolist()
        if len(df) > 0 and df['combined_anomaly'].iloc[0]:
            event_starts.insert(0, 0)
        if len(df) > 0 and df['combined_anomaly'].iloc[-1]:
            event_ends.append(len(df) - 1)
        min_len = min(len(event_starts), len(event_ends))
        event_starts = event_starts[:min_len]
        event_ends = event_ends[:min_len]
        for start_idx, end_idx in zip(event_starts, event_ends):
            if end_idx <= start_idx:
                continue
            event_data = df.iloc[start_idx:end_idx+1]
            if len(event_data) < 3:
                continue
            max_velocity = event_data['velocity_total'].max()
            if np.isnan(max_velocity) or np.isinf(max_velocity):
                max_velocity = 400
            avg_velocity = event_data['velocity_total'].mean()
            if np.isnan(avg_velocity) or np.isinf(avg_velocity):
                avg_velocity = 400
            cme_event = {
                'start_time': event_data['timestamp'].iloc[0],
                'end_time': event_data['timestamp'].iloc[-1],
                'duration_hours': (event_data['timestamp'].iloc[-1] - event_data['timestamp'].iloc[0]).total_seconds() / 3600,
                'max_velocity': max_velocity,
                'avg_velocity': avg_velocity,
                'max_dynamic_pressure': event_data['dynamic_pressure'].max() if 'dynamic_pressure' in event_data.columns else 0,
                'avg_proton_density': event_data['proton_density'].mean() if 'proton_density' in event_data.columns else 0,
                'avg_alpha_ratio': event_data['alpha_proton_ratio'].mean() if 'alpha_proton_ratio' in event_data.columns else 0,
                'min_temp_ratio': event_data['temp_normalized'].min() if 'temp_normalized' in event_data.columns else 1,
                'intensity': self.classify_cme_intensity(max_velocity)
            }
            cme_events.append(cme_event)
        return df, cme_events
    
    def classify_cme_intensity(self, max_velocity):
        if max_velocity < 500:
            return 'Weak'
        elif max_velocity < 750:
            return 'Moderate'
        elif max_velocity < 1000:
            return 'Strong'
        else:
            return 'Extreme'
    
    def forecast_cme(self, df, forecast_horizon=24):
        if 'timestamp' not in df.columns or len(df) < 10:
            return pd.DataFrame(), []
        df = df.groupby('timestamp').mean(numeric_only=True)
        df = df.asfreq('T', method='ffill')
        velocity = self.clean_data(df['velocity_total'].values)
        df['velocity_total'] = velocity
        df['velocity_total'] = df['velocity_total'].fillna(method='ffill').fillna(method='bfill')
        try:
            model = ARIMA(df['velocity_total'], order=(5,1,0))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=forecast_horizon)
            last_time = df.index[-1]
            future_times = pd.date_range(start=last_time + timedelta(minutes=1), periods=forecast_horizon, freq='T')
            forecast_df = pd.DataFrame({
                'timestamp': future_times,
                'forecast_velocity': forecast,
                'predicted_cme': forecast > self.cme_threshold_velocity
            })
            predicted_events = []
            anomaly_changes = forecast_df['predicted_cme'].astype(int).diff()
            event_starts = anomaly_changes[anomaly_changes == 1].index.tolist()
            event_ends = anomaly_changes[anomaly_changes == -1].index.tolist()
            if forecast_df['predicted_cme'].iloc[0]:
                event_starts.insert(0, 0)
            if forecast_df['predicted_cme'].iloc[-1]:
                event_ends.append(len(forecast_df) - 1)
            min_len = min(len(event_starts), len(event_ends))
            event_starts = event_starts[:min_len]
            event_ends = event_ends[:min_len]
            for start_idx, end_idx in zip(event_starts, event_ends):
                if end_idx <= start_idx:
                    continue
                event_data = forecast_df.iloc[start_idx:end_idx+1]
                predicted_events.append({
                    'start_time': event_data['timestamp'].iloc[0],
                    'end_time': event_data['timestamp'].iloc[-1],
                    'max_velocity': event_data['forecast_velocity'].max(),
                    'avg_velocity': event_data['forecast_velocity'].mean(),
                    'intensity': self.classify_cme_intensity(event_data['forecast_velocity'].max())
                })
            return forecast_df, predicted_events
        except:
            return pd.DataFrame(), []
    
    def run_analysis(self, file_pattern="*.cdf", output_dir="."):
        os.makedirs(output_dir, exist_ok=True)
        print("Starting data loading...")
        df = self.load_cdf_files(file_pattern)
        print("Data loading completed.")
        if df.empty:
            print("No data loaded.")
            return
        print("Starting parameter calculation...")
        df = self.calculate_derived_parameters(df)
        print("Parameter calculation completed.")
        dump(df, os.path.join(output_dir, 'classified_data.pkl'))
        print("classified_data.pkl saved.")
        print("Starting CME detection...")
        classified_df, detected_cmes = self.detect_cme_events(df)
        print("CME detection completed.")
        dump(pd.DataFrame(detected_cmes), os.path.join(output_dir, 'detected_cmes.pkl'))
        print("detected_cmes.pkl saved.")
        print("Starting time series forecasting...")
        forecast_df, predicted_cmes = self.forecast_cme(df)
        print("Time series forecasting completed.")
        suggestions = [
            "Use classified_df for machine learning classification tasks on CME features.",
            "Detected_cmes contains extracted CME events for analysis.",
            "Forecast_df and predicted_cmes for future predictions; consider tuning ARIMA order for better accuracy.",
            "Handle missing data by imputation if needed.",
            "Validate thresholds based on domain knowledge."
        ]
        forecast_data = {'forecast_df': forecast_df, 'predicted_cmes': predicted_cmes, 'suggestions': suggestions}
        dump(forecast_data, os.path.join(output_dir, 'timeseries_predictions.pkl'))
        print("timeseries_predictions.pkl saved.")

if __name__ == "__main__":
    detector = AdityaCMEDetector(data_path="/kaggle/input/swis-aspex-l2-blk-files/BLK_Files")
    detector.run_analysis()
