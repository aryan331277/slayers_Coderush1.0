# %%
"""
Aditya-L1 MAG: CME, Shock, and Magnetic-Cloud Detection
Jupyter/VSCode-ready Python file (use Python Interactive / run cells with `# %%` separators).

What this notebook/script does:
- Load multiple NetCDF (.nc) MAG files (Aditya-L1) into an xarray Dataset
- Convert to a clean pandas.DataFrame indexed by time with Bx, By, Bz and |B|
- Provide 3 detection routines:
    1) CME detection (sustained southward Bz)
    2) Shock detection (sudden jumps in |B| / high derivative)
    3) Magnetic-cloud detection (low |B| variance + smooth rotation in transverse field)
- Visualization helpers (time series + hodogram)
- Event catalog export to CSV

Notes:
- The code attempts to auto-detect common variable names but you can explicitly pass variable names.
- Designed for clarity, robustness, and Jupyter friendliness (use in VS Code interactive window).

Dependencies:
- xarray, netCDF4, numpy, pandas, matplotlib, scipy

"""

# %%
# Auto-install dependencies if needed
import subprocess
import sys

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Check and install required packages
required_packages = {
    'xarray': 'xarray',
    'netCDF4': 'netcdf4', 
    'numpy': 'numpy',
    'pandas': 'pandas',
    'matplotlib': 'matplotlib',
    'scipy': 'scipy'
}

missing_packages = []
for package_name, pip_name in required_packages.items():
    try:
        __import__(package_name)
    except ImportError:
        print(f"Installing {package_name}...")
        try:
            install_package(pip_name)
        except Exception as e:
            missing_packages.append(package_name)
            print(f"Failed to install {package_name}: {e}")

if missing_packages:
    print(f"Please manually install these packages: {', '.join(missing_packages)}")
    print("Run: pip install " + " ".join([required_packages[pkg] for pkg in missing_packages]))

# %%
# Imports
import glob
import os
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy import signal

print("All required packages imported successfully!")

# %%
# Utilities: loading and variable discovery

def load_aditya_nc(path_pattern: str) -> xr.Dataset:
    """Load one or multiple .nc files into a single xarray Dataset.

    Parameters
    ----------
    path_pattern : str
        A glob pattern pointing to one or many .nc files (e.g. 'data/*.nc').

    Returns
    -------
    xr.Dataset
        Combined dataset, time-sorted.
    """
    files = sorted(glob.glob(path_pattern))
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {path_pattern}")

    # Try an intelligent open (works in most CF-compliant NetCDFs)
    try:
        ds = xr.open_mfdataset(files, combine='by_coords', parallel=False)
    except Exception:
        # Fallback: concatenate along time
        try:
            ds = xr.open_mfdataset(files, combine='nested', concat_dim='time')
        except Exception:
            # Final fallback: open first file only
            print(f"Warning: Could not combine multiple files. Opening first file only: {files[0]}")
            ds = xr.open_dataset(files[0])

    # Ensure there's a time coordinate and it's sorted
    time_keys = [k for k in ds.coords if 'time' in k.lower()]
    if time_keys:
        time_key = time_keys[0]
        try:
            ds = ds.sortby(time_key)
        except Exception:
            print(f"Warning: Could not sort by time coordinate {time_key}")
            pass
    return ds


def guess_variable_names(ds: xr.Dataset) -> Dict[str, str]:
    """Attempt to guess Bx, By, Bz and time variable names from dataset.

    Returns mapping: {'time': 'time', 'Bx': 'Bx_name', ...}
    """
    names = list(ds.variables.keys())
    lowermap = {n.lower(): n for n in names}  # Fixed: key-value order

    def find_any(key_tokens: List[str]):
        for tok in key_tokens:
            if tok in lowermap:
                return lowermap[tok]
        # Partial match fallback
        for ln, n in lowermap.items():
            for tok in key_tokens:
                if tok in ln:
                    return n
        return None

    mapping = {}
    mapping['time'] = find_any(['time', 'epoch', 'date']) or 'time'

    # Common component names
    mapping['Bx'] = find_any(['bx', 'b_x', 'bx_gse', 'bx_gsm'])
    mapping['By'] = find_any(['by', 'b_y', 'by_gse', 'by_gsm'])
    mapping['Bz'] = find_any(['bz', 'b_z', 'bz_gse', 'bz_gsm'])

    # Try magnitude
    mapping['Bmag'] = find_any(['bmag', 'b_mag', 'bt', 'b_total', 'mag'])

    # If any component missing, try fallback heuristics
    if not mapping['Bx'] or not mapping['By'] or not mapping['Bz']:
        # Check variables with single-letter b and index
        for ln, n in lowermap.items():
            if mapping['Bx'] is None and ('b1' in ln or ln == 'bx'):
                mapping['Bx'] = n
            if mapping['By'] is None and ('b2' in ln or ln == 'by'):
                mapping['By'] = n
            if mapping['Bz'] is None and ('b3' in ln or ln == 'bz'):
                mapping['Bz'] = n

    return mapping


# %%
# Convert dataset to pandas DataFrame (time-indexed) with required columns

def dataset_to_dataframe(ds: xr.Dataset, varmap: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Return a clean DataFrame indexed by time with columns Bx, By, Bz, Bmag.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset loaded from .nc files.
    varmap : dict, optional
        Mapping giving variable names for 'time','Bx','By','Bz','Bmag'. If None the code will try to guess.
    """
    if varmap is None:
        varmap = guess_variable_names(ds)

    # Find variables present
    for k in ['Bx', 'By', 'Bz']:
        if varmap.get(k) is None or varmap[k] not in ds.variables:
            raise KeyError(f"Could not find variable for {k}. Available vars: {list(ds.variables.keys())}")

    time_key = varmap.get('time')
    if time_key not in ds.coords and time_key in ds.variables:
        # promote to coord if necessary
        ds = ds.assign_coords({time_key: ds[time_key]})

    if time_key in ds.coords:
        times = ds.coords[time_key].values
    else:
        # fallback: pick any coordinate that looks like time
        time_candidates = [c for c in ds.coords if 'time' in c.lower()] or list(ds.coords)
        times = ds.coords[time_candidates[0]].values

    try:
        time_index = pd.to_datetime(times)
    except Exception:
        # If already datetime64 it will convert; otherwise, try numeric -> epoch
        try:
            time_index = pd.to_datetime(times, unit='s')
        except Exception:
            time_index = pd.to_datetime(times, errors='coerce')

    # Extract arrays and build DataFrame
    Bx = np.asarray(ds[varmap['Bx']].squeeze())
    By = np.asarray(ds[varmap['By']].squeeze())
    Bz = np.asarray(ds[varmap['Bz']].squeeze())

    # Ensure lengths match time index
    minlen = min(len(time_index), len(Bx), len(By), len(Bz))
    df = pd.DataFrame({'Bx': Bx[:minlen], 'By': By[:minlen], 'Bz': Bz[:minlen]}, index=time_index[:minlen])
    df.index.name = 'time'
    df = df.sort_index()

    # Compute magnitude (if not present)
    if varmap.get('Bmag') and varmap['Bmag'] in ds.variables:
        Bmag = np.asarray(ds[varmap['Bmag']].squeeze())[:minlen]
        df['Bmag'] = Bmag
    else:
        df['Bmag'] = np.sqrt(df['Bx']**2 + df['By']**2 + df['Bz']**2)

    # Remove NaN-only rows and interpolate small gaps
    df = df[~(df[['Bx', 'By', 'Bz']].isna().all(axis=1))]
    df = df.interpolate(method='time', limit=5)

    return df


# %%
# Visualization helpers

def plot_time_series(df: pd.DataFrame, start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None, figsize=(12, 6)):
    """Plot Bx, By, Bz and |B| over a selected interval (or whole dataframe).
    """
    sub = df.loc[start:end] if start or end else df
    if len(sub) == 0:
        print("No data in the specified time range")
        return
        
    fig, ax = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    ax[0].plot(sub.index, sub['Bx'], label='Bx', alpha=0.8)
    ax[0].plot(sub.index, sub['By'], label='By', alpha=0.8)
    ax[0].plot(sub.index, sub['Bz'], label='Bz', alpha=0.8)
    ax[0].legend(loc='best')
    ax[0].set_ylabel('B (nT)')
    ax[0].grid(True, alpha=0.3)
    ax[0].set_title('Magnetic Field Components')

    ax[1].plot(sub.index, sub['Bmag'], label='|B|', color='black')
    ax[1].set_ylabel('|B| (nT)')
    ax[1].set_xlabel('Time')
    ax[1].grid(True, alpha=0.3)
    ax[1].set_title('Magnetic Field Magnitude')
    plt.tight_layout()
    plt.show()


def plot_hodogram(df: pd.DataFrame, start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None, show=True):
    """Plot Bx vs By hodogram (transverse field) for visualization of rotation.
    """
    sub = df.loc[start:end] if start or end else df
    if len(sub) == 0:
        print("No data in the specified time range")
        return
        
    plt.figure(figsize=(6, 6))
    plt.plot(sub['Bx'], sub['By'], '-o', markersize=2, alpha=0.7)
    plt.xlabel('Bx (nT)')
    plt.ylabel('By (nT)')
    plt.title('Hodogram (Bx vs By)')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')  # Equal aspect ratio
    if show:
        plt.show()


# %%
# Detection algorithms

# 1) CME detection (sustained southward Bz)

def detect_cmes(df: pd.DataFrame, bz_col='Bz', threshold_nT=-5.0, min_duration=pd.Timedelta('30min')) -> pd.DataFrame:
    """Detect candidate CME intervals where Bz remains southward below `threshold_nT` for >= min_duration.

    Returns a DataFrame with start, end, duration, min_Bz, mean_Bmag.
    """
    if len(df) == 0:
        return pd.DataFrame()
        
    s = df[bz_col] <= threshold_nT
    s = s.astype(int)
    
    # Identify contiguous True regions using diff
    s_diff = s.diff()
    starts = s_diff[s_diff == 1].index  # transitions from 0 to 1
    ends = s_diff[s_diff == -1].index   # transitions from 1 to 0
    
    # Handle edge cases
    if len(starts) == 0:
        return pd.DataFrame()
    
    # If series starts with True, add the first timestamp as start
    if s.iloc[0] == 1:
        starts = pd.Index([df.index[0]]).union(starts)
    
    # If series ends with True, add the last timestamp as end
    if s.iloc[-1] == 1:
        ends = ends.union(pd.Index([df.index[-1]]))
    
    # Match starts and ends
    min_len = min(len(starts), len(ends))
    starts = starts[:min_len]
    ends = ends[:min_len]

    events = []
    for st, en in zip(starts, ends):
        if en <= st:
            continue
        duration = en - st
        if duration >= min_duration:
            window = df.loc[st:en]
            if len(window) > 0:
                events.append({
                    'start': st,
                    'end': en,
                    'duration': duration,
                    'min_Bz': float(window[bz_col].min()),
                    'mean_Bmag': float(window['Bmag'].mean()),
                    'n_points': int(len(window))
                })
    return pd.DataFrame(events)


# 2) Shock detection (sudden jumps in |B|)

def detect_shocks(df: pd.DataFrame, mag_col='Bmag', z_thresh=5.0, min_sep_seconds=30) -> pd.DataFrame:
    """Detect shocks using peaks in the time derivative of |B| (z-score based).

    Parameters
    ----------
    df : DataFrame
    z_thresh : float
        Peaks where derivative z-score exceeds z_thresh are marked as shocks.
    min_sep_seconds : int
        Minimum separation between detected peaks.
    """
    if len(df) < 3:
        return pd.DataFrame()
        
    # Compute time derivative (nT/sec)
    dt = df.index.to_series().diff().dt.total_seconds()
    dt = dt.fillna(method='bfill') if hasattr(dt, 'fillna') else dt.bfill()  # Updated for newer pandas
    db = df[mag_col].diff()
    db = db.fillna(0) if hasattr(db, 'fillna') else db.bfill()  # Updated for newer pandas
    db_dt = db / (dt + 1e-12)  # Avoid division by zero

    # Remove infinite and NaN values
    db_dt = db_dt.replace([np.inf, -np.inf], np.nan).fillna(0)

    # z-score normalization
    db_dt_mean = np.nanmean(db_dt)
    db_dt_std = np.nanstd(db_dt)
    if db_dt_std < 1e-12:
        return pd.DataFrame()  # No variation in data
        
    dbz = (db_dt - db_dt_mean) / db_dt_std

    # Find peaks in positive derivative (sudden increase)
    median_dt = np.nanmedian(dt)
    if np.isnan(median_dt) or median_dt == 0:
        distance = 1
    else:
        distance = max(1, int(min_sep_seconds / median_dt))
    
    try:
        peaks, props = signal.find_peaks(dbz, height=z_thresh, distance=distance)
    except Exception as e:
        print(f"Warning: Peak finding failed: {e}")
        return pd.DataFrame()

    events = []
    for p in peaks:
        if p >= len(df):
            continue
        t = df.index[p]
        # refine window around peak to get pre/post means
        pre_start = max(0, p - 10)
        post_end = min(len(df), p + 11)  # +11 because end is exclusive
        
        pre_vals = df[mag_col].iloc[pre_start:p]
        post_vals = df[mag_col].iloc[p:post_end]
        
        pre_mean = pre_vals.mean() if len(pre_vals) > 0 else df[mag_col].iloc[p]
        post_mean = post_vals.mean() if len(post_vals) > 0 else df[mag_col].iloc[p]
        jump = float(post_mean - pre_mean)
        
        events.append({
            'time': t, 
            'index': int(p), 
            'db_dt_nT_per_s': float(db_dt.iloc[p]), 
            'zscore': float(dbz[p]), 
            'jump_nT': jump
        })

    return pd.DataFrame(events)


# 3) Magnetic-cloud detection (low |B| variance + smooth rotation)

def detect_magnetic_clouds(
    df: pd.DataFrame,
    window_hours: Tuple[float, float] = (6.0, 48.0),
    step_minutes: float = 10.0,
    std_thresh_nT: float = 5.0,
    rotation_deg_thresh: float = 90.0,
) -> pd.DataFrame:
    """Scan with sliding windows to find candidate magnetic clouds.

    Heuristics:
    - Window lengths between window_hours (min,max) hours
    - Standard deviation of |B| less than std_thresh_nT
    - Unwrapped rotation angle of transverse field (Bx,By) exceeds rotation_deg_thresh
    """
    if len(df) < 10:
        return pd.DataFrame()
        
    events = []
    min_window = pd.Timedelta(hours=window_hours[0])
    max_window = pd.Timedelta(hours=window_hours[1])
    step = pd.Timedelta(minutes=step_minutes)

    t0 = df.index[0]
    t_end = df.index[-1]
    
    # We'll try multiple window sizes (coarse search)
    window = min_window
    while window <= max_window:
        t = t0
        while t + window <= t_end:
            win = df.loc[t:t+window]
            if len(win) < 5:
                t += step
                continue
                
            std_B = win['Bmag'].std()
            if std_B <= std_thresh_nT:
                # Check smooth rotation in transverse plane
                try:
                    # Handle zero values that could cause issues with angle calculation
                    bx_vals = win['Bx'].values
                    by_vals = win['By'].values
                    
                    # Only calculate if we have non-zero transverse field
                    transverse_mag = np.sqrt(bx_vals**2 + by_vals**2)
                    if np.all(transverse_mag < 1e-6):
                        t += step
                        continue
                    
                    angles = np.unwrap(np.angle(bx_vals + 1j * by_vals))
                    rotation_rad = angles[-1] - angles[0]
                    rotation_deg = np.degrees(np.abs(rotation_rad))
                    
                    # Smoothness check: how well single-tone rotation fits
                    # We'll measure linearity of angle vs time (R^2)
                    if len(angles) > 1:
                        x = np.arange(len(angles))
                        A = np.vstack([x, np.ones(len(x))]).T
                        try:
                            m, c = np.linalg.lstsq(A, angles, rcond=None)[0]
                            residuals = angles - (m * x + c)
                            angle_var = np.var(angles)
                            if angle_var > 1e-12:
                                smoothness = 1 - (np.var(residuals) / angle_var)
                            else:
                                smoothness = 0
                        except:
                            smoothness = 0
                    else:
                        smoothness = 0
                    
                    if rotation_deg >= rotation_deg_thresh and smoothness > 0.5:
                        events.append({
                            'start': t,
                            'end': t + window,
                            'duration': window,
                            'std_Bmag': float(std_B),
                            'rotation_deg': float(rotation_deg),
                            'smoothness': float(smoothness),
                            'n_points': int(len(win)),
                        })
                except Exception as e:
                    print(f"Warning: Angle calculation failed: {e}")
                    pass
                    
            t += step
        window += pd.Timedelta(hours=6)  # increase window in 6-hour steps

    # Merge overlapping windows (coalescing)
    if not events:
        return pd.DataFrame()

    edf = pd.DataFrame(events).sort_values('start').reset_index(drop=True)
    merged = []
    cur = edf.loc[0].to_dict()
    
    for i in range(1, len(edf)):
        row = edf.loc[i].to_dict()
        if row['start'] <= cur['end']:
            # merge
            cur['end'] = max(cur['end'], row['end'])
            cur['duration'] = cur['end'] - cur['start']
            cur['std_Bmag'] = min(cur['std_Bmag'], row['std_Bmag'])
            cur['rotation_deg'] = max(cur['rotation_deg'], row['rotation_deg'])
            cur['smoothness'] = max(cur['smoothness'], row['smoothness'])
            cur['n_points'] = max(cur['n_points'], row['n_points'])  # Take max instead of sum
        else:
            merged.append(cur)
            cur = row
    merged.append(cur)
    return pd.DataFrame(merged)


# %%
# Example: Putting it all together

def run_analysis(path_glob: str):
    """Main analysis function that can be called with your data path."""
    
    # 1) Load dataset
    print('\nLoading dataset...')
    try:
        ds = load_aditya_nc(path_glob)
        print('Dataset loaded successfully.')
        print('Variables found:', list(ds.variables.keys()))
        print('Dataset dimensions:', dict(ds.dims))
    except Exception as e:
        print(f'Error loading dataset: {e}')
        return None

    # 2) Convert to DataFrame
    print('\nConverting to DataFrame...')
    try:
        varmap = guess_variable_names(ds)
        print('Guessed variable mapping:', varmap)
        
        # Check if essential variables were found
        missing_vars = [k for k in ['Bx', 'By', 'Bz'] if not varmap.get(k)]
        if missing_vars:
            print(f"Warning: Could not find variables for {missing_vars}")
            print("Available variables:", list(ds.variables.keys()))
            return None
            
        df = dataset_to_dataframe(ds, varmap=varmap)
        print(f'DataFrame ready. Rows: {len(df)}')
        print(f'Time range: {df.index.min()} to {df.index.max()}')
        print('Data columns:', df.columns.tolist())
        print('\nData summary:')
        print(df.describe())
        
    except Exception as e:
        print(f'Error converting to DataFrame: {e}')
        return None

    # 3) Quick plot (first few hours or full dataset if small)
    print('\nGenerating time series plot...')
    try:
        if len(df) > 1000:
            # Plot first 6 hours if dataset is large
            end_time = df.index[0] + pd.Timedelta(hours=6)
            plot_time_series(df, start=df.index[0], end=end_time)
        else:
            plot_time_series(df)
    except Exception as e:
        print(f'Plotting failed: {e}')

    # 4) Detect CMEs (sustained southward Bz)
    print('\nDetecting CME candidates...')
    try:
        cmes = detect_cmes(df, bz_col='Bz', threshold_nT=-10.0, min_duration=pd.Timedelta('20min'))
        print(f'Found {len(cmes)} CME candidate(s)')
        if not cmes.empty:
            print('\nCME candidates:')
            print(cmes)
    except Exception as e:
        print(f'CME detection failed: {e}')
        cmes = pd.DataFrame()

    # 5) Detect Shocks
    print('\nDetecting shocks...')
    try:
        shocks = detect_shocks(df, mag_col='Bmag', z_thresh=5.0, min_sep_seconds=30)
        print(f'Found {len(shocks)} shock candidate(s)')
        if not shocks.empty:
            print('\nShock candidates:')
            print(shocks)
    except Exception as e:
        print(f'Shock detection failed: {e}')
        shocks = pd.DataFrame()

    # 6) Detect Magnetic Clouds
    print('\nDetecting magnetic-cloud candidates...')
    try:
        mclouds = detect_magnetic_clouds(
            df, 
            window_hours=(6.0, 48.0), 
            step_minutes=15.0, 
            std_thresh_nT=5.0, 
            rotation_deg_thresh=90.0
        )
        print(f'Found {len(mclouds)} magnetic-cloud candidate(s)')
        if not mclouds.empty:
            print('\nMagnetic cloud candidates:')
            print(mclouds)
    except Exception as e:
        print(f'Magnetic cloud detection failed: {e}')
        mclouds = pd.DataFrame()

    # 7) Build a simple catalog and save
    print('\nBuilding event catalog...')
    try:
        catalog_entries = []
        
        # Add CME events
        for _, r in cmes.iterrows():
            catalog_entries.append({
                'type': 'CME', 
                'start': r['start'], 
                'end': r['end'], 
                'duration': r['duration'].total_seconds() / 3600,  # Convert to hours
                'min_Bz': r['min_Bz'], 
                'mean_Bmag': r['mean_Bmag'],
                'n_points': r['n_points']
            })
            
        # Add Shock events
        for _, r in shocks.iterrows():
            catalog_entries.append({
                'type': 'Shock', 
                'start': r['time'], 
                'end': r['time'], 
                'duration': 0,
                'db_dt': r['db_dt_nT_per_s'], 
                'jump': r['jump_nT'],
                'zscore': r['zscore']
            })
            
        # Add Magnetic Cloud events
        for _, r in mclouds.iterrows():
            catalog_entries.append({
                'type': 'MagCloud', 
                'start': r['start'], 
                'end': r['end'], 
                'duration': r['duration'].total_seconds() / 3600,  # Convert to hours
                'rotation_deg': r['rotation_deg'], 
                'std_Bmag': r['std_Bmag'],
                'smoothness': r['smoothness'],
                'n_points': r['n_points']
            })

        if catalog_entries:
            catalog_df = pd.DataFrame(catalog_entries)
            out_csv = 'adityal1_mag_event_catalog.csv'
            catalog_df.to_csv(out_csv, index=False)
            print(f'Catalog saved to {out_csv}')
            print(f'Total events: {len(catalog_df)}')
            print('\nEvent summary:')
            print(catalog_df['type'].value_counts())
        else:
            print('No events found; catalog is empty.')
            
    except Exception as e:
        print(f'Catalog creation failed: {e}')

    print('\nAnalysis completed successfully!')
    return df

# %%
# Usage example - modify the path to point to your data files
if __name__ == '__main__':
    # Example usage: edit the path glob below to match your files
    # Examples:
    # path_glob = 'data/*.nc'
    # path_glob = '/path/to/aditya_l1_data/*.nc'
    # path_glob = 'aditya_l1_mag_*.nc'
    
    # COMMON PATH PATTERNS - uncomment the one that matches your setup:
    # path_glob = 'E:\\MAG\\*.nc'                    # All .nc files in MAG folder
    # path_glob = 'E:\\MAG\\**\\*.nc'                # All .nc files recursively
    # path_glob = 'E:\\MAG\\aditya*.nc'              # Files starting with 'aditya'
    # path_glob = 'E:\\MAG\\mag_data_*.nc'           # Files with specific pattern
    
    path_glob = 'E:\\MAG\\*.nc'  # UPDATE THIS PATH!
    
    print("Aditya-L1 Magnetometer Data Analysis")
    print("=" * 40)
    
    # Diagnostic: check what files exist
    print(f"Looking for files with pattern: {path_glob}")
    files = glob.glob(path_glob, recursive='**' in path_glob)
    
    if not files:
        print(f"No files found for pattern: {path_glob}")
        print("\nDiagnostic: Checking directory contents...")
        
        # Extract directory from pattern
        import os
        base_dir = os.path.dirname(path_glob.replace('*', '').replace('\\*.nc', ''))
        if os.path.exists(base_dir):
            print(f"Contents of {base_dir}:")
            try:
                for item in os.listdir(base_dir)[:10]:  # Show first 10 items
                    item_path = os.path.join(base_dir, item)
                    item_type = "DIR" if os.path.isdir(item_path) else "FILE"
                    print(f"  {item_type}: {item}")
                
                # Look for any .nc files
                all_nc = glob.glob(os.path.join(base_dir, '**', '*.nc'), recursive=True)
                if all_nc:
                    print(f"\nFound {len(all_nc)} .nc files in subdirectories:")
                    for f in all_nc[:3]:
                        print(f"  {f}")
                    if len(all_nc) > 3:
                        print(f"  ... and {len(all_nc) - 3} more")
                        
                    # Suggest correct pattern
                    if all_nc:
                        suggested_pattern = os.path.join(base_dir, '**', '*.nc')
                        print(f"\nSuggested pattern: {suggested_pattern}")
                else:
                    print("No .nc files found in directory or subdirectories")
                    
            except PermissionError:
                print(f"Permission denied accessing {base_dir}")
        else:
            print(f"Directory {base_dir} does not exist!")
            
        print("\nPlease update the path_glob variable to point to your .nc files")
        
    else:
        print(f"Found {len(files)} files matching pattern:")
        for f in files[:3]:  # Show first 3 files
            print(f"  {f}")
        if len(files) > 3:
            print(f"  ... and {len(files) - 3} more")
            
        df = run_analysis(path_glob)
