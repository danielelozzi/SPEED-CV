# speed_script_events.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import traceback
import pickle
from scipy.signal import welch, spectrogram
from scipy.stats import gaussian_kde

# --- Constants ---
SAMPLING_FREQ = 200  # Hz
NS_TO_S = 1e9

# ==============================================================================
# HELPER AND DATA LOADING FUNCTIONS
# ==============================================================================

def euclidean_distance(x1, y1, x2, y2):
    """Calculates the Euclidean distance between two points."""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def load_all_data(data_dir: Path, un_enriched_mode: bool):
    """Loads all necessary CSV files from the data directory."""
    files_to_load = {
        'events': 'events.csv', 'fixations_not_enr': 'fixations.csv', 'gaze_not_enr': 'gaze.csv',
        'pupil': '3d_eye_states.csv', 'blinks': 'blinks.csv', 'saccades': 'saccades.csv'
    }
    if not un_enriched_mode:
        files_to_load.update({'gaze_enr': 'gaze_enriched.csv', 'fixations_enr': 'fixations_enriched.csv'})

    dataframes = {}
    for name, filename in files_to_load.items():
        try:
            dataframes[name] = pd.read_csv(data_dir / filename)
        except FileNotFoundError:
            if name in ['gaze_enr', 'fixations_enr']:
                 dataframes[name] = pd.DataFrame()
            else:
                raise FileNotFoundError(f"Required data file not found: {filename}")
    return dataframes

def get_timestamp_col(df):
    """Gets the correct timestamp column from a dataframe."""
    for col in ['start timestamp [ns]', 'timestamp [ns]']:
        if col in df.columns:
            return col
    return None

def filter_data_by_segment(all_data, start_ts, end_ts, rec_id):
    """Filters all dataframes for a specific time segment [start_ts, end_ts)."""
    segment_data = {}
    for name, df in all_data.items():
        if df.empty or name == 'events':
            segment_data[name] = df
            continue
        ts_col = get_timestamp_col(df)
        if ts_col:
            mask = (df[ts_col] >= start_ts) & (df[ts_col] < end_ts)
            if 'recording id' in df.columns:
                mask &= (df['recording id'] == rec_id)
            segment_data[name] = df[mask].copy().reset_index(drop=True)
        else:
            segment_data[name] = pd.DataFrame(columns=df.columns)
    return segment_data

def get_video_dimensions(video_path: Path):
    """Gets width and height from a video file."""
    if not video_path.exists():
        print(f"WARNING: Video file not found at {video_path}. Cannot get dimensions.")
        return None, None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"WARNING: Could not open video file {video_path}.")
        return None, None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height

# ==============================================================================
# CORE DATA ANALYSIS FUNCTIONS
# ==============================================================================

def process_gaze_movements(gaze_df, un_enriched_mode: bool):
    """Identifies and processes gaze movements from ENRICHED gaze data."""
    if un_enriched_mode or gaze_df.empty or 'fixation id' not in gaze_df.columns or 'gaze detected on surface' not in gaze_df.columns:
        return pd.DataFrame()
    
    gaze_df['fixation id'] = gaze_df['fixation id'].fillna(-1)
    gaze_on_surface = gaze_df[gaze_df['gaze detected on surface'] == True].copy()
    if gaze_on_surface.empty:
        return pd.DataFrame()
    
    is_movement = gaze_on_surface['fixation id'] == -1
    gaze_on_surface.loc[:, 'movement_id'] = (is_movement != is_movement.shift()).cumsum()[is_movement]
    
    movements = []
    for _, group in gaze_on_surface.dropna(subset=['movement_id']).groupby('movement_id'):
        if len(group) < 2:
            continue
        start_row, end_row = group.iloc[0], group.iloc[-1]
        x, y = group['gaze position on surface x [normalized]'], group['gaze position on surface y [normalized]']
        movements.append({
            'duration_ns': end_row['timestamp [ns]'] - start_row['timestamp [ns]'],
            'total_displacement': euclidean_distance(x.shift(), y.shift(), x, y).sum(),
            'effective_displacement': euclidean_distance(x.iloc[0], y.iloc[0], x.iloc[-1], y.iloc[-1])
        })
    return pd.DataFrame(movements)

def calculate_summary_features(data, movements_df, subj_name, event_name, un_enriched_mode: bool, video_width: int, video_height: int):
    """Calculates a dictionary of summary features for a segment."""
    pupil, blinks, saccades = data.get('pupil', pd.DataFrame()), data.get('blinks', pd.DataFrame()), data.get('saccades', pd.DataFrame())
    gaze_enr, gaze_not_enr = data.get('gaze_enr', pd.DataFrame()), data.get('gaze_not_enr', pd.DataFrame())
    fixations_enr, fixations_not_enr = data.get('fixations_enr', pd.DataFrame()), data.get('fixations_not_enr', pd.DataFrame())

    results = {'participant': subj_name, 'event': event_name}

    # --- Fixation Features ---
    fixations_to_analyze = fixations_not_enr
    if not un_enriched_mode and not fixations_enr.empty:
        fixations_to_analyze = fixations_enr[fixations_enr['fixation detected on surface'] == True]
    
    if not fixations_to_analyze.empty:
        results.update({
            'n_fixation': fixations_to_analyze['fixation id'].nunique(),
            'fixation_avg_duration_ms': fixations_to_analyze['duration [ms]'].mean(),
            'fixation_std_duration_ms': fixations_to_analyze['duration [ms]'].std()
        })
        
        x_coords, y_coords = pd.Series(dtype='float64'), pd.Series(dtype='float64')
        if 'fixation x [normalized]' in fixations_to_analyze.columns and not un_enriched_mode:
            x_coords, y_coords = fixations_to_analyze['fixation x [normalized]'], fixations_to_analyze['fixation y [normalized]']
        elif 'fixation x [px]' in fixations_to_analyze.columns and video_width and video_height:
            x_coords = fixations_to_analyze['fixation x [px]'] / video_width
            y_coords = fixations_to_analyze['fixation y [px]'] / video_height
        
        if not x_coords.empty:
            results.update({
                'fixation_avg_x': x_coords.mean(), 'fixation_std_x': x_coords.std(),
                'fixation_avg_y': y_coords.mean(), 'fixation_std_y': y_coords.std()
            })

    # --- Other Features ---
    if not blinks.empty:
        results.update({'n_blink': len(blinks), 'blink_avg_duration_ms': blinks['duration [ms]'].mean()})

    if not pupil.empty and 'pupil diameter left [mm]' in pupil.columns:
        pupil_diam = pupil['pupil diameter left [mm]'].dropna()
        if not pupil_diam.empty:
            results.update({'pupil_avg_mm': pupil_diam.mean(), 'pupil_std_mm': pupil_diam.std()})

    return results

def run_analysis(subj_name: str, data_dir_str: str, output_dir_str: str, un_enriched_mode: bool):
    """
    Performs data analysis, calculates stats, and saves processed data for later use.
    """
    pd.options.mode.chained_assignment = None
    data_dir, output_dir = Path(data_dir_str), Path(output_dir_str)
    processed_data_dir = output_dir / 'processed_data'
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    video_width, video_height = get_video_dimensions(data_dir / 'external.mp4')
    all_data = load_all_data(data_dir, un_enriched_mode)
    events_df = all_data.get('events')
    if events_df is None or events_df.empty:
        raise ValueError("events.csv not found or is empty.")
        
    all_results = []
    print(f"Found {len(events_df)} events, processing {len(events_df) - 1} segments.")
    for i in range(len(events_df) - 1):
        event_row = events_df.iloc[i]
        start_ts, end_ts = event_row['timestamp [ns]'], events_df.iloc[i+1]['timestamp [ns]']
        event_name = event_row.get('name', f"segment_{i}")
        rec_id = event_row['recording id']
        
        print(f"--- Analyzing segment for event: '{event_name}' ---")
        segment_data = filter_data_by_segment(all_data, start_ts, end_ts, rec_id)
        
        if all(df.empty for name, df in segment_data.items() if name != 'events'):
            print(f"  -> Skipping segment '{event_name}' due to no data in the time range.")
            continue
        
        movements_df = process_gaze_movements(segment_data.get('gaze_enr', pd.DataFrame()), un_enriched_mode)
        event_results = calculate_summary_features(segment_data, movements_df, subj_name, event_name, un_enriched_mode, video_width, video_height)
        all_results.append(event_results)

        # Save processed data for this segment for on-demand plotting
        segment_output_path = processed_data_dir / f"segment_{i}_{event_name}.pkl"
        with open(segment_output_path, 'wb') as f:
            pickle.dump(segment_data, f)
        print(f"  -> Saved processed data to {segment_output_path}")

    # Save the aggregated summary results dataframe
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_dir / f'summary_results_{subj_name}.csv', index=False)
        print("\nAggregated summary results saved.")
    else:
        print("\nNo results were generated from the analysis.")

# ==============================================================================
# ON-DEMAND PLOT GENERATION FUNCTIONS
# ==============================================================================

def _plot_histogram(data_series, title, xlabel, output_path):
    """Helper function to create and save a standardized histogram."""
    if data_series.dropna().empty: return
    plt.figure(figsize=(10, 6))
    plt.hist(data_series.dropna(), bins=25, color='royalblue', edgecolor='black', alpha=0.7)
    plt.title(title, fontsize=15); plt.xlabel(xlabel, fontsize=12); plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout(); plt.savefig(output_path); plt.close()

def _plot_path(df, x_col, y_col, title, output_path, is_normalized, color):
    """Helper function to create and save a path plot."""
    if df.empty or x_col not in df.columns or y_col not in df.columns: return
    plt.figure(figsize=(10, 8))
    plt.plot(df[x_col], df[y_col], marker='o', linestyle='-', color=color, markersize=4, alpha=0.6)
    plt.title(title, fontsize=15)
    if is_normalized:
        plt.xlabel('Normalized X'); plt.ylabel('Normalized Y'); plt.xlim(0, 1); plt.ylim(1, 0) # Flipped Y for screen coordinates
    else:
        plt.xlabel('Pixel X'); plt.ylabel('Pixel Y')
    plt.grid(True); plt.tight_layout(); plt.savefig(output_path); plt.close()

def _plot_heatmap(df, x_col, y_col, title, output_path, is_normalized):
    """Helper function to create and save a density heatmap."""
    if df.empty or x_col not in df.columns or y_col not in df.columns: return
    x = df[x_col].dropna(); y = df[y_col].dropna()
    if len(x) < 3: return
    
    try:
        k = gaussian_kde(np.vstack([x, y]))
        xi, yi = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        
        plt.figure(figsize=(10, 8))
        plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap='Reds')
        plt.title(title, fontsize=15)
        if is_normalized:
            plt.xlabel('Normalized X'); plt.ylabel('Normalized Y'); plt.xlim(0, 1); plt.ylim(1, 0)
        else:
            plt.xlabel('Pixel X'); plt.ylabel('Pixel Y')
        plt.grid(True); plt.tight_layout(); plt.savefig(output_path); plt.close()
    except np.linalg.LinAlgError:
        print(f"WARNING: Could not generate heatmap for '{title}' (singular matrix).")
    except Exception as e:
        print(f"WARNING: Unexpected error during heatmap generation for '{title}': {e}")

def _plot_spectral_analysis(pupil_series, title_prefix, output_dir):
    """Generates and saves periodogram and spectrogram for a pupil diameter series."""
    if pupil_series.dropna().empty or len(pupil_series.dropna()) <= SAMPLING_FREQ:
        print(f"Skipping spectral analysis for '{title_prefix}': not enough data.")
        return
    
    ts = pupil_series.dropna().to_numpy()
    
    # Periodogram
    freqs, Pxx = welch(ts, fs=SAMPLING_FREQ, nperseg=min(len(ts), 100))
    plt.figure(figsize=(10, 5))
    plt.semilogy(freqs, Pxx)
    plt.title(f'Periodogram - {title_prefix}'); plt.xlabel('Frequency [Hz]'); plt.ylabel('Power Spectral Density')
    plt.grid(True); plt.savefig(output_dir / f'periodogram_{title_prefix}.pdf'); plt.close()

    # Spectrogram
    f, t, Sxx = spectrogram(ts, fs=SAMPLING_FREQ, nperseg=min(len(ts), 256), noverlap=min(len(ts)//2, 50))
    plt.figure(figsize=(10, 5))
    plt.pcolormesh(t, f, 10 * np.log10(np.maximum(Sxx, 1e-10)), shading='gouraud')
    plt.title(f'Spectrogram - {title_prefix}'); plt.ylabel('Frequency [Hz]'); plt.xlabel('Time [s]')
    plt.colorbar(label='Power [dB]'); plt.savefig(output_dir / f'spectrogram_{title_prefix}.pdf'); plt.close()


def generate_plots_on_demand(output_dir_str: str, subj_name: str, plot_selections: dict, un_enriched_mode: bool):
    """
    Generates plots based on pre-processed data and user selections.
    """
    output_dir = Path(output_dir_str)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    processed_data_dir = output_dir / 'processed_data'

    if not processed_data_dir.exists():
        raise FileNotFoundError("Processed data directory not found. Please run the Core Analysis first.")

    video_width, video_height = get_video_dimensions(output_dir / 'eyetracking_file' / 'external.mp4')

    for pkl_file in sorted(processed_data_dir.glob("*.pkl")):
        event_name = "_".join(pkl_file.stem.split('_')[2:])
        print(f"--- Generating plots for event: '{event_name}' ---")
        
        with open(pkl_file, 'rb') as f:
            segment_data = pickle.load(f)

        # Extract dataframes
        fixations_enr = segment_data.get('fixations_enr', pd.DataFrame())
        fixations_not_enr = segment_data.get('fixations_not_enr', pd.DataFrame())
        gaze_enr = segment_data.get('gaze_enr', pd.DataFrame())
        gaze_not_enr = segment_data.get('gaze_not_enr', pd.DataFrame())
        blinks = segment_data.get('blinks', pd.DataFrame())
        saccades = segment_data.get('saccades', pd.DataFrame())
        pupil = segment_data.get('pupil', pd.DataFrame())

        # Generate selected plots
        if plot_selections.get("histograms"):
            _plot_histogram(fixations_not_enr['duration [ms]'], f"Fixation Duration (Un-enriched) - {event_name}", "Duration [ms]", plots_dir / f"hist_fix_unenriched_{event_name}.pdf")
            if not un_enriched_mode:
                _plot_histogram(fixations_enr['duration [ms]'], f"Fixation Duration (Enriched) - {event_name}", "Duration [ms]", plots_dir / f"hist_fix_enriched_{event_name}.pdf")
            _plot_histogram(blinks['duration [ms]'], f"Blink Duration - {event_name}", "Duration [ms]", plots_dir / f"hist_blinks_{event_name}.pdf")
            _plot_histogram(saccades['duration [ms]'], f"Saccade Duration - {event_name}", "Duration [ms]", plots_dir / f"hist_saccades_{event_name}.pdf")

        if plot_selections.get("path_plots"):
            _plot_path(fixations_not_enr, 'fixation x [px]', 'fixation y [px]', f"Fixation Path (Un-enriched) - {event_name}", plots_dir / f"path_fix_unenriched_{event_name}.pdf", False, 'purple')
            _plot_path(gaze_not_enr, 'gaze x [px]', 'gaze y [px]', f"Gaze Path (Un-enriched) - {event_name}", plots_dir / f"path_gaze_unenriched_{event_name}.pdf", False, 'blue')
            if not un_enriched_mode:
                _plot_path(fixations_enr, 'fixation x [normalized]', 'fixation y [normalized]', f"Fixation Path (Enriched) - {event_name}", plots_dir / f"path_fix_enriched_{event_name}.pdf", True, 'green')
                _plot_path(gaze_enr, 'gaze position on surface x [normalized]', 'gaze position on surface y [normalized]', f"Gaze Path (Enriched) - {event_name}", plots_dir / f"path_gaze_enriched_{event_name}.pdf", True, 'red')

        if plot_selections.get("heatmaps"):
            if video_width and video_height:
                # Normalize un-enriched data for consistent heatmap scaling
                fix_not_enr_norm = fixations_not_enr.copy(); fix_not_enr_norm['x_norm'] = fix_not_enr_norm['fixation x [px]'] / video_width; fix_not_enr_norm['y_norm'] = fix_not_enr_norm['fixation y [px]'] / video_height
                gaze_not_enr_norm = gaze_not_enr.copy(); gaze_not_enr_norm['x_norm'] = gaze_not_enr_norm['gaze x [px]'] / video_width; gaze_not_enr_norm['y_norm'] = gaze_not_enr_norm['gaze y [px]'] / video_height
                _plot_heatmap(fix_not_enr_norm, 'x_norm', 'y_norm', f"Fixation Heatmap (Un-enriched) - {event_name}", plots_dir / f"heatmap_fix_unenriched_{event_name}.pdf", True)
                _plot_heatmap(gaze_not_enr_norm, 'x_norm', 'y_norm', f"Gaze Heatmap (Un-enriched) - {event_name}", plots_dir / f"heatmap_gaze_unenriched_{event_name}.pdf", True)
            if not un_enriched_mode:
                _plot_heatmap(fixations_enr, 'fixation x [normalized]', 'fixation y [normalized]', f"Fixation Heatmap (Enriched) - {event_name}", plots_dir / f"heatmap_fix_enriched_{event_name}.pdf", True)
                _plot_heatmap(gaze_enr, 'gaze position on surface x [normalized]', 'gaze position on surface y [normalized]', f"Gaze Heatmap (Enriched) - {event_name}", plots_dir / f"heatmap_gaze_enriched_{event_name}.pdf", True)

        if plot_selections.get("pupillometry") and not pupil.empty:
            # Time Series Plot
            plt.figure(figsize=(12, 6))
            plt.plot(pupil['timestamp [ns]'] / NS_TO_S, pupil['pupil diameter left [mm]'], label='Left Pupil', alpha=0.8)
            plt.plot(pupil['timestamp [ns]'] / NS_TO_S, pupil['pupil diameter right [mm]'], label='Right Pupil', alpha=0.8)
            plt.title(f"Pupil Diameter Time Series - {event_name}", fontsize=15)
            plt.xlabel("Time [s]"); plt.ylabel("Pupil Diameter [mm]"); plt.legend(); plt.grid(True)
            plt.tight_layout(); plt.savefig(plots_dir / f"pupillometry_{event_name}.pdf"); plt.close()
            
            # Spectral Analysis (Total)
            _plot_spectral_analysis(pupil['pupil diameter left [mm]'], f"total_{event_name}", plots_dir)
            
            # Spectral Analysis (On Surface Only)
            if not un_enriched_mode and not gaze_enr.empty:
                merged_pupil_gaze = pd.merge(pupil, gaze_enr[['timestamp [ns]', 'gaze detected on surface']], on='timestamp [ns]', how='inner')
                pupil_on_surface = merged_pupil_gaze[merged_pupil_gaze['gaze detected on surface'] == True]
                _plot_spectral_analysis(pupil_on_surface['pupil diameter left [mm]'], f"onsurface_{event_name}", plots_dir)

    print("--- Plot generation finished. ---")
