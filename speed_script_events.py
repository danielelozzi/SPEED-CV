import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import traceback
import os
from scipy.signal import welch, spectrogram
from scipy.stats import gaussian_kde

# --- Constants ---
SAMPLING_FREQ = 200  # Hz
NS_TO_S = 1e9

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
        files_to_load.update({'gaze': 'gaze_enriched.csv', 'fixations_enr': 'fixations_enriched.csv'})

    dataframes = {}
    for name, filename in files_to_load.items():
        try:
            dataframes[name] = pd.read_csv(data_dir / filename)
        except FileNotFoundError:
            # If the file is optional or depends on the mode, create an empty DF
            if name in ['gaze', 'fixations_enr'] and un_enriched_mode:
                print(f"Info: File {filename} not loaded as per 'un-enriched' mode.")
                dataframes[name] = pd.DataFrame()
            elif name in ['gaze', 'fixations_enr']:
                 print(f"Info: Optional file {filename} not found, proceeding without it.")
                 dataframes[name] = pd.DataFrame()
            else:
                # If the file is essential, raise an error
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

def process_gaze_movements(gaze_df, un_enriched_mode: bool):
    """Identifies and processes gaze movements from ENRICHED gaze data."""
    if un_enriched_mode or gaze_df.empty or 'fixation id' not in gaze_df.columns or 'gaze detected on surface' not in gaze_df.columns:
        return pd.DataFrame()
    
    gaze_df['fixation id'] = gaze_df['fixation id'].fillna(-1)
    gaze_on_surface = gaze_df[gaze_df['gaze detected on surface'] == True].copy()
    if gaze_on_surface.empty:
        return pd.DataFrame()
    
    is_movement = gaze_on_surface['fixation id'] == -1
    gaze_on_surface.loc[is_movement, 'movement_id'] = (is_movement != is_movement.shift()).cumsum()[is_movement]
    
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
    """Calculates a dictionary of summary features, including normalization from pixels."""
    pupil, blinks, saccades = data.get('pupil', pd.DataFrame()), data.get('blinks', pd.DataFrame()), data.get('saccades', pd.DataFrame())
    gaze_enr, gaze_not_enr = data.get('gaze', pd.DataFrame()), data.get('gaze_not_enr', pd.DataFrame())
    fixations_enr, fixations_not_enr = data.get('fixations_enr', pd.DataFrame()), data.get('fixations_not_enr', pd.DataFrame())

    results = {
        'participant': subj_name, 'event': event_name, 'n_fixation': np.nan, 'fixation_avg_duration_ms': np.nan,
        'fixation_std_duration_ms': np.nan, 'fixation_avg_x': np.nan, 'fixation_std_x': np.nan,
        'fixation_avg_y': np.nan, 'fixation_std_y': np.nan, 'n_blink': np.nan, 'blink_avg_duration_ms': np.nan,
        'blink_std_duration_ms': np.nan, 'pupil_start_mm': np.nan, 'pupil_end_mm': np.nan, 'pupil_avg_mm': np.nan,
        'pupil_std_mm': np.nan, 'n_movements': np.nan, 'sum_time_movement_s': np.nan, 'avg_time_movement_s': np.nan,
        'std_time_movement_s': np.nan, 'total_disp_sum': np.nan, 'total_disp_avg': np.nan, 'total_disp_std': np.nan,
        'effective_disp_sum': np.nan, 'effective_disp_avg': np.nan, 'effective_disp_std': np.nan,
        'n_gaze_per_fixation_avg': np.nan
    }

    # --- Fixation Features ---
    fixations_to_analyze = fixations_not_enr if not fixations_not_enr.empty else pd.DataFrame()
    if not un_enriched_mode and not fixations_enr.empty and 'fixation detected on surface' in fixations_enr.columns:
        enriched_on_surface = fixations_enr[fixations_enr['fixation detected on surface'] == True].copy()
        if not enriched_on_surface.empty:
            fixations_to_analyze = enriched_on_surface
    
    if not fixations_to_analyze.empty:
        results.update({'n_fixation': fixations_to_analyze['fixation id'].nunique(), 'fixation_avg_duration_ms': fixations_to_analyze['duration [ms]'].mean(), 'fixation_std_duration_ms': fixations_to_analyze['duration [ms]'].std()})
        
        x_coords, y_coords = pd.Series(dtype='float64'), pd.Series(dtype='float64')
        if 'fixation x [normalized]' in fixations_to_analyze.columns and not un_enriched_mode:
            x_coords, y_coords = fixations_to_analyze['fixation x [normalized]'], fixations_to_analyze['fixation y [normalized]']
        elif 'fixation x [px]' in fixations_to_analyze.columns:
            if video_width and video_height and video_width > 0 and video_height > 0:
                x_coords = fixations_to_analyze['fixation x [px]'] / video_width
                y_coords = fixations_to_analyze['fixation y [px]'] / video_height
            else:
                print(f"WARNING: Fixation coordinates for event '{event_name}' are in pixels, but video dimensions are unavailable. Cannot normalize.")
        
        if not x_coords.empty:
            results.update({
                'fixation_avg_x': x_coords.mean(), 'fixation_std_x': x_coords.std(),
                'fixation_avg_y': y_coords.mean(), 'fixation_std_y': y_coords.std()
            })

    # --- Other Features ---
    gaze_for_fix_count = gaze_not_enr if un_enriched_mode else gaze_enr
    if not gaze_for_fix_count.empty and 'fixation id' in gaze_for_fix_count.columns:
        results['n_gaze_per_fixation_avg'] = gaze_for_fix_count.groupby('fixation id').size().mean()

    if not blinks.empty:
        results.update({'n_blink': len(blinks), 'blink_avg_duration_ms': blinks['duration [ms]'].mean(), 'blink_std_duration_ms': blinks['duration [ms]'].std()})

    if not pupil.empty and 'pupil diameter left [mm]' in pupil.columns and not pupil['pupil diameter left [mm]'].dropna().empty:
        pupil_diam = pupil['pupil diameter left [mm]'].dropna()
        if not pupil_diam.empty:
            results.update({'pupil_start_mm': pupil_diam.iloc[0], 'pupil_end_mm': pupil_diam.iloc[-1], 'pupil_avg_mm': pupil_diam.mean(), 'pupil_std_mm': pupil_diam.std()})

    if not un_enriched_mode and not movements_df.empty:
        results.update({
            'n_movements': len(movements_df), 'sum_time_movement_s': movements_df['duration_ns'].sum() / NS_TO_S,
            'avg_time_movement_s': movements_df['duration_ns'].mean() / NS_TO_S, 'std_time_movement_s': movements_df['duration_ns'].std() / NS_TO_S,
            'total_disp_sum': movements_df['total_displacement'].sum(), 'total_disp_avg': movements_df['total_displacement'].mean(),
            'total_disp_std': movements_df['total_displacement'].std(), 'effective_disp_sum': movements_df['effective_displacement'].sum(),
            'effective_disp_avg': movements_df['effective_displacement'].mean(), 'effective_disp_std': movements_df['effective_displacement'].std()
        })
    elif not saccades.empty:
        sacc_duration_s = saccades['duration [ms]'] / 1000
        amplitude = saccades['amplitude [deg]'] if 'amplitude [deg]' in saccades.columns else pd.Series(dtype='float64')
        results.update({
            'n_movements': len(saccades), 'sum_time_movement_s': sacc_duration_s.sum(), 'avg_time_movement_s': sacc_duration_s.mean(),
            'std_time_movement_s': sacc_duration_s.std(), 'total_disp_sum': amplitude.sum(), 'total_disp_avg': amplitude.mean(),
            'total_disp_std': amplitude.std(), 'effective_disp_sum': amplitude.sum(), 'effective_disp_avg': amplitude.mean(),
            'effective_disp_std': amplitude.std()
        })
            
    return results

def _plot_histogram(data_series, title, xlabel, output_path):
    """Helper function to create a standardized histogram."""
    if data_series.dropna().empty:
        return
    plt.figure(figsize=(10, 6))
    plt.hist(data_series.dropna(), bins=25, color='royalblue', edgecolor='black', alpha=0.7)
    plt.title(title, fontsize=15)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_heatmap(x, y, title, output_path, xlim=None, ylim=None, xlabel='X', ylabel='Y'):
    """Helper function to create a density heatmap with error handling."""
    x = x.dropna(); y = y.dropna()
    valid_points = pd.concat([x, y], axis=1).dropna()
    if len(valid_points) < 3:
        print(f"WARNING: Insufficient data (less than 3 valid points) to generate heatmap: '{title}'")
        return
    try:
        k = gaussian_kde(valid_points.values.T)
        if xlim and ylim:
            xi, yi = np.mgrid[xlim[0]:xlim[1]:100j, ylim[0]:ylim[1]:100j]
        else:
            xi, yi = np.mgrid[valid_points.iloc[:,0].min():valid_points.iloc[:,0].max():100j,
                              valid_points.iloc[:,1].min():valid_points.iloc[:,1].max():100j]

        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        plt.figure(figsize=(10, 8)); plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap='Reds'); plt.plot(valid_points.iloc[:,0], valid_points.iloc[:,1], 'k.', markersize=2, alpha=0.2); plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
        if xlim: plt.xlim(xlim)
        if ylim: plt.ylim(ylim)
        plt.grid(True, linestyle='--'); plt.tight_layout(); plt.savefig(output_path); plt.close()
    except np.linalg.LinAlgError:
        print(f"WARNING: Could not generate heatmap for '{title}' (singular matrix, data not varied enough).")
    except Exception as e:
        print(f"WARNING: Unexpected error during heatmap generation for '{title}': {e}")

def generate_plots(data, movements_df, subj_name, event_name, output_dir: Path, un_enriched_mode: bool, video_width: int, video_height: int):
    """
    Generates and saves all plots for the event, unifying the logic.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fixations_enr, fixations_not_enr = data.get('fixations_enr', pd.DataFrame()), data.get('fixations_not_enr', pd.DataFrame())
    gaze_enr, gaze_not_enr = data.get('gaze', pd.DataFrame()), data.get('gaze_not_enr', pd.DataFrame())
    pupil_df, blinks_df, saccades_df = data.get('pupil', pd.DataFrame()), data.get('blinks', pd.DataFrame()), data.get('saccades', pd.DataFrame())

    # --- Pupillometry Plots ---
    # The logic for pupil plots remains, but the colored background depends on the mode
    if not pupil_df.empty and ('pupil diameter left [mm]' in pupil_df.columns or 'pupil diameter right [mm]' in pupil_df.columns):
        pupil_plot_df = pupil_df.copy()
        
        # Add the surface indicator only if in enriched mode
        if not un_enriched_mode and not gaze_enr.empty and 'gaze detected on surface' in gaze_enr.columns:
            gaze_enr_unique_ts = gaze_enr.drop_duplicates(subset=['timestamp [ns]']).copy()
            pupil_plot_df = pd.merge(pupil_plot_df, gaze_enr_unique_ts[['timestamp [ns]', 'gaze detected on surface']], on='timestamp [ns]', how='left')
            pupil_plot_df['gaze detected on surface'].fillna(False, inplace=True)
        else:
            pupil_plot_df['gaze detected on surface'] = False

        # Left/Right Pupil Plot
        plt.figure(figsize=(12, 6))
        if 'pupil diameter left [mm]' in pupil_plot_df.columns:
            plt.plot(pupil_plot_df['timestamp [ns]'] / NS_TO_S, pupil_plot_df['pupil diameter left [mm]'], label='Pupil Diameter Left [mm]', color='blue', alpha=0.7)
        if 'pupil diameter right [mm]' in pupil_plot_df.columns:
            plt.plot(pupil_plot_df['timestamp [ns]'] / NS_TO_S, pupil_plot_df['pupil diameter right [mm]'], label='Pupil Diameter Right [mm]', color='purple', alpha=0.7)

        # Color the background only in enriched mode
        if not un_enriched_mode:
            timestamps_seconds = pupil_plot_df['timestamp [ns]'] / NS_TO_S
            if len(timestamps_seconds) > 1:
                epsilon = (timestamps_seconds.diff().mean() / 2)
                start_time = timestamps_seconds.iloc[0]
                current_status = pupil_plot_df.iloc[0]['gaze detected on surface']
                for i in range(1, len(pupil_plot_df)):
                    if pupil_plot_df.iloc[i]['gaze detected on surface'] != current_status:
                        end_time = timestamps_seconds.iloc[i]
                        color = 'lightgreen' if current_status else 'lightcoral'
                        plt.axvspan(start_time - epsilon, end_time - epsilon, facecolor=color, alpha=0.3, lw=0)
                        start_time = end_time
                        current_status = pupil_plot_df.iloc[i]['gaze detected on surface']
                color = 'lightgreen' if current_status else 'lightcoral'
                plt.axvspan(start_time - epsilon, timestamps_seconds.iloc[-1] + epsilon, facecolor=color, alpha=0.3, lw=0)

        plt.title(f"Pupil Diameter - {subj_name} - {event_name}", fontsize=15)
        plt.xlabel('Time [s]'); plt.ylabel('Pupil Diameter [mm]'); plt.legend(); plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
        plt.savefig(output_dir / f'pupil_diameter_timeseries_{subj_name}_{event_name}.pdf'); plt.close()

    # --- Histograms ---
    if not un_enriched_mode and not fixations_enr.empty:
        fixations_on_surface = fixations_enr[fixations_enr['fixation detected on surface'] == True]
        if not fixations_on_surface.empty:
            _plot_histogram(fixations_on_surface['duration [ms]'], f"Fixation Duration (Enriched) - {subj_name} - {event_name}", "Duration [ms]", output_dir / f'hist_fixations_enriched_{subj_name}_{event_name}.pdf')
    if not fixations_not_enr.empty:
        _plot_histogram(fixations_not_enr['duration [ms]'], f"Fixation Duration (Un-enriched) - {subj_name} - {event_name}", "Duration [ms]", output_dir / f'hist_fixations_not_enriched_{subj_name}_{event_name}.pdf')
    if not blinks_df.empty:
        _plot_histogram(blinks_df['duration [ms]'], f"Blink Duration - {subj_name} - {event_name}", "Duration [ms]", output_dir / f'hist_blinks_{subj_name}_{event_name}.pdf')
    if not saccades_df.empty:
        _plot_histogram(saccades_df['duration [ms]'], f"Saccade Duration - {subj_name} - {event_name}", "Duration [ms]", output_dir / f'hist_saccades_{subj_name}_{event_name}.pdf')

    # --- Path Plots ---
    if not un_enriched_mode and not fixations_enr.empty:
        enriched_on_surface = fixations_enr[fixations_enr['fixation detected on surface'] == True]
        if not enriched_on_surface.empty:
            plt.figure(figsize=(10, 6)); plt.plot(enriched_on_surface['fixation x [normalized]'], enriched_on_surface['fixation y [normalized]'], marker='o', linestyle='-', color='green'); plt.title(f"Fixation Path (Enriched) - {subj_name} - {event_name}"); plt.xlabel('Normalized X'); plt.ylabel('Normalized Y'); plt.grid(True); plt.savefig(output_dir / f'path_fixation_enriched_{subj_name}_{event_name}.pdf'); plt.close()
    if not fixations_not_enr.empty:
        x, y, xlabel, ylabel = (fixations_not_enr['fixation x [px]'] / video_width, fixations_not_enr['fixation y [px]'] / video_height, 'Normalized X', 'Normalized Y') if video_width and video_height else (fixations_not_enr['fixation x [px]'], fixations_not_enr['fixation y [px]'], 'Pixel X', 'Pixel Y')
        plt.figure(figsize=(10, 6)); plt.plot(x, y, marker='o', linestyle='-', color='purple'); plt.title(f"Fixation Path (Un-enriched) - {subj_name} - {event_name}"); plt.xlabel(xlabel); plt.ylabel(ylabel); plt.grid(True); plt.savefig(output_dir / f'path_fixation_not_enriched_{subj_name}_{event_name}.pdf'); plt.close()

    if not un_enriched_mode and not gaze_enr.empty:
        gaze_on_surface = gaze_enr[gaze_enr['gaze detected on surface'] == True]
        if not gaze_on_surface.empty:
            plt.figure(figsize=(10, 6)); plt.plot(gaze_on_surface['gaze position on surface x [normalized]'], gaze_on_surface['gaze position on surface y [normalized]'], marker='.', linestyle='-', color='red', alpha=0.5); plt.title(f"Gaze Path (Enriched) - {subj_name} - {event_name}"); plt.xlabel('Normalized X'); plt.ylabel('Normalized Y'); plt.grid(True); plt.savefig(output_dir / f'path_gaze_enriched_{subj_name}_{event_name}.pdf'); plt.close()
    if not gaze_not_enr.empty:
        x, y, xlabel, ylabel = (gaze_not_enr['gaze x [px]'] / video_width, gaze_not_enr['gaze y [px]'] / video_height, 'Normalized X', 'Normalized Y') if video_width and video_height else (gaze_not_enr['gaze x [px]'], gaze_not_enr['gaze y [px]'], 'Pixel X', 'Pixel Y')
        plt.figure(figsize=(10, 6)); plt.plot(x, y, marker='.', linestyle='-', color='blue', alpha=0.5); plt.title(f"Gaze Path (Un-enriched) - {subj_name} - {event_name}"); plt.xlabel(xlabel); plt.ylabel(ylabel); plt.grid(True); plt.savefig(output_dir / f'path_gaze_not_enriched_{subj_name}_{event_name}.pdf'); plt.close()

    # --- Heatmaps ---
    if not un_enriched_mode and not fixations_enr.empty:
        fixations_on_surface = fixations_enr[fixations_enr['fixation detected on surface'] == True]
        plot_heatmap(fixations_on_surface['fixation x [normalized]'], fixations_on_surface['fixation y [normalized]'], f"Fixation Density Heatmap (Enriched) - {subj_name} - {event_name}", output_dir / f'heatmap_fixation_enriched_{subj_name}_{event_name}.pdf', xlim=(0, 1), ylim=(0, 1), xlabel='Normalized X', ylabel='Normalized Y')
    if not fixations_not_enr.empty:
        x, y, xlim, ylim, xlabel, ylabel = (fixations_not_enr['fixation x [px]'] / video_width, fixations_not_enr['fixation y [px]'] / video_height, (0,1), (0,1), 'Normalized X', 'Normalized Y') if video_width and video_height else (fixations_not_enr['fixation x [px]'], fixations_not_enr['fixation y [px]'], None, None, 'Pixel X', 'Pixel Y')
        plot_heatmap(x, y, f"Fixation Density Heatmap (Un-enriched) - {subj_name} - {event_name}", output_dir / f'heatmap_fixation_not_enriched_{subj_name}_{event_name}.pdf', xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)

def process_segment(event_row, start_ts, end_ts, all_data, subj_name, output_dir, un_enriched_mode, video_width, video_height):
    """Main processing pipeline for a single event segment."""
    event_name = event_row.get('name', f"segment_{event_row.name}")
    print(f"--- Processing segment for event: '{event_name}' ---")
    rec_id = event_row['recording id']
    
    segment_data = filter_data_by_segment(all_data, start_ts, end_ts, rec_id)
    if all(df.empty for name, df in segment_data.items() if name != 'events'):
        print(f"  -> Skipping segment '{event_name}' because no data was found in the time range.")
        return None
    
    movements_df = process_gaze_movements(segment_data.get('gaze', pd.DataFrame()), un_enriched_mode)
    results = calculate_summary_features(segment_data, movements_df, subj_name, event_name, un_enriched_mode, video_width, video_height)
    generate_plots(segment_data, movements_df, subj_name, event_name, output_dir, un_enriched_mode, video_width, video_height)

    return results

def get_video_dimensions(video_path: Path):
    """Gets width and height from a video file."""
    if not video_path.exists():
        print(f"WARNING: Video file not found at {video_path}. Cannot get dimensions for normalization.")
        return None, None
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"WARNING: Could not open video file {video_path}.")
        return None, None
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height

def create_analysis_video(data_dir: Path, output_dir: Path):
    """
    Creates an analysis video combining eye tracking, world view, and pupil diameter.
    """
    print("\nCreating analysis video...")
    internal_video_path = data_dir / 'internal.mp4'
    external_video_path = data_dir / 'external.mp4'
    pupil_data_path = data_dir / '3d_eye_states.csv'
    blinks_data_path = data_dir / 'blinks.csv'

    if not all([p.exists() for p in [internal_video_path, external_video_path, pupil_data_path, blinks_data_path]]):
        print("Skipping video creation: required files not found.")
        return

    try:
        pupil_data = pd.read_csv(pupil_data_path)
        blinks_data = pd.read_csv(blinks_data_path)
        t0 = pupil_data['timestamp [ns]'].min()
        pupil_data['time_sec'] = (pupil_data['timestamp [ns]'] - t0) / 1e9
        blinks_data['start_sec'] = (blinks_data['start timestamp [ns]'] - t0) / 1e9
        blinks_data['end_sec'] = blinks_data['start_sec'] + (blinks_data['duration [ms]'] / 1000)
        pupil_data['pupil_diameter_mean'] = pupil_data[['pupil diameter left [mm]', 'pupil diameter right [mm]']].mean(axis=1)

        cap1 = cv2.VideoCapture(str(internal_video_path))
        cap2 = cv2.VideoCapture(str(external_video_path))
        fps = cap1.get(cv2.CAP_PROP_FPS)
        if not cap1.isOpened() or not cap2.isOpened():
            print("Error opening video files for animation."); return

        fig, (video_axes1, video_axes2, ts_axes) = plt.subplots(3, 1, figsize=(12, 9), gridspec_kw={'height_ratios': [3, 3, 2]})
        fig.tight_layout(pad=4.0)
        
        ts_axes.plot(pupil_data['time_sec'], pupil_data['pupil diameter left [mm]'], color='dodgerblue', alpha=0.8, label='Left Pupil')
        ts_axes.plot(pupil_data['time_sec'], pupil_data['pupil diameter right [mm]'], color='orchid', alpha=0.8, label='Right Pupil')
        ts_axes.plot(pupil_data['time_sec'], pupil_data['pupil_diameter_mean'], color='black', linestyle='--', lw=1.5, label='Mean Pupil')
        for _, blink in blinks_data.iterrows():
            ts_axes.axvspan(blink['start_sec'], blink['end_sec'], color='red', alpha=0.4, lw=0)
        ts_axes.set_title("Pupil Diameter Over Time", fontsize=16); ts_axes.set_xlabel("Time (s)"); ts_axes.set_ylabel("Diameter (mm)"); ts_axes.legend(); ts_axes.grid(True, linestyle='--', alpha=0.6)
        ts_axes.set_xlim(0, pupil_data['time_sec'].max())
        min_pupil = min(pupil_data['pupil diameter left [mm]'].min(), pupil_data['pupil diameter right [mm]'].min())
        max_pupil = max(pupil_data['pupil diameter left [mm]'].max(), pupil_data['pupil diameter right [mm]'].max())
        ts_axes.set_ylim(min_pupil * 0.95, max_pupil * 1.05)
        time_indicator = ts_axes.axvline(x=0, color='red', linestyle='-', lw=2)

        output_video_path = output_dir / 'output_analysis_video.mp4'
        w, h = fig.canvas.get_width_height()
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (w, h))
        if not out.isOpened():
            print(f"Error: Could not create the output video file: {output_video_path}"); cap1.release(); cap2.release(); plt.close(fig); return

        frame_count = 0
        pbar = tqdm(total=int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Creating Summary Video")
        while cap1.isOpened() and cap2.isOpened():
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            if not ret1 or not ret2: break
            
            pbar.update(1)
            current_video_time_sec = cap1.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            video_axes1.clear(); video_axes2.clear()
            video_axes1.imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)); video_axes1.axis('off'); video_axes1.set_title("Internal View (Eye)")
            video_axes2.imshow(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)); video_axes2.axis('off'); video_axes2.set_title("External View")
            time_indicator.set_xdata([current_video_time_sec])
            fig.canvas.draw()
            img = np.array(fig.canvas.buffer_rgba())
            out.write(cv2.cvtColor(img, cv2.COLOR_RGBA2BGR))
            frame_count += 1

        pbar.close(); cap1.release(); cap2.release(); out.release(); plt.close(fig)
        print(f"Analysis video saved to {output_video_path}")
    except Exception as e:
        print(f"An unexpected error occurred during video creation: {e}"); traceback.print_exc()

def run_analysis(subj_name='subj_01', data_dir_str='./eyetracking_file', output_dir_str='./results', un_enriched_mode=False, generate_video=True):
    """Main function to run the entire analysis pipeline based on event segments."""
    pd.options.mode.chained_assignment = None
    data_dir, output_dir = Path(data_dir_str), Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_width, video_height = get_video_dimensions(data_dir / 'external.mp4')
    
    try:
        all_data = load_all_data(data_dir, un_enriched_mode)
    except FileNotFoundError as e:
        print(f"Analysis stopped. {e}"); return
        
    events_df = all_data.get('events')
    if events_df is None or events_df.empty:
        print("Error: events.csv not loaded or empty. Cannot proceed."); return
        
    all_results = []
    if len(events_df) > 1:
        print(f"\nFound {len(events_df)} events, processing {len(events_df) - 1} segments.")
        for i in range(len(events_df) - 1):
            event_row, start_ts, end_ts = events_df.iloc[i], events_df.iloc[i]['timestamp [ns]'], events_df.iloc[i+1]['timestamp [ns]']
            try:
                event_results = process_segment(event_row, start_ts, end_ts, all_data, subj_name, output_dir, un_enriched_mode, video_width, video_height)
                if event_results:
                    all_results.append(event_results)
            except Exception as e:
                print(f"Could not process segment for event '{event_row.get('name', i)}'. Error: {e}"); traceback.print_exc()
    else:
        print("Warning: Fewer than two events found. Cannot process segments.")
        
    if all_results:
        results_df = pd.DataFrame(all_results)
        column_order = [
            'participant', 'event', 'n_fixation', 'fixation_avg_duration_ms', 'fixation_std_duration_ms',
            'fixation_avg_x', 'fixation_std_x', 'fixation_avg_y', 'fixation_std_y', 'n_gaze_per_fixation_avg',
            'n_blink', 'blink_avg_duration_ms', 'blink_std_duration_ms', 'pupil_start_mm', 'pupil_end_mm',
            'pupil_avg_mm', 'pupil_std_mm', 'n_movements', 'sum_time_movement_s', 'avg_time_movement_s',
            'std_time_movement_s', 'total_disp_sum', 'total_disp_avg', 'total_disp_std', 'effective_disp_sum',
            'effective_disp_avg', 'effective_disp_std'
        ]
        final_columns = [col for col in column_order if col in results_df.columns]
        results_df = results_df[final_columns]
        results_filename = output_dir / f'summary_results_{subj_name}.csv'
        results_df.to_csv(results_filename, index=False)
        print(f"\nAggregate results saved to {results_filename}")
    else:
        print("\nNo analysis results were generated.")
        
    if generate_video:
        create_analysis_video(data_dir, output_dir)
