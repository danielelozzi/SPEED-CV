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
            if name in ['gaze', 'fixations_enr', 'fixations_not_enr', 'gaze_not_enr']:
                print(f"Info: Optional or base file {filename} not found, proceeding without it.")
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
        if 'fixation x [normalized]' in fixations_to_analyze.columns:
            print("DEBUG: Using pre-normalized coordinates from the enriched file.")
            x_coords, y_coords = fixations_to_analyze['fixation x [normalized]'], fixations_to_analyze['fixation y [normalized]']
        elif 'fixation x [px]' in fixations_to_analyze.columns:
            if video_width and video_height and video_width > 0 and video_height > 0:
                print(f"DEBUG: Normalizing pixel coordinates using video dimensions {video_width}x{video_height}.")
                x_coords = fixations_to_analyze['fixation x [px]'] / video_width
                y_coords = fixations_to_analyze['fixation y [px]'] / video_height
            else:
                print("WARNING: Fixation coordinates are in pixels, but video dimensions are unavailable. Cannot normalize.")
        
        if not x_coords.empty:
            results.update({
                'fixation_avg_x': x_coords.mean(), 'fixation_std_x': x_coords.std(),
                'fixation_avg_y': y_coords.mean(), 'fixation_std_y': y_coords.std()
            })

    # --- Other Features ---
    gaze_for_fix_count = pd.DataFrame()
    if not un_enriched_mode and not gaze_enr.empty and 'fixation id' in gaze_enr.columns:
        gaze_for_fix_count = gaze_enr
    elif not gaze_not_enr.empty and 'fixation id' in gaze_not_enr.columns:
        gaze_for_fix_count = gaze_not_enr
    if not gaze_for_fix_count.empty:
        results['n_gaze_per_fixation_avg'] = gaze_for_fix_count.groupby('fixation id').size().mean()

    if not blinks.empty:
        results.update({'n_blink': len(blinks), 'blink_avg_duration_ms': blinks['duration [ms]'].mean(), 'blink_std_duration_ms': blinks['duration [ms]'].std()})

    if not pupil.empty and 'pupil diameter left [mm]' in pupil.columns and not pupil['pupil diameter left [mm]'].dropna().empty:
        pupil_diam = pupil['pupil diameter left [mm]'].dropna()
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
        print("DEBUG: Calculating movement features from saccades.csv")
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

def generate_plots(data, movements_df, subj_name, event_name, output_dir: Path, un_enriched_mode: bool, video_width: int, video_height: int):
    """
    Generates and saves all plots for the event, unifying logic.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fixations_enr, fixations_not_enr = data.get('fixations_enr', pd.DataFrame()), data.get('fixations_not_enr', pd.DataFrame())
    gaze_enr, gaze_not_enr = data.get('gaze', pd.DataFrame()), data.get('gaze_not_enr', pd.DataFrame())
    pupil_df, blinks_df, saccades_df = data.get('pupil', pd.DataFrame()), data.get('blinks', pd.DataFrame()), data.get('saccades', pd.DataFrame())

    # --- Pupillometry Plot (Left and Right separate) ---
    if not pupil_df.empty and ('pupil diameter left [mm]' in pupil_df.columns or 'pupil diameter right [mm]' in pupil_df.columns):
        if not un_enriched_mode and not gaze_enr.empty and 'gaze detected on surface' in gaze_enr.columns:
            gaze_enr_unique_ts = gaze_enr.drop_duplicates(subset=['timestamp [ns]']).copy()
            pupil_with_gaze_status = pd.merge(
                pupil_df,
                gaze_enr_unique_ts[['timestamp [ns]', 'gaze detected on surface']],
                on='timestamp [ns]',
                how='left'
            )
            pupil_with_gaze_status['gaze detected on surface'] = pupil_with_gaze_status['gaze detected on surface'].fillna(False)
            pupil_plot_df = pupil_with_gaze_status
        else:
            pupil_plot_df = pupil_df.copy()
            pupil_plot_df['gaze detected on surface'] = False # Default to False if no enriched data or un_enriched_mode

        if not pupil_plot_df.empty:
            plt.figure(figsize=(12, 6))
            
            if 'pupil diameter left [mm]' in pupil_plot_df.columns:
                plt.plot(pupil_plot_df['timestamp [ns]'] / NS_TO_S, pupil_plot_df['pupil diameter left [mm]'], label='Pupil Diameter Left [mm]', color='blue', alpha=0.7)
            if 'pupil diameter right [mm]' in pupil_plot_df.columns:
                plt.plot(pupil_plot_df['timestamp [ns]'] / NS_TO_S, pupil_plot_df['pupil diameter right [mm]'], label='Pupil Diameter Right [mm]', color='purple', alpha=0.7)

            if 'gaze detected on surface' in pupil_plot_df.columns and not un_enriched_mode:
                timestamps_seconds = pupil_plot_df['timestamp [ns]'] / NS_TO_S
                epsilon = (timestamps_seconds.diff().mean() / 2) if len(timestamps_seconds) > 1 else 0.01

                start_time = timestamps_seconds.iloc[0] if not timestamps_seconds.empty else None
                current_status = pupil_plot_df.iloc[0]['gaze detected on surface'] if not pupil_plot_df.empty else None
                
                if start_time is not None and current_status is not None:
                    for i in range(1, len(pupil_plot_df)):
                        if pupil_plot_df.iloc[i]['gaze detected on surface'] != current_status:
                            end_time = timestamps_seconds.iloc[i]
                            color = 'lightgreen' if current_status else 'lightcoral'
                            plt.axvspan(start_time - epsilon, end_time - epsilon, facecolor=color, alpha=0.3, lw=0)
                            start_time = end_time
                            current_status = pupil_plot_df.iloc[i]['gaze detected on surface']
                    color = 'lightgreen' if current_status else 'lightcoral'
                    plt.axvspan(start_time - epsilon, timestamps_seconds.iloc[-1] + epsilon, facecolor=color, alpha=0.3, lw=0)

            plt.title(f"Pupil Diameter with Gaze On Surface - {subj_name} - {event_name}", fontsize=15)
            plt.xlabel('Time [s]', fontsize=12)
            plt.ylabel('Pupil Diameter [mm]', fontsize=12)
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(output_dir / f'pupil_diameter_gaze_surface_{subj_name}_{event_name}.pdf')
            plt.close()
        else:
            print(f"WARNING: Insufficient pupil data for event '{event_name}'. No plot generated.")

    # --- Pupillometry Plot (Mean of Left and Right) ---
    if not pupil_df.empty and 'pupil diameter left [mm]' in pupil_df.columns and 'pupil diameter right [mm]' in pupil_df.columns and not un_enriched_mode:
        pupil_df_mean = pupil_df.copy()
        pupil_df_mean['pupil_diameter_mean'] = pupil_df_mean[['pupil diameter left [mm]', 'pupil diameter right [mm]']].mean(axis=1)

        gaze_enr_unique_ts = gaze_enr.drop_duplicates(subset=['timestamp [ns]']).copy()
        pupil_with_gaze_status = pd.merge(pupil_df_mean, gaze_enr_unique_ts[['timestamp [ns]', 'gaze detected on surface']], on='timestamp [ns]', how='left')
        pupil_with_gaze_status['gaze detected on surface'] = pupil_with_gaze_status['gaze detected on surface'].fillna(False)
        pupil_with_gaze_status.dropna(subset=['pupil_diameter_mean'], inplace=True)
        pupil_with_gaze_status.reset_index(drop=True, inplace=True)

        if not pupil_with_gaze_status.empty:
            plt.figure(figsize=(12, 6))
            plt.plot(pupil_with_gaze_status['timestamp [ns]'] / NS_TO_S, pupil_with_gaze_status['pupil_diameter_mean'], label='Mean Pupil Diameter [mm]', color='black', alpha=0.8)

            timestamps_seconds = pupil_with_gaze_status['timestamp [ns]'] / NS_TO_S
            epsilon = (timestamps_seconds.diff().mean() / 2) if len(timestamps_seconds) > 1 else 0.01
            
            start_time = timestamps_seconds.iloc[0] if not timestamps_seconds.empty else None
            current_status = pupil_with_gaze_status.iloc[0]['gaze detected on surface'] if not pupil_with_gaze_status.empty else None
            
            if start_time is not None and current_status is not None:
                for i in range(1, len(pupil_with_gaze_status)):
                    if pupil_with_gaze_status.iloc[i]['gaze detected on surface'] != current_status:
                        end_time = timestamps_seconds.iloc[i]
                        color = 'lightgreen' if current_status else 'lightcoral'
                        plt.axvspan(start_time - epsilon, end_time - epsilon, facecolor=color, alpha=0.3, lw=0)
                        start_time = end_time
                        current_status = pupil_with_gaze_status.iloc[i]['gaze detected on surface']
                color = 'lightgreen' if current_status else 'lightcoral'
                plt.axvspan(start_time - epsilon, timestamps_seconds.iloc[-1] + epsilon, facecolor=color, alpha=0.3, lw=0)

            plt.title(f"Mean Pupil Diameter with Gaze On Surface - {subj_name} - {event_name}", fontsize=15)
            plt.xlabel('Time [s]'); plt.ylabel('Mean Pupil Diameter [mm]'); plt.legend(); plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
            plt.savefig(output_dir / f'pupil_diameter_mean_gaze_surface_{subj_name}_{event_name}.pdf'); plt.close()
        else:
            print(f"WARNING: Insufficient mean pupil data for event '{event_name}'. No plot generated.")

    # --- Periodogram and Spectrogram ---
    if not pupil_df.empty and 'pupil diameter left [mm]' in pupil_df.columns:
        ts = pupil_df['pupil diameter left [mm]'].dropna().to_numpy()
        if len(ts) > SAMPLING_FREQ:
            try:
                freqs, Pxx = welch(ts, fs=SAMPLING_FREQ, nperseg=min(len(ts), 100))
                plt.figure(figsize=(10, 5)); plt.semilogy(freqs, Pxx); plt.title(f'Periodogram - {subj_name} - {event_name}'); plt.xlabel('Frequency [Hz]'); plt.ylabel('Power Spectral Density [V^2/Hz]'); plt.grid(True); plt.savefig(output_dir / f'periodogram_{subj_name}_{event_name}.pdf'); plt.close()
                
                f, t, Sxx = spectrogram(ts, fs=SAMPLING_FREQ, nperseg=min(len(ts), 256), noverlap=min(len(ts)//2, 50))
                plt.figure(figsize=(10, 5)); plt.pcolormesh(t, f, 10 * np.log10(np.maximum(Sxx, 1e-10)), shading='gouraud'); plt.title(f'Spectrogram - {subj_name} - {event_name}'); plt.ylabel('Frequency [Hz]'); plt.xlabel('Time [s]'); plt.colorbar(label='Power [dB]'); plt.savefig(output_dir / f'spectrogram_{subj_name}_{event_name}.pdf'); plt.close()
            except Exception as e:
                print(f"WARNING: Could not generate periodogram/spectrogram for '{event_name}'. Error: {e}")
        else:
            print(f"WARNING: Insufficient pupil data for periodogram/spectrogram for event '{event_name}'.")

    # --- Histograms ---
    if not un_enriched_mode and not fixations_enr.empty and 'duration [ms]' in fixations_enr.columns:
        fixations_on_surface = fixations_enr[fixations_enr['fixation detected on surface'] == True]
        if not fixations_on_surface.empty:
            _plot_histogram(fixations_on_surface['duration [ms]'], f"Fixation Duration (Enriched on Surface) - {subj_name} - {event_name}", "Duration [ms]", output_dir / f'hist_fixations_enriched_{subj_name}_{event_name}.pdf')
    if not fixations_not_enr.empty and 'duration [ms]' in fixations_not_enr.columns:
        _plot_histogram(fixations_not_enr['duration [ms]'], f"Fixation Duration (Un-enriched) - {subj_name} - {event_name}", "Duration [ms]", output_dir / f'hist_fixations_not_enriched_{subj_name}_{event_name}.pdf')

    if not blinks_df.empty and 'duration [ms]' in blinks_df.columns:
        _plot_histogram(blinks_df['duration [ms]'], f"Blink Duration Histogram - {subj_name} - {event_name}", "Duration [ms]", output_dir / f'hist_blinks_{subj_name}_{event_name}.pdf')
    if not saccades_df.empty and 'duration [ms]' in saccades_df.columns:
        _plot_histogram(saccades_df['duration [ms]'], f"Saccade Duration Histogram - {subj_name} - {event_name}", "Duration [ms]", output_dir / f'hist_saccades_{subj_name}_{event_name}.pdf')

    # --- Path Plots (Fixation Path) ---
    if not un_enriched_mode and not fixations_enr.empty and 'fixation x [normalized]' in fixations_enr.columns:
        enriched_on_surface = fixations_enr[fixations_enr['fixation detected on surface'] == True]
        if not enriched_on_surface.empty:
            plt.figure(figsize=(10, 6)); plt.plot(enriched_on_surface['fixation x [normalized]'], enriched_on_surface['fixation y [normalized]'], marker='o', linestyle='-', color='green'); plt.title(f"Fixation Path (Enriched) - {subj_name} - {event_name}"); plt.xlabel('Normalized X'); plt.ylabel('Normalized Y'); plt.grid(True); plt.savefig(output_dir / f'path_fixation_enriched_{subj_name}_{event_name}.pdf'); plt.close()
        else:
            print(f"WARNING: No enriched fixations detected on surface for event '{event_name}'. No plot generated.")

    if not fixations_not_enr.empty and 'fixation x [px]' in fixations_not_enr.columns:
        plt.figure(figsize=(10, 6))
        
        x_coords_px = fixations_not_enr['fixation x [px]']
        y_coords_px = fixations_not_enr['fixation y [px]']

        if video_width and video_height and video_width > 0 and video_height > 0:
            x_coords, y_coords = x_coords_px / video_width, y_coords_px / video_height
            xlabel_text, ylabel_text = 'Normalized X', 'Normalized Y'
        else:
            x_coords, y_coords = x_coords_px, y_coords_px
            xlabel_text, ylabel_text = 'Pixel X', 'Pixel Y'
            print("WARNING: Un-enriched fixation coordinates are in pixels and video dimensions are not available for normalization.")

        plt.plot(x_coords, y_coords, marker='o', linestyle='-', color='purple')
        plt.title(f"Fixation Path (Un-enriched) - {subj_name} - {event_name}"); plt.xlabel(xlabel_text); plt.ylabel(ylabel_text); plt.grid(True)
        plt.savefig(output_dir / f'path_fixation_not_enriched_{subj_name}_{event_name}.pdf')
        plt.close()
    else:
        print(f"WARNING: Insufficient un-enriched fixation data for event '{event_name}'. No plot generated.")
    
    # --- Path Plots (Gaze Path) ---
    if not un_enriched_mode and not gaze_enr.empty and 'gaze position on surface x [normalized]' in gaze_enr.columns:
        gaze_on_surface = gaze_enr[gaze_enr['gaze detected on surface'] == True]
        if not gaze_on_surface.empty:
            plt.figure(figsize=(10, 6)); plt.plot(gaze_on_surface['gaze position on surface x [normalized]'], gaze_on_surface['gaze position on surface y [normalized]'], marker='.', linestyle='-', color='red', alpha=0.5); plt.title(f"Gaze Path (Enriched) - {subj_name} - {event_name}"); plt.xlabel('Normalized X'); plt.ylabel('Normalized Y'); plt.grid(True); plt.savefig(output_dir / f'path_gaze_enriched_{subj_name}_{event_name}.pdf'); plt.close()

    if not gaze_not_enr.empty:
        x_col, y_col, coords_type = None, None, None
        if 'gaze position x [px]' in gaze_not_enr.columns and 'gaze position y [px]' in gaze_not_enr.columns:
            x_col, y_col, coords_type = 'gaze position x [px]', 'gaze position y [px]', 'pixel'
        elif 'gaze x [norm]' in gaze_not_enr.columns and 'gaze y [norm]' in gaze_not_enr.columns:
            x_col, y_col, coords_type = 'gaze x [norm]', 'gaze y [norm]', 'normalized'
        
        if x_col:
            plt.figure(figsize=(10, 6))
            x, y = gaze_not_enr[x_col], gaze_not_enr[y_col]
            
            if coords_type == 'pixel' and video_width and video_height and video_width > 0 and video_height > 0:
                x, y = x / video_width, y / video_height
                xlabel, ylabel = 'Normalized X', 'Normalized Y'
            elif coords_type == 'normalized':
                 xlabel, ylabel = 'Normalized X', 'Normalized Y'
            else:
                xlabel, ylabel = 'Pixel X', 'Pixel Y'
                
            plt.plot(x, y, marker='.', linestyle='-', color='blue', alpha=0.5)
            plt.title(f"Gaze Path (Un-enriched) - {subj_name} - {event_name}")
            plt.xlabel(xlabel); plt.ylabel(ylabel); plt.grid(True)
            plt.savefig(output_dir / f'path_gaze_not_enriched_{subj_name}_{event_name}.pdf'); plt.close()
        else:
            print(f"WARNING: Un-enriched gaze columns ('gaze position x [px]' or 'gaze x [norm]') not found for event '{event_name}'.")
    else:
        print(f"WARNING: Insufficient un-enriched gaze data for event '{event_name}'. No plot generated.")

    # --- Saccade Velocities Plot ---
    if not saccades_df.empty and 'mean velocity [px/s]' in saccades_df.columns and 'peak velocity [px/s]' in saccades_df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(saccades_df.index, saccades_df['mean velocity [px/s]'], label='mean velocity', color='blue')
        plt.plot(saccades_df.index, saccades_df['peak velocity [px/s]'], label='peak velocity', color='orange')
        plt.title(f'Mean and Peak Saccade Velocity - {subj_name} - {event_name}', fontsize=15)
        plt.xlabel('Frames (n)', fontsize=12)
        plt.ylabel('px/s', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_dir / f'saccade_velocities_{subj_name}_{event_name}.pdf')
        plt.close()
    else:
        print(f"WARNING: Insufficient saccade velocity data for event '{event_name}'.")

    # --- Saccade Amplitude Plot ---
    if not saccades_df.empty and 'amplitude [px]' in saccades_df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(saccades_df.index, saccades_df['amplitude [px]'], label='amplitude', color='teal')
        plt.title(f'Saccade Amplitude - {subj_name} - {event_name}', fontsize=15)
        plt.xlabel('Frames (n)', fontsize=12)
        plt.ylabel('px', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_dir / f'saccade_amplitude_{subj_name}_{event_name}.pdf')
        plt.close()
    else:
        print(f"WARNING: Insufficient saccade amplitude data for event '{event_name}'.")

    # --- Blink Time Series Plot ---
    if not blinks_df.empty and 'start timestamp [ns]' in blinks_df.columns and 'duration [ms]' in blinks_df.columns:
        all_timestamps = []
        if 'timestamp [ns]' in pupil_df.columns:
            all_timestamps.extend(pupil_df['timestamp [ns]'].tolist())
        if 'start timestamp [ns]' in blinks_df.columns:
            all_timestamps.extend(blinks_df['start timestamp [ns]'].tolist())
        if 'start timestamp [ns]' in saccades_df.columns:
            all_timestamps.extend(saccades_df['start timestamp [ns]'].tolist())
        
        if not all_timestamps:
            print(f"WARNING: No timestamps available to create blink time series for event '{event_name}'.")
            return

        min_ts = min(all_timestamps)
        max_ts = max(all_timestamps)

        duration_s = (max_ts - min_ts) / NS_TO_S
        num_points = int(duration_s * SAMPLING_FREQ)
        
        if num_points <= 0:
            print(f"WARNING: Segment duration too short to generate blink time series for event '{event_name}'.")
            return

        time_axis_s = np.linspace(0, duration_s, num_points)
        blink_time_series = np.zeros(num_points)

        for _, row in blinks_df.iterrows():
            blink_start_ns = row['start timestamp [ns]']
            blink_end_ns = blink_start_ns + (row['duration [ms]'] * 1_000_000) # Convert ms to ns

            start_idx = int(((blink_start_ns - min_ts) / NS_TO_S) * SAMPLING_FREQ)
            end_idx = int(((blink_end_ns - min_ts) / NS_TO_S) * SAMPLING_FREQ)

            start_idx = max(0, start_idx)
            end_idx = min(num_points, end_idx)

            if start_idx < end_idx:
                blink_time_series[start_idx:end_idx] = 1

        plt.figure(figsize=(12, 4))
        plt.plot(time_axis_s, blink_time_series, drawstyle='steps-post', color='blue')
        plt.title(f'Blink Time Series - {subj_name} - {event_name}', fontsize=15)
        plt.xlabel('Time [s]', fontsize=12)
        plt.ylabel('Blink (0 = No, 1 = Yes)', fontsize=12)
        plt.yticks([0, 1])
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.ylim(-0.1, 1.1)
        plt.tight_layout()
        plt.savefig(output_dir / f'blink_time_series_{subj_name}_{event_name}.pdf')
        plt.close()
    else:
        print(f"WARNING: Insufficient blink data for event '{event_name}'.")

    # --- Density Heatmaps ---
    # Enriched Fixations Heatmap (on surface)
    if not un_enriched_mode and not fixations_enr.empty and 'fixation x [normalized]' in fixations_enr.columns:
        fixations_enriched_on_surface = fixations_enr[fixations_enr['fixation detected on surface'] == True].copy()
        plot_heatmap(fixations_enriched_on_surface['fixation x [normalized]'], fixations_enriched_on_surface['fixation y [normalized]'], 
                     f"Fixation Density Heatmap (Enriched) - {subj_name} - {event_name}", 
                     output_dir / f'heatmap_fixation_enriched_{subj_name}_{event_name}.pdf', xlim=(0, 1), ylim=(0, 1), 
                     xlabel='Normalized X', ylabel='Normalized Y')

    # Un-enriched Fixations Heatmap
    if not fixations_not_enr.empty and 'fixation x [px]' in fixations_not_enr.columns:
        x_px = fixations_not_enr['fixation x [px]'].dropna()
        y_px = fixations_not_enr['fixation y [px]'].dropna()

        if video_width and video_height and video_width > 0 and video_height > 0:
            x_coords, y_coords = x_px / video_width, y_px / video_height
            xlabel_text, ylabel_text = 'Normalized X', 'Normalized Y'
            xlim, ylim = (0, 1), (0, 1)
        else:
            x_coords, y_coords = x_px, y_px
            xlabel_text, ylabel_text = 'Pixel X', 'Pixel Y'
            xlim, ylim = (None, None), (None, None) 
            print("WARNING: Un-enriched fixation coordinates are in pixels and video dimensions are not available for heatmap normalization.")
        
        plot_heatmap(x_coords, y_coords, f"Fixation Density Heatmap (Un-enriched) - {subj_name} - {event_name}", 
                     output_dir / f'heatmap_fixation_not_enriched_{subj_name}_{event_name}.pdf', xlim=xlim, ylim=ylim, 
                     xlabel=xlabel_text, ylabel=ylabel_text)

    # Enriched Gaze Heatmap (on surface)
    if not un_enriched_mode and not gaze_enr.empty and 'gaze position on surface x [normalized]' in gaze_enr.columns:
        gaze_on_surface = gaze_enr[gaze_enr['gaze detected on surface'] == True].copy()
        plot_heatmap(gaze_on_surface['gaze position on surface x [normalized]'], gaze_on_surface['gaze position on surface y [normalized]'], 
                     f"Gaze Density Heatmap (Enriched) - {subj_name} - {event_name}", 
                     output_dir / f'heatmap_gaze_enriched_{subj_name}_{event_name}.pdf', xlim=(0, 1), ylim=(0, 1), 
                     xlabel='Normalized X', ylabel='Normalized Y')

    # Un-enriched Gaze Heatmap
    if not gaze_not_enr.empty:
        x_col, y_col, coords_type = None, None, None
        if 'gaze position x [px]' in gaze_not_enr.columns and 'gaze position y [px]' in gaze_not_enr.columns:
            x_col, y_col, coords_type = 'gaze position x [px]', 'gaze position y [px]', 'pixel'
        elif 'gaze x [norm]' in gaze_not_enr.columns and 'gaze y [norm]' in gaze_not_enr.columns:
            x_col, y_col, coords_type = 'gaze x [norm]', 'gaze y [norm]', 'normalized'
            
        if x_col:
            x_coords, y_coords = gaze_not_enr[x_col], gaze_not_enr[y_col]
            
            if coords_type == 'pixel' and video_width and video_height and video_width > 0 and video_height > 0:
                x_coords, y_coords = x_coords / video_width, y_coords / video_height
                xlim, ylim, xlabel, ylabel = (0, 1), (0, 1), 'Normalized X', 'Normalized Y'
            elif coords_type == 'normalized':
                xlim, ylim, xlabel, ylabel = (0, 1), (0, 1), 'Normalized X', 'Normalized Y'
            else:
                xlim, ylim, xlabel, ylabel = (None, None), (None, None), 'Pixel X', 'Pixel Y' 
                print("WARNING: Un-enriched gaze coordinates are in pixels and video dimensions are not available for heatmap normalization.")
                
            plot_heatmap(x_coords, y_coords, f"Gaze Density Heatmap (Un-enriched) - {subj_name} - {event_name}", 
                         output_dir / f'heatmap_gaze_not_enriched_{subj_name}_{event_name}.pdf', xlim=xlim, ylim=ylim, 
                         xlabel=xlabel, ylabel=ylabel)
        else:
            print(f"WARNING: Un-enriched gaze columns ('gaze position x [px]' or 'gaze x [norm]') not found for heatmap for event '{event_name}'.")
    else:
        print(f"WARNING: Insufficient un-enriched gaze data for heatmap for event '{event_name}'. No plot generated.")

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

def downsample_video(input_file, output_file, input_fps, output_fps):
    """Downsamples a video file to a lower FPS."""
    cap = cv2.VideoCapture(str(input_file))
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_file}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(str(output_file), fourcc, output_fps, (width, height))

    if out.isOpened():
        frame_interval = int(input_fps / output_fps)
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret: break
            if i % frame_interval == 0:
                out.write(frame)
        print(f"Video downsampled to {output_file}")
    else:
        print(f"Error: Could not open the output video file for downsampling: {output_file}")
    cap.release()
    out.release()

def create_analysis_video(data_dir: Path, output_dir: Path):
    """
    Creates an analysis video combining eye tracking, world view, and pupil diameter.
    MODIFIED VERSION for temporal synchronization and improved plots.
    """
    print("\nCreating analysis video...")

    # --- File Paths ---
    internal_video_path = data_dir / 'internal.mp4'
    external_video_path = data_dir / 'external.mp4'
    pupil_data_path = data_dir / '3d_eye_states.csv'
    blinks_data_path = data_dir / 'blinks.csv'

    if not all([p.exists() for p in [internal_video_path, external_video_path, pupil_data_path, blinks_data_path]]):
        print("Skipping video creation: required files not found (internal.mp4, external.mp4, 3d_eye_states.csv, blinks.csv).")
        return

    try:
        # --- Data Loading and Preparation ---
        pupil_data = pd.read_csv(pupil_data_path)
        blinks_data = pd.read_csv(blinks_data_path)

        # Find the initial timestamp to normalize the time axis
        t0 = pupil_data['timestamp [ns]'].min()
        
        # Convert timestamps to seconds from the start
        pupil_data['time_sec'] = (pupil_data['timestamp [ns]'] - t0) / 1e9
        blinks_data['start_sec'] = (blinks_data['start timestamp [ns]'] - t0) / 1e9
        blinks_data['end_sec'] = blinks_data['start_sec'] + (blinks_data['duration [ms]'] / 1000)
        
        # Calculate the mean of pupil diameters
        pupil_data['pupil_diameter_mean'] = pupil_data[['pupil diameter left [mm]', 'pupil diameter right [mm]']].mean(axis=1)


        # --- Video Setup ---
        cap1 = cv2.VideoCapture(str(internal_video_path))
        cap2 = cv2.VideoCapture(str(external_video_path))
        fps = cap1.get(cv2.CAP_PROP_FPS)

        if not cap1.isOpened() or not cap2.isOpened():
            print("Error opening video files for animation.")
            return

        # --- Plot Setup ---
        fig, (video_axes1, video_axes2, ts_axes) = plt.subplots(3, 1, figsize=(12, 9), gridspec_kw={'height_ratios': [3, 3, 2]})
        fig.tight_layout(pad=4.0)
        
        # Plot the entire time series once
        ts_axes.plot(pupil_data['time_sec'], pupil_data['pupil diameter left [mm]'], color='dodgerblue', alpha=0.8, label='Left Pupil')
        ts_axes.plot(pupil_data['time_sec'], pupil_data['pupil diameter right [mm]'], color='orchid', alpha=0.8, label='Right Pupil')
        ts_axes.plot(pupil_data['time_sec'], pupil_data['pupil_diameter_mean'], color='black', linestyle='--', lw=1.5, label='Mean Pupil')


        # Add red bands for blinks
        for _, blink in blinks_data.iterrows():
            ts_axes.axvspan(blink['start_sec'], blink['end_sec'], color='red', alpha=0.4, lw=0)

        ts_axes.set_title("Pupil Diameter Over Time", fontsize=16)
        ts_axes.set_xlabel("Time (s)")
        ts_axes.set_ylabel("Diameter (mm)")
        ts_axes.legend()
        ts_axes.grid(True, linestyle='--', alpha=0.6)
        
        # Set axis limits to prevent them from changing
        max_time = pupil_data['time_sec'].max()
        ts_axes.set_xlim(0, max_time)
        min_pupil = min(pupil_data['pupil diameter left [mm]'].min(), pupil_data['pupil diameter right [mm]'].min())
        max_pupil = max(pupil_data['pupil diameter left [mm]'].max(), pupil_data['pupil diameter right [mm]'].max())
        ts_axes.set_ylim(min_pupil * 0.95, max_pupil * 1.05)

        # Add a vertical line that will indicate the current time
        time_indicator = ts_axes.axvline(x=0, color='red', linestyle='-', lw=2)

        # --- Video Writer Setup ---
        output_video_path = output_dir / 'output_analysis_video.mp4'
        w, h = fig.canvas.get_width_height()
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (w, h))

        if not out.isOpened():
            print(f"Error: Could not create the output video file: {output_video_path}")
            cap1.release(); cap2.release(); plt.close(fig)
            return

        # --- Video Creation Loop ---
        frame_count = 0
        while cap1.isOpened() and cap2.isOpened():
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not ret1 or not ret2:
                break

            # Synchronize using the video's timestamp
            current_video_time_sec = cap1.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            # Update video views
            video_axes1.clear(); video_axes2.clear()
            video_axes1.imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)); video_axes1.axis('off'); video_axes1.set_title("Internal View (Eye)")
            video_axes2.imshow(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)); video_axes2.axis('off'); video_axes2.set_title("External View")
            
            # Update only the position of the vertical line
            time_indicator.set_xdata([current_video_time_sec])

            # Capture the plot frame and write it to the video
            fig.canvas.draw()
            img_buf = fig.canvas.buffer_rgba()
            img = np.array(img_buf)
            out.write(cv2.cvtColor(img, cv2.COLOR_RGBA2BGR))
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"  ...processed frame {frame_count} ({current_video_time_sec:.2f}s)")

        # --- Cleanup ---
        cap1.release(); cap2.release(); out.release(); plt.close(fig)
        print(f"Analysis video saved to {output_video_path}")

    except Exception as e:
        print(f"An unexpected error occurred during video creation: {e}")
        import traceback
        traceback.print_exc()

def run_analysis(subj_name='subj_01', data_dir_str='./eyetracking_file', output_dir_str='./results', un_enriched_mode=False, generate_video=True):
    """Main function to run the entire analysis pipeline based on event segments."""
    pd.options.mode.chained_assignment = None
    data_dir, output_dir = Path(data_dir_str), Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_width, video_height = get_video_dimensions(data_dir / 'external.mp4')
    
    try:
        all_data = load_all_data(data_dir, un_enriched_mode)
    except FileNotFoundError as e:
        print(f"Analysis stopped. {e}")
        return
        
    events_df = all_data.get('events')
    if events_df is None or events_df.empty:
        print("Error: events.csv not loaded or empty. Cannot proceed.")
        return
        
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
                print(f"Could not process segment for event '{event_row.get('name', i)}'. Error: {e}")
                traceback.print_exc()
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