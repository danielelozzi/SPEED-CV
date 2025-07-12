# video_generator.py
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

def create_custom_video(data_dir: Path, output_dir: Path, subj_name: str, options: dict, un_enriched_mode: bool):
    """
    Crea un video di analisi altamente personalizzato basato sulle opzioni fornite.
    """
    # --- Percorsi dei file di input ---
    external_video_path = data_dir / 'external.mp4'
    internal_video_path = data_dir / 'internal.mp4' if options.get("include_internal_cam") else None
    gaze_path = data_dir / ('gaze.csv' if un_enriched_mode else 'gaze_enriched.csv')
    pupil_path = data_dir / '3d_eye_states.csv'
    surface_path = data_dir / 'surface_positions.csv' if options.get("crop_to_surface") else None
    world_timestamps_path = data_dir / 'world_timestamps.csv'

    # --- Validazione dei file necessari ---
    if not external_video_path.exists():
        raise FileNotFoundError(f"Video esterno non trovato: {external_video_path}")
    if not gaze_path.exists():
        raise FileNotFoundError(f"File Gaze non trovato: {gaze_path}")

    # --- Setup Video ---
    cap_ext = cv2.VideoCapture(str(external_video_path))
    cap_int = cv2.VideoCapture(str(internal_video_path)) if internal_video_path and internal_video_path.exists() else None
    
    fps = cap_ext.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap_ext.get(cv2.CAP_PROP_FRAME_COUNT))

    # --- Caricamento e allineamento dati ---
    print("Loading and aligning data for video generation...")
    world_timestamps = pd.read_csv(world_timestamps_path)
    gaze_data = pd.read_csv(gaze_path)
    
    # Allinea gaze data con i frame del video
    gaze_data.rename(columns={'timestamp [ns]': 'gaze_timestamp_ns'}, inplace=True)
    world_timestamps.rename(columns={'timestamp [ns]': 'world_timestamp_ns'}, inplace=True)
    aligned_gaze = pd.merge_asof(
        world_timestamps.sort_values('world_timestamp_ns'),
        gaze_data.sort_values('gaze_timestamp_ns'),
        left_on='world_timestamp_ns', right_on='gaze_timestamp_ns',
        direction='nearest', tolerance=pd.Timedelta('100ms')
    )
    
    surface_data = pd.read_csv(surface_path) if surface_path and surface_path.exists() else None

    # --- Loop Principale di Generazione Video ---
    output_video_path = output_dir / options.get('output_filename', 'video_output.mp4')
    out_writer = None

    pbar = tqdm(total=total_frames, desc="Generating Custom Video")
    for frame_idx in range(total_frames):
        ret_ext, frame_ext = cap_ext.read()
        if not ret_ext:
            break

        current_surface = surface_data[surface_data['world_index'] == frame_idx].iloc[0] if surface_data is not None and frame_idx in surface_data['world_index'].values else None
        
        # --- 1. Ritaglio e Correzione Prospettica ---
        if options.get("crop_to_surface") and current_surface is not None and not current_surface[['tl x [px]']].isnull().values.any():
            src_pts = np.float32([[current_surface[f'{c} x [px]'], current_surface[f'{c} y [px]']] for c in ['tl', 'tr', 'br', 'bl']])
            
            w = int(max(np.linalg.norm(src_pts[0] - src_pts[1]), np.linalg.norm(src_pts[3] - src_pts[2])))
            h = int(max(np.linalg.norm(src_pts[0] - src_pts[3]), np.linalg.norm(src_pts[1] - src_pts[2])))
            
            if w == 0 or h == 0:
                pbar.update(1)
                continue

            if options.get("apply_perspective"):
                dst_pts = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
                matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                final_frame = cv2.warpPerspective(frame_ext, matrix, (w, h))
            else:
                x, y, w_crop, h_crop = cv2.boundingRect(src_pts)
                final_frame = frame_ext[y:y+h_crop, x:x+w_crop]
        else:
            final_frame = frame_ext

        if out_writer is None:
            h, w, _ = final_frame.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (w, h))

        # --- 2. Overlay degli Elementi ---
        current_gaze = aligned_gaze[aligned_gaze['world_index'] == frame_idx].iloc[0] if frame_idx in aligned_gaze['world_index'].values else None
        
        if options.get("overlay_gaze") and current_gaze is not None:
            # La logica per proiettare il gaze deve considerare se il frame è stato deformato
            # Questa è una semplificazione.
            gaze_x = current_gaze.get('gaze x [px]')
            gaze_y = current_gaze.get('gaze y [px]')
            if pd.notna(gaze_x) and pd.notna(gaze_y):
                 # Qui servirebbe la trasformazione se il frame è stato deformato
                 cv2.drawMarker(final_frame, (int(gaze_x), int(gaze_y)), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

        if options.get("overlay_yolo"):
            # Qui andrebbe la logica per caricare i risultati YOLO per il frame e disegnarli
            cv2.putText(final_frame, "YOLO Overlay", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if options.get("include_internal_cam") and cap_int is not None:
            ret_int, frame_int = cap_int.read()
            if ret_int:
                h, w, _ = final_frame.shape
                pip_h, pip_w = h // 4, w // 4
                frame_int_resized = cv2.resize(frame_int, (pip_w, pip_h))
                final_frame[h-pip_h-10:h-10, w-pip_w-10:w-10] = frame_int_resized

        if options.get("overlay_pupil_plot"):
            # Logica per disegnare il grafico della pupillometria
             cv2.putText(final_frame, "Pupil Plot", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)


        # --- Scrittura del Frame ---
        # Ridimensiona se necessario per adattarsi al writer
        h_out, w_out = out_writer.get(cv2.CAP_PROP_FRAME_HEIGHT), out_writer.get(cv2.CAP_PROP_FRAME_WIDTH)
        if final_frame.shape[0] != h_out or final_frame.shape[1] != w_out:
            final_frame = cv2.resize(final_frame, (w_out, h_out))

        out_writer.write(final_frame)
        pbar.update(1)

    # --- Pulizia ---
    pbar.close()
    cap_ext.release()
    if cap_int: cap_int.release()
    if out_writer: out_writer.release()
    cv2.destroyAllWindows()
    print(f"Video generation complete. Saved to {output_video_path}")

