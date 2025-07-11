# main_analyzer.py

import os
from pathlib import Path
import traceback
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

# --- Analysis Module Imports ---
try:
    import speed_script_events as speed_events
    # 'process_gaze_data_v2' is for full-frame YOLO analysis, not used in this version
    # import process_gaze_data_v2 as yolo_events 
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("Analytics and YOLO modules successfully imported.")
except ImportError as e:
    YOLO_AVAILABLE = False
    print(f"WARNING: Unable to import analytics or YOLO modules. Some features may not be available. Error: {e}")

# ==============================================================================
# FUNCTION FOR SURFACE-BASED ANALYSIS
# ==============================================================================

def run_yolo_surface_analysis(data_dir: Path, output_dir: Path, subj_name: str, generate_video: bool, un_enriched_mode: bool):
    """
    Performs YOLO analysis by warping each video frame based on the 'surface_positions.csv' file.
    Handles both enriched and un-enriched modes.
    """
    print("\n>>> RUNNING YOLO ANALYSIS ON MARKER-MAPPED SURFACE...")
    if not YOLO_AVAILABLE:
        print("Library 'ultralytics' not found. Skipping this analysis.")
        return

    # Required file paths
    world_timestamps_path = data_dir / 'world_timestamps.csv'
    surface_positions_path = data_dir / 'surface_positions.csv'
    input_video_path = data_dir / 'external.mp4'
    
    # Select the gaze file based on the mode
    if un_enriched_mode:
        gaze_csv_path = data_dir / 'gaze.csv'
        print("Surface analysis running in UN-ENRICHED mode (using gaze.csv).")
    else:
        gaze_csv_path = data_dir / 'gaze_enriched.csv'
        print("Surface analysis running in ENRICHED mode (using gaze_enriched.csv).")

    for f in [world_timestamps_path, gaze_csv_path, surface_positions_path, input_video_path]:
        if not f.exists():
            raise FileNotFoundError(f"Input file required for surface analysis not found: {f}")

    # 1. Gaze Data Alignment
    print("Timestamp alignment for surface analysis...")
    world_timestamps = pd.read_csv(world_timestamps_path)
    world_timestamps['world_index'] = world_timestamps.index
    gaze_data = pd.read_csv(gaze_csv_path)

    # Prepare the gaze dataframe based on the mode
    if un_enriched_mode:
        gaze_data.rename(columns={'timestamp [ns]': 'gaze_timestamp_ns',
                                  'gaze x [px]': 'gaze_x_px',
                                  'gaze y [px]': 'gaze_y_px'}, inplace=True)
        gaze_to_align = gaze_data[['gaze_timestamp_ns', 'gaze_x_px', 'gaze_y_px']]
    else:
        gaze_on_surface = gaze_data[gaze_data['gaze detected on surface'] == True].copy()
        gaze_on_surface.rename(columns={'gaze position on surface x [normalized]': 'gaze_x_norm', 
                                        'gaze position on surface y [normalized]': 'gaze_y_norm',
                                        'timestamp [ns]': 'gaze_timestamp_ns'}, inplace=True)
        gaze_to_align = gaze_on_surface[['gaze_timestamp_ns', 'gaze_x_norm', 'gaze_y_norm']]

    world_timestamps.rename(columns={'timestamp [ns]': 'world_timestamp_ns'}, inplace=True)
    aligned_data = pd.merge_asof(
        world_timestamps.sort_values('world_timestamp_ns'),
        gaze_to_align.sort_values('gaze_timestamp_ns'),
        left_on='world_timestamp_ns', right_on='gaze_timestamp_ns',
        direction='nearest', tolerance=pd.Timedelta('100ms')
    )
    print("Alignment completed.")

    # 2. Load Surface Data and YOLO Model
    surface_positions = pd.read_csv(surface_positions_path)
    if 'world_index' not in surface_positions.columns:
        surface_positions['world_index'] = surface_positions.index

    print("Loading YOLOv8 model...")
    model = YOLO("yolov8n.pt")
    print("YOLOv8 model loaded.")

    # 3. Video Processing
    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise IOError(f"Error: Unable to open video {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    output_video_path = output_dir / f"surface_analysis_video_{subj_name}.mp4"
    out = None
    output_width, output_height = None, None
    analysis_results = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="Surface Video Analysis")

    frame_count = 0
    processed_frame_count = 0
    while cap.isOpened():
        ret, original_frame = cap.read()
        if not ret:
            pbar.update(total_frames - frame_count)
            break
        pbar.update(1)

        surface_info = surface_positions[surface_positions['world_index'] == frame_count]
        gaze_info = aligned_data[aligned_data['world_index'] == frame_count]

        if surface_info.empty or surface_info[['tl x [px]', 'tl y [px]']].isnull().values.any():
            frame_count += 1
            continue
        
        # 4. Perspective Warping
        try:
            src_pts = np.float32([[surface_info[f'{corner} x [px]'].iloc[0], surface_info[f'{corner} y [px]'].iloc[0]] for corner in ['tl', 'tr', 'br', 'bl']])
            
            if output_width is None or output_height is None:
                output_width = int(max(np.linalg.norm(src_pts[0] - src_pts[1]), np.linalg.norm(src_pts[3] - src_pts[2])))
                output_height = int(max(np.linalg.norm(src_pts[0] - src_pts[3]), np.linalg.norm(src_pts[1] - src_pts[2])))
                if output_width == 0 or output_height == 0:
                    frame_count +=1
                    continue
                if generate_video:
                    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (output_width, output_height))
            
            dst_pts = np.float32([[0, 0], [output_width - 1, 0], [output_width - 1, output_height - 1], [0, output_height - 1]])
            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped_frame = cv2.warpPerspective(original_frame, matrix, (output_width, output_height))
        except Exception as e:
            frame_count += 1
            continue

        # 5. YOLO Analysis on the Warped Frame
        yolo_results = model(warped_frame, verbose=False)
        
        gaze_x_norm, gaze_y_norm = np.nan, np.nan
        if not gaze_info.empty:
            if un_enriched_mode:
                gaze_x_px = gaze_info.iloc[0].get('gaze_x_px')
                gaze_y_px = gaze_info.iloc[0].get('gaze_y_px')
                if pd.notna(gaze_x_px) and pd.notna(gaze_y_px):
                    # Transform gaze coordinates from pixels to warped surface coordinates
                    gaze_point_px = np.float32([[[gaze_x_px, gaze_y_px]]])
                    warped_gaze_point = cv2.perspectiveTransform(gaze_point_px, matrix)
                    warped_x = warped_gaze_point[0][0][0]
                    warped_y = warped_gaze_point[0][0][1]
                    gaze_x_norm = warped_x / output_width
                    gaze_y_norm = warped_y / output_height
            else:
                gaze_x_norm = gaze_info.iloc[0].get('gaze_x_norm')
                gaze_y_norm = gaze_info.iloc[0].get('gaze_y_norm')

        frame_detections = []
        for r in yolo_results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                cls_id, conf, class_name = int(box.cls[0]), float(box.conf[0]), model.names[int(box.cls[0])]
                
                if generate_video and out:
                    cv2.rectangle(warped_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    cv2.putText(warped_frame, f"{class_name} {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                gaze_in_box = False
                if pd.notna(gaze_x_norm):
                    gaze_px_on_surface = int(gaze_x_norm * output_width)
                    gaze_py_on_surface = int(gaze_y_norm * output_height)
                    if (x1 <= gaze_px_on_surface <= x2) and (y1 <= gaze_py_on_surface <= y2):
                        gaze_in_box = True

                frame_detections.append({
                    'frame_input': frame_count, 'frame_output': processed_frame_count, 'gaze_x_norm': gaze_x_norm, 'gaze_y_norm': gaze_y_norm,
                    'object_class': class_name, 'confidence': conf, 'bbox_x1_norm': float(x1/output_width), 'bbox_y1_norm': float(y1/output_height),
                    'bbox_x2_norm': float(x2/output_width), 'bbox_y2_norm': float(y2/output_height), 'gaze_in_box': gaze_in_box
                })
        
        if not frame_detections and pd.notna(gaze_x_norm):
             frame_detections.append({'frame_input': frame_count, 'frame_output': processed_frame_count, 'gaze_x_norm': gaze_x_norm, 'gaze_y_norm': gaze_y_norm})
        
        analysis_results.extend(frame_detections)
        if generate_video and out:
            if pd.notna(gaze_x_norm):
                cv2.drawMarker(warped_frame, (int(gaze_x_norm * output_width), int(gaze_y_norm * output_height)), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            out.write(warped_frame)
        
        processed_frame_count += 1
        frame_count += 1

    pbar.close()
    cap.release()
    if out: out.release()
    cv2.destroyAllWindows()
    
    if analysis_results:
        results_df = pd.DataFrame(analysis_results)
        output_csv_path = output_dir / f"surface_analysis_data_{subj_name}.csv"
        results_df.to_csv(output_csv_path, index=False, float_format='%.4f')
        print(f"Surface analysis data saved to {output_csv_path}")
    if generate_video:
        print(f"Surface analysis video saved to {output_video_path}")


# ==============================================================================
# MAIN ORCHESTRATOR FUNCTION
# ==============================================================================

def run_full_analysis(subj_name: str, data_dir_str: str, output_dir_str: str, options: dict):
    """
    Orchestrates the entire analysis pipeline by calling the imported functions.
    """
    print(f"--- STARTING FULL ANALYSIS FOR SUBJECT: {subj_name} ---")
    
    data_dir = Path(data_dir_str)
    output_dir = Path(output_dir_str)
    un_enriched_mode = options.get("un_enriched_mode", False)
    
    # --- 1. STANDARD EVENT-BASED ANALYSIS ---
    if options.get("run_standard"):
        print("\n>>> RUNNING STANDARD EVENT-BASED ANALYSIS...")
        try:
            speed_events.run_analysis(
                subj_name=subj_name,
                data_dir_str=str(data_dir),
                output_dir_str=str(output_dir),
                un_enriched_mode=un_enriched_mode,
                generate_video=options.get("generate_standard_video")
            )
            print(">>> Standard analysis finished.")
        except Exception as e:
            print(f"!!! Error during standard analysis: {e}")
            traceback.print_exc()

    # --- 2. YOLO-BASED ANALYSIS ---
    if options.get("run_yolo"):
        if options.get("run_surface_yolo"):
            # Surface Mode
            try:
                run_yolo_surface_analysis(
                    data_dir=data_dir,
                    output_dir=output_dir,
                    subj_name=subj_name,
                    generate_video=options.get("generate_surface_video"),
                    un_enriched_mode=un_enriched_mode
                )
            except Exception as e:
                print(f"!!! Error during YOLO surface analysis: {e}")
                traceback.print_exc()
        else:
            # Full-frame Mode (placeholder)
            print("\n>>> RUNNING FULL-FRAME YOLO ANALYSIS...")
            print("NOTE: Full-frame YOLO analysis is not the primary feature of this version.")
            print("Surface analysis will be performed if selected.")

    print(f"\n--- FULL ANALYSIS FOR {subj_name} COMPLETED ---")
