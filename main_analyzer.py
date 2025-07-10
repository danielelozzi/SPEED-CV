# main_analyzer.py

import os
from pathlib import Path
import traceback
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

# --- IMPORTAZIONE DEI MODULI DI ANALISI ---
try:
    import speed_script_events as speed_events
    import process_gaze_data_v2 as yolo_events
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("Moduli di analisi e YOLO importati con successo.")
except ImportError as e:
    YOLO_AVAILABLE = False
    print(f"ATTENZIONE: Impossibile importare i moduli di analisi o YOLO. Alcune funzionalità potrebbero non essere disponibili. Errore: {e}")

# ==============================================================================
# NUOVA FUNZIONE PER ANALISI SU SUPERFICIE
# ==============================================================================

def run_yolo_surface_analysis(data_dir: Path, output_dir: Path, subj_name: str, generate_video: bool):
    """
    Esegue l'analisi YOLO ritagliando e raddrizzando ogni frame del video
    in base al file 'surface_positions.csv'.
    """
    print("\n>>> ESECUZIONE ANALISI YOLO SU SUPERFICIE MARKER-MAPPED...")
    if not YOLO_AVAILABLE:
        print("Libreria 'ultralytics' non trovata. Salto questa analisi.")
        return

    # Percorsi dei file necessari
    world_timestamps_path = data_dir / 'world_timestamps.csv'
    gaze_csv_path = data_dir / 'gaze_enriched.csv'
    surface_positions_path = data_dir / 'surface_positions.csv'
    input_video_path = data_dir / 'external.mp4'

    for f in [world_timestamps_path, gaze_csv_path, surface_positions_path, input_video_path]:
        if not f.exists():
            raise FileNotFoundError(f"File di input richiesto per l'analisi su superficie non trovato: {f}")

    # 1. Allineamento Dati Gaze
    print("Allineamento timestamp per analisi su superficie...")
    world_timestamps = pd.read_csv(world_timestamps_path)
    world_timestamps['world_index'] = world_timestamps.index
    gaze = pd.read_csv(gaze_csv_path)
    gaze_on_surface = gaze[gaze['gaze detected on surface'] == True].copy()
    gaze_on_surface.rename(columns={'gaze position on surface x [normalized]': 'gaze_x_norm', 
                                    'gaze position on surface y [normalized]': 'gaze_y_norm',
                                    'timestamp [ns]': 'gaze_timestamp_ns'}, inplace=True)
    world_timestamps.rename(columns={'timestamp [ns]': 'world_timestamp_ns'}, inplace=True)
    aligned_data = pd.merge_asof(
        world_timestamps.sort_values('world_timestamp_ns'),
        gaze_on_surface.sort_values('gaze_timestamp_ns'),
        left_on='world_timestamp_ns', right_on='gaze_timestamp_ns',
        direction='nearest', tolerance=pd.Timedelta('100ms')
    )
    print("Allineamento completato.")

    # 2. Caricamento Dati Superficie e Modello YOLO
    surface_positions = pd.read_csv(surface_positions_path)
    if 'world_index' not in surface_positions.columns:
        surface_positions['world_index'] = surface_positions.index

    print("Caricamento modello YOLOv8...")
    model = YOLO("yolov8n.pt")
    print("Modello YOLOv8 caricato.")

    # 3. Processamento Video
    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise IOError(f"Errore: Impossibile aprire il video {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    output_video_path = output_dir / f"surface_analysis_video_{subj_name}.mp4"
    out = None
    output_width, output_height = None, None
    analysis_results = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="Analisi Video su Superficie")

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
        
        # 4. Raddrizzamento Prospettico
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

        # 5. Analisi YOLO sul frame raddrizzato
        yolo_results = model(warped_frame, verbose=False)
        
        gaze_x_norm, gaze_y_norm = (gaze_info.iloc[0]['gaze_x_norm'], gaze_info.iloc[0]['gaze_y_norm']) if not gaze_info.empty else (np.nan, np.nan)
        
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
                    gaze_px, gaze_py = int(gaze_x_norm * output_width), int(gaze_y_norm * output_height)
                    if (x1 <= gaze_px <= x2) and (y1 <= gaze_py <= y2):
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
        print(f"Dati di analisi su superficie salvati in {output_csv_path}")
    if generate_video:
        print(f"Video di analisi su superficie salvato in {output_video_path}")


# ==============================================================================
# FUNZIONE ORCHESTRATRICE PRINCIPALE
# ==============================================================================

def run_full_analysis(subj_name: str, data_dir_str: str, output_dir_str: str, options: dict):
    """
    Orchestra l'intera pipeline di analisi chiamando le funzioni importate.
    """
    print(f"--- AVVIO ANALISI COMPLETA PER IL SOGGETTO: {subj_name} ---")
    
    data_dir = Path(data_dir_str)
    output_dir = Path(output_dir_str)
    
    # --- 1. ANALISI STANDARD BASATA SU EVENTI ---
    if options.get("run_standard"):
        print("\n>>> ESECUZIONE ANALISI STANDARD BASATA SU EVENTI...")
        try:
            speed_events.run_analysis(
                subj_name=subj_name,
                data_dir_str=str(data_dir),
                output_dir_str=str(output_dir),
                un_enriched_mode=False,
                generate_video=options.get("generate_standard_video")
            )
            print(">>> Analisi standard completata.")
        except Exception as e:
            print(f"!!! Errore durante l'analisi standard: {e}")
            traceback.print_exc()

    # --- 2. ANALISI CON YOLO ---
    if options.get("run_yolo"):
        if options.get("run_surface_yolo"):
            # Modalità su superficie
            try:
                run_yolo_surface_analysis(
                    data_dir=data_dir,
                    output_dir=output_dir,
                    subj_name=subj_name,
                    generate_video=options.get("generate_surface_video")
                )
            except Exception as e:
                print(f"!!! Errore durante l'analisi YOLO su superficie: {e}")
                traceback.print_exc()
        else:
            # Modalità su frame intero (utilizzando il modulo yolo_events)
            print("\n>>> ESECUZIONE ANALISI YOLO SU FRAME INTERO...")
            try:
                yolo_output_dir = output_dir / "yolo_fullframe_analysis"
                yolo_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Esegui la funzione main da process_gaze_data_v2 ma con argomenti passati programmaticamente
                # (Questo richiede di modificare process_gaze_data_v2 per accettare argomenti invece di usare argparse)
                # Per semplicità, qui simuliamo la chiamata. L'implementazione completa richiederebbe una refactorizzazione
                # di process_gaze_data_v2.py per trasformare il suo blocco main in una funzione chiamabile.
                print("NOTA: L'analisi YOLO su frame intero richiede una refactorizzazione di 'process_gaze_data_v2.py'.")
                print("Questa funzionalità è simulata. Verrà eseguita l'analisi su superficie se selezionata.")

            except Exception as e:
                print(f"!!! Errore durante l'analisi YOLO su frame intero: {e}")
                traceback.print_exc()

    print(f"\n--- ANALISI COMPLETA PER {subj_name} TERMINATA ---")