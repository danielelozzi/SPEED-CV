# process_gaze_data_v2.py

import pandas as pd
import numpy as np
import torch
import pickle
import os
import argparse
from ultralytics import YOLO
from collections import defaultdict
from tqdm import tqdm
import cv2
import gc

# ==============================================================================
# FUNZIONI HELPER (Invariate rispetto alla versione precedente)
# ==============================================================================

def setup_directories(dirs):
    """Crea le directory specificate se non esistono."""
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Cartella '{d}' creata.")

def read_pickle(file_path):
    """Legge un singolo file pickle."""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except (FileNotFoundError, pickle.UnpicklingError, EOFError) as e:
        # Silently handle cases where pickle is missing or corrupt
        return None

def save_pickle(obj, file_path):
    """Salva un oggetto Python in un file pickle."""
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
    except Exception as e:
        print(f"Errore durante il salvataggio dell'oggetto in {file_path}: {e}")

def align_data(gaze_path, world_path, pupil_path, events_path):
    """
    Allinea i dati da vari file CSV basandosi sui timestamp.
    """
    print("Allineamento dei dati in corso...")
    external = pd.read_csv(world_path)
    external.rename(columns={'timestamp [ns]': 'timestamp [ns] external'}, inplace=True)
    external['index_external'] = range(1, len(external) + 1)

    gaze = pd.read_csv(gaze_path)
    gaze.rename(columns={'timestamp [ns]': 'timestamp [ns] gaze'}, inplace=True)
    
    pupil = pd.read_csv(pupil_path)
    pupil.rename(columns={'timestamp [ns]': 'timestamp [ns] pupil'}, inplace=True)
    
    events = pd.read_csv(events_path)
    events.rename(columns={'timestamp [ns]': 'timestamp [ns] events'}, inplace=True)

    df_merged = pd.merge_asof(gaze, external, left_on='timestamp [ns] gaze', right_on='timestamp [ns] external', direction='nearest')
    df_merged = pd.merge_asof(df_merged, pupil, left_on='timestamp [ns] gaze', right_on='timestamp [ns] pupil', direction='nearest')
    
    agg_functions = {
        'gaze x [px]': 'mean', 'gaze y [px]': 'mean', 'pupil diameter left [mm]': 'mean',
        'pupil diameter right [mm]': 'mean', 'timestamp [ns] gaze': 'mean',
        'recording id': 'first', 'timestamp [ns] external': 'first'
    }
    mean_timestamps = df_merged.groupby('index_external').agg(agg_functions).reset_index()

    mean_timestamps = mean_timestamps.sort_values('timestamp [ns] external')
    events = events.sort_values('timestamp [ns] events')
    final_df = pd.merge_asof(mean_timestamps, events, left_on='timestamp [ns] external', right_on='timestamp [ns] events', direction='backward')

    final_df['gaze x [px]'] = final_df['gaze x [px]'] / 1600.0
    final_df['gaze y [px]'] = final_df['gaze y [px]'] / 1200.0

    print("Allineamento dati completato.")
    return final_df.dropna(subset=['name'])

# ==============================================================================
# CLASSE PRINCIPALE PER L'ANALISI
# ==============================================================================

class GazeAnalyzer:
    def __init__(self, video_path, aligned_data_df, data_dir='data'):
        self.video_path = video_path
        self.aligned_data = aligned_data_df
        self.data_dir = data_dir
        # Selezione del dispositivo con priorità: CUDA > MPS (Apple Silicon) > CPU
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        print(f"Utilizzo del dispositivo: {self.device}")
        self.model = YOLO("yolov8n.pt")
        self.model.to(self.device)
        self.model_classes = self.model.names

    def generate_annotated_video(self, output_path):
        """Genera un video con annotazioni di gaze e bounding box."""
        print(f"Inizio generazione video annotato: '{output_path}'...")

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Errore: Impossibile aprire il video sorgente {self.video_path}")
            return

        # Ottieni proprietà del video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Setup del video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Mappa per un accesso rapido ai dati allineati per frame
        data_map = {row['index_external'] - 1: row for _, row in self.aligned_data.iterrows()}

        for frame_idx in tqdm(range(frame_count), desc="Generazione Video"):
            ret, frame = cap.read()
            if not ret:
                break

            # Carica e disegna dati YOLO per il frame corrente
            pickle_path = os.path.join(self.data_dir, f"{frame_idx}.pickle")
            yolo_data = read_pickle(pickle_path)
            if yolo_data and yolo_data.boxes and yolo_data.boxes.id is not None:
                for i in range(len(yolo_data.boxes.id)):
                    xmin, ymin, xmax, ymax = yolo_data.boxes.xyxy[i]
                    class_idx = int(yolo_data.boxes.cls[i])
                    class_name = self.model_classes[class_idx]
                    object_id = int(yolo_data.boxes.id[i])
                    label = f"{class_name} ID:{object_id}"
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)
                    cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Disegna il punto del gaze se presente
            if frame_idx in data_map and pd.notna(data_map[frame_idx]['gaze x [px]']):
                gaze_x = int(data_map[frame_idx]['gaze x [px]'] * width)
                gaze_y = int(data_map[frame_idx]['gaze y [px]'] * height)
                cv2.drawMarker(frame, (gaze_x, gaze_y), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

            out.write(frame)

        cap.release()
        out.release()
        print(f"Video annotato salvato con successo in '{output_path}'.")

    def run_yolo_tracking(self):
        """Esegue il tracking YOLO sul video e salva i risultati per ogni frame."""
        print(f"Avvio del tracking YOLO su '{self.video_path}'...")
        if len(os.listdir(self.data_dir)) >= len(self.aligned_data['index_external'].unique()):
            print("I dati di tracking YOLO sembrano già esistere. Salto questa fase.")
            return
            
        stream = self.model.track(self.video_path, stream=True, verbose=False)
        
        for n_frame, track_frame in tqdm(enumerate(stream), desc="Tracking Video"):
            try:
                track_frame.orig_img = None
                file_name = os.path.join(self.data_dir, f"{n_frame}.pickle")
                save_pickle(track_frame, file_name)
            except Exception as e:
                print(f'Errore al frame {n_frame}: {e}')
        gc.collect()

    def analyze_fixations_and_stats(self):
        """
        Analizza le fissazioni e calcola le statistiche per classe e per istanza.
        """
        print("Analisi delle fissazioni e calcolo delle statistiche...")
        
        # Strutture dati per accumulare le statistiche
        # defaultdict annidati per creare automaticamente le chiavi mancanti
        stats_per_class = defaultdict(lambda: defaultdict(lambda: {'detection_count': 0, 'fixation_count': 0, 'pupil_diameters': []}))
        stats_per_instance = defaultdict(lambda: defaultdict(lambda: {'detection_count': 0, 'fixation_count': 0, 'pupil_diameters': []}))
        instance_to_class_map = {}

        for _, row in tqdm(self.aligned_data.iterrows(), total=self.aligned_data.shape[0], desc="Analisi Frame"):
            event_name = row['name']
            frame_idx = row['index_external'] - 1
            
            pickle_path = os.path.join(self.data_dir, f"{frame_idx}.pickle")
            yolo_data = read_pickle(pickle_path)
            
            # Controlla se ci sono dati di tracking validi. L'attributo .id è None se il tracker non ha assegnato ID.
            if yolo_data is None or yolo_data.boxes is None or yolo_data.boxes.id is None:
                continue

            # Aggiorna conteggio rilevazioni per ogni oggetto presente nel frame
            detected_ids_in_frame = set()
            for i in range(len(yolo_data.boxes.id)):
                class_idx = int(yolo_data.boxes.cls[i])
                class_name = self.model_classes[class_idx]
                object_id = int(yolo_data.boxes.id[i])
                instance_id = f"{class_name}_{object_id}"

                if instance_id not in instance_to_class_map:
                    instance_to_class_map[instance_id] = class_name

                stats_per_class[event_name][class_name]['detection_count'] += 1
                if instance_id not in detected_ids_in_frame:
                     stats_per_instance[event_name][instance_id]['detection_count'] += 1
                     detected_ids_in_frame.add(instance_id)


            # Controlla fissazione
            gaze_x = row['gaze x [px]']
            gaze_y = row['gaze y [px]']
            pupil_diameter = row['pupil diameter left [mm]']

            if pd.isna(gaze_x) or pd.isna(pupil_diameter):
                continue

            for i in range(len(yolo_data.boxes.xyxyn)):
                xmin, ymin, xmax, ymax = yolo_data.boxes.xyxyn[i]
                if xmin <= gaze_x <= xmax and ymin <= gaze_y <= ymax:
                    class_idx = int(yolo_data.boxes.cls[i])
                    class_name = self.model_classes[class_idx]
                    object_id = int(yolo_data.boxes.id[i])
                    instance_id = f"{class_name}_{object_id}"
                    
                    # Accumula dati per la fissazione
                    stats_per_class[event_name][class_name]['fixation_count'] += 1
                    stats_per_class[event_name][class_name]['pupil_diameters'].append(pupil_diameter)
                    
                    stats_per_instance[event_name][instance_id]['fixation_count'] += 1
                    stats_per_instance[event_name][instance_id]['pupil_diameters'].append(pupil_diameter)
                    
                    break # Passa al frame successivo dopo aver trovato la prima fissazione

        print("Analisi completata. Generazione delle tabelle...")
        return stats_per_class, stats_per_instance, instance_to_class_map

    def create_results_tables(self, stats_per_class, stats_per_instance, instance_to_class_map):
        """Crea due DataFrame finali con le statistiche calcolate."""
        
        # Tabella 1: Per Classe
        event_names = sorted(stats_per_class.keys())
        all_classes = sorted(set(cls for events in stats_per_class.values() for cls in events.keys()))
        
        class_data = []
        for event in event_names:
            row = {'evento': event}
            for cls in all_classes:
                stats = stats_per_class[event][cls]
                fixations = stats['fixation_count']
                detections = stats['detection_count']
                pupil_diams = stats['pupil_diameters']
                
                row[f'{cls}_fissazioni_normalizzate'] = fixations / detections if detections > 0 else 0
                row[f'{cls}_diametro_pupilla_medio'] = np.mean(pupil_diams) if pupil_diams else np.nan
            class_data.append(row)
        df_class = pd.DataFrame(class_data)

        # Tabella 2: Per Istanza
        all_instances = sorted(set(inst for events in stats_per_instance.values() for inst in events.keys()))

        instance_data = []
        for event in event_names:
            row = {'evento': event}
            for inst_id in all_instances:
                stats = stats_per_instance[event][inst_id]
                fixations = stats['fixation_count']
                detections = stats['detection_count']
                pupil_diams = stats['pupil_diameters']

                row[f'{inst_id}_fissazioni_normalizzate'] = fixations / detections if detections > 0 else 0
                row[f'{inst_id}_diametro_pupilla_medio'] = np.mean(pupil_diams) if pupil_diams else np.nan
            instance_data.append(row)
        df_instance = pd.DataFrame(instance_data)

        # Tabella 3: Mappa ID -> Classe
        id_map_data = list(instance_to_class_map.items())
        df_id_map = pd.DataFrame(id_map_data, columns=['instance_id', 'classe_oggetto'])
        df_id_map = df_id_map.sort_values(by=['classe_oggetto', 'instance_id']).reset_index(drop=True)

        return df_class, df_instance, df_id_map

# ==============================================================================
# BLOCCO PRINCIPALE
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analizza dati di gaze tracking e YOLO.")
    parser.add_argument("--video", required=True, help="Percorso del file video.")
    parser.add_argument("--gaze_csv", required=True, help="Percorso del file gaze.csv.")
    parser.add_argument("--world_csv", required=True, help="Percorso del file world_timestamps.csv.")
    parser.add_argument("--pupil_csv", required=True, help="Percorso del file 3d_eye_states.csv.")
    parser.add_argument("--events_csv", required=True, help="Percorso del file events.csv.")
    parser.add_argument("--output_class_csv", default="statistiche_per_classe.csv", help="Nome del file CSV per le statistiche per classe.")
    parser.add_argument("--output_instance_csv", default="statistiche_per_istanza.csv", help="Nome del file CSV per le statistiche per istanza.")
    parser.add_argument("--output_id_map_csv", default="mappa_id_classe.csv", help="Nome del file CSV per la mappa ID-classe.")
    parser.add_argument("--generate_video", action='store_true', help="Abilita la generazione di un video con annotazioni (gaze e bounding box).")
    parser.add_argument("--output_video", default="output_annotated.mp4", help="Percorso del file video di output con annotazioni.")
    
    args = parser.parse_args()

    try:
        # FASE 0: Setup
        DATA_DIR = "./data"
        setup_directories([DATA_DIR])

        # FASE 1: Allineamento Dati
        aligned_df = align_data(
            gaze_path=args.gaze_csv, world_path=args.world_csv,
            pupil_path=args.pupil_csv, events_path=args.events_csv
        )
        
        # FASE 2: Analisi Video e Fissazioni
        analyzer = GazeAnalyzer(video_path=args.video, aligned_data_df=aligned_df, data_dir=DATA_DIR)
        analyzer.run_yolo_tracking()
        
        # FASE 3: Calcolo Statistiche
        stats_class, stats_instance, id_map = analyzer.analyze_fixations_and_stats()
        df_class_results, df_instance_results, df_id_map_results = analyzer.create_results_tables(stats_class, stats_instance, id_map)
        
        # FASE 4: Output
        df_class_results.to_csv(args.output_class_csv, index=False, float_format='%.4f')
        df_instance_results.to_csv(args.output_instance_csv, index=False, float_format='%.4f')
        df_id_map_results.to_csv(args.output_id_map_csv, index=False)
        
        print("\n--- Statistiche per Classe ---")
        print(df_class_results)
        print(f"\nRisultati salvati con successo in '{args.output_class_csv}'")
        
        print("\n--- Statistiche per Istanza (Classe + ID) ---")
        print(df_instance_results)
        print(f"\nRisultati salvati con successo in '{args.output_instance_csv}'")

        print("\n--- Mappa ID Oggetto -> Classe ---")
        print(df_id_map_results)
        print(f"\nRisultati salvati con successo in '{args.output_id_map_csv}'")

        # FASE 5: Generazione Video (Opzionale)
        if args.generate_video:
            analyzer.generate_annotated_video(output_path=args.output_video)

    except Exception as e:
        print(f"\nSi è verificato un errore: {e}")
        import traceback
        traceback.print_exc()
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Pulizia della memoria completata.")

if __name__ == "__main__":
    main()