# main_analyzer_advanced.py

from pathlib import Path
import traceback
import pandas as pd
import json

# Importa i moduli di analisi e generazione
import speed_script_events as speed_events
import video_generator

def run_core_analysis(subj_name: str, data_dir_str: str, output_dir_str: str, un_enriched_mode: bool, run_yolo: bool):
    """
    Esegue solo l'analisi dei dati e salva i risultati intermedi per un uso successivo.
    """
    print(f"--- STARTING CORE ANALYSIS FOR SUBJECT: {subj_name} ---")
    data_dir = Path(data_dir_str)
    output_dir = Path(output_dir_str)
    processed_data_dir = output_dir / 'processed_data'
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Analisi Standard basata su eventi ---
    print("\n>>> RUNNING STANDARD EVENT-BASED ANALYSIS (DATA ONLY)...")
    try:
        # Esegue l'analisi che calcola le statistiche e le salva
        speed_events.run_analysis(
            subj_name=subj_name,
            data_dir_str=str(data_dir),
            output_dir_str=str(output_dir),
            un_enriched_mode=un_enriched_mode,
            generate_video=False,  # Non generare il video standard qui
            generate_plots=False # Non generare i grafici qui
        )
        print(">>> Standard data analysis finished.")
    except Exception as e:
        print(f"!!! Error during standard data analysis: {e}")
        traceback.print_exc()
        raise

    # --- 2. Analisi YOLO (se richiesta) ---
    if run_yolo:
        print("\n>>> RUNNING YOLO OBJECT DETECTION (DATA ONLY)...")
        # Questa è una semplificazione. Un'implementazione reale richiederebbe
        # di eseguire il tracking YOLO e salvare i risultati per frame.
        # Per ora, simuliamo questo passaggio.
        print("NOTE: YOLO analysis is a placeholder in this version.")
        # In un'implementazione completa, qui si eseguirebbe il tracking e si salverebbero
        # i risultati in `processed_data_dir`
    
    print(f"\n--- CORE ANALYSIS FOR {subj_name} COMPLETED ---")
    print(f"Processed data saved in: {processed_data_dir}")

def generate_selected_plots(output_dir_str: str, subj_name: str, plot_selections: dict):
    """
    Genera solo i grafici selezionati dall'utente, caricando i dati necessari.
    """
    print("\n>>> GENERATING SELECTED PLOTS...")
    output_dir = Path(output_dir_str)
    
    # Carica la configurazione per sapere se l'analisi era arricchita o meno
    with open(output_dir / 'config.json', 'r') as f:
        config = json.load(f)
    
    # Chiama una funzione dedicata in speed_script_events per generare i grafici
    speed_events.generate_plots_on_demand(
        output_dir_str=str(output_dir),
        subj_name=subj_name,
        plot_selections=plot_selections,
        un_enriched_mode=config.get("unenriched_mode", False)
    )
    print(">>> Plot generation finished.")

def generate_custom_video(output_dir_str: str, subj_name: str, video_options: dict):
    """
    Chiama il generatore di video con la configurazione specificata.
    """
    print("\n>>> GENERATING CUSTOM VIDEO...")
    output_dir = Path(output_dir_str)
    
    # Carica la configurazione
    with open(output_dir / 'config.json', 'r') as f:
        config = json.load(f)
        
    video_generator.create_custom_video(
        data_dir=output_dir / 'eyetracking_file',
        output_dir=output_dir,
        subj_name=subj_name,
        options=video_options,
        un_enriched_mode=config.get("unenriched_mode", False)
    )
    print(">>> Video generation finished.")

