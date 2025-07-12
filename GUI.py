# GUI_advanced.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import shutil
import webbrowser
from pathlib import Path
import traceback
import json

# Importa le funzioni orchestratrici principali
try:
    import main_analyzer_advanced as main_analyzer
except ImportError:
    messagebox.showerror("Errore Critico", "File 'main_analyzer_advanced.py' non trovato.")
    exit()

# --- Configurazione File ---
FILE_DESCRIPTIONS = {
    "events.csv": "Select the events CSV file",
    "gaze_enriched.csv": "Select the gaze CSV file (enriched)",
    "fixations_enriched.csv": "Select the enriched fixations CSV file (with surface data)",
    "gaze.csv": "Select the un-enriched gaze CSV file",
    "fixations.csv": "Select the un-enriched fixations CSV file",
    "3d_eye_states.csv": "Select the 3D eye states CSV file (pupil)",
    "blinks.csv": "Select the blinks CSV file",
    "saccades.csv": "Select the saccades CSV file",
    "internal.mp4": "Select the internal video (eye)",
    "external.mp4": "Select the external video (scene)",
    "world_timestamps.csv": "Select the world timestamps CSV",
    "surface_positions.csv": "Select the Marker Mapper surface positions CSV",
}

class SpeedApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SPEED v3.0 - Advanced Output Generator")
        self.root.geometry("720x800")  # Aumentata leggermente la larghezza per la scrollbar
        self.file_entries = {}
        self.plot_vars = {}
        self.video_vars = {}

        # --- Creazione di un Canvas e di una Scrollbar ---
        self.canvas = tk.Canvas(root)
        self.scrollbar = ttk.Scrollbar(root, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Binding della rotellina del mouse
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        
        # --- Tutti i widget ora vanno dentro lo scrollable_frame ---
        main_frame = self.scrollable_frame 

        # --- Sezione Setup ---
        setup_frame = tk.LabelFrame(main_frame, text="1. Project Setup", padx=10, pady=10)
        setup_frame.pack(fill=tk.X, pady=10, padx=10)

        # Nome Partecipante
        name_frame = tk.Frame(setup_frame); name_frame.pack(fill=tk.X, pady=2)
        tk.Label(name_frame, text="Participant Name:", width=20, anchor='w').pack(side=tk.LEFT)
        self.participant_name_var = tk.StringVar(); self.participant_name_var.trace_add("write", self.update_output_dir_default)
        self.name_entry = tk.Entry(name_frame, textvariable=self.participant_name_var); self.name_entry.pack(fill=tk.X, expand=True)

        # Cartella di Output
        output_frame = tk.Frame(setup_frame); output_frame.pack(fill=tk.X, pady=2)
        tk.Label(output_frame, text="Output Folder:", width=20, anchor='w').pack(side=tk.LEFT)
        self.output_dir_entry = tk.Entry(output_frame); self.output_dir_entry.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=(0, 5))
        tk.Button(output_frame, text="Browse...", command=self.select_output_dir).pack(side=tk.RIGHT)

        # --- Sezione File di Input ---
        files_frame = tk.LabelFrame(main_frame, text="2. Input Files", padx=10, pady=10)
        files_frame.pack(fill=tk.X, pady=5, padx=10)
        for i, (std_name, desc) in enumerate(FILE_DESCRIPTIONS.items()):
            row = tk.Frame(files_frame)
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=f"{std_name}:", width=25, anchor='w').pack(side=tk.LEFT)
            entry = tk.Entry(row)
            entry.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=(0, 5))
            self.file_entries[std_name] = entry
            tk.Button(row, text="Browse...", command=lambda e=entry, d=desc: self.select_file(e, d)).pack(side=tk.RIGHT)

        # --- Sezione Analisi Principale ---
        analysis_frame = tk.LabelFrame(main_frame, text="3. Run Core Analysis", padx=10, pady=10)
        analysis_frame.pack(fill=tk.X, pady=5, padx=10)
        self.unenriched_var = tk.BooleanVar(value=False)
        tk.Checkbutton(analysis_frame, text="Analyze un-enriched data only (ignores _enriched files)", variable=self.unenriched_var).pack(anchor='w')
        self.yolo_var = tk.BooleanVar(value=False)
        tk.Checkbutton(analysis_frame, text="Run YOLO Object Detection (requires GPU and specific files)", variable=self.yolo_var).pack(anchor='w')
        tk.Button(analysis_frame, text="RUN CORE ANALYSIS", command=self.run_core_analysis, font=('Helvetica', 10, 'bold'), bg='#c5e1a5').pack(pady=5)

        # --- Notebook per Output ---
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)
        
        plot_tab = tk.Frame(notebook); notebook.add(plot_tab, text='4. Generate Plots')
        video_tab = tk.Frame(notebook); notebook.add(video_tab, text='5. Generate Videos')

        # --- Tab Generazione Grafici ---
        self.setup_plot_tab(plot_tab)
        
        # --- Tab Generazione Video ---
        self.setup_video_tab(video_tab)

    def _on_mousewheel(self, event):
        """Gestisce lo scroll con la rotellina del mouse."""
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def setup_plot_tab(self, parent_tab):
        """Configura l'interfaccia per la generazione dei grafici."""
        plot_options_frame = tk.LabelFrame(parent_tab, text="Plot Options", padx=10, pady=10)
        plot_options_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        plot_types = {
            "path_plots": "Path Plots (Fixation and Gaze)",
            "heatmaps": "Density Heatmaps (Fixation and Gaze)",
            "histograms": "Duration Histograms (Fixations, Blinks, Saccades)",
            "pupillometry": "Pupillometry (Time Series and Spectral Analysis)",
        }
        for key, text in plot_types.items():
            self.plot_vars[key] = tk.BooleanVar(value=True)
            tk.Checkbutton(plot_options_frame, text=text, variable=self.plot_vars[key]).pack(anchor='w')
        
        tk.Button(parent_tab, text="GENERATE SELECTED PLOTS", command=self.run_plot_generation, font=('Helvetica', 10, 'bold'), bg='#90caf9').pack(pady=10)

    def setup_video_tab(self, parent_tab):
        """Configura l'interfaccia per la generazione dei video."""
        video_options_frame = tk.LabelFrame(parent_tab, text="Video Composition Options", padx=10, pady=10)
        video_options_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        video_opts = {
            "crop_to_surface": "Crop video to marker-defined surface",
            "apply_perspective": "Apply perspective correction to surface",
            "overlay_yolo": "Overlay YOLO object detections",
            "overlay_gaze": "Overlay gaze point",
            "overlay_pupil_plot": "Overlay pupillometry plot",
            "include_internal_cam": "Include internal camera view (PiP)",
        }
        for key, text in video_opts.items():
            self.video_vars[key] = tk.BooleanVar(value=False)
            tk.Checkbutton(video_options_frame, text=text, variable=self.video_vars[key]).pack(anchor='w')

        # Nome file video
        tk.Label(video_options_frame, text="\nOutput Video Filename:").pack(anchor='w')
        self.video_filename_var = tk.StringVar(value="video_output_1.mp4")
        tk.Entry(video_options_frame, textvariable=self.video_filename_var).pack(fill=tk.X, pady=5)

        tk.Button(parent_tab, text="GENERATE VIDEO", command=self.run_video_generation, font=('Helvetica', 10, 'bold'), bg='#ef9a9a').pack(pady=10)

    def update_output_dir_default(self, *args):
        subj_name = self.participant_name_var.get().strip()
        if subj_name:
            self.output_dir_entry.delete(0, tk.END)
            self.output_dir_entry.insert(0, str(Path(f'./analysis_results_{subj_name}').resolve()))

    def select_output_dir(self):
        directory_path = filedialog.askdirectory(title="Select Output Folder")
        if directory_path:
            self.output_dir_entry.delete(0, tk.END)
            self.output_dir_entry.insert(0, directory_path)

    def select_file(self, entry_widget, description):
        file_path = filedialog.askopenfilename(title=description)
        if file_path:
            entry_widget.delete(0, tk.END); entry_widget.insert(0, file_path)

    def _get_common_paths(self):
        """Funzione helper per ottenere percorsi comuni e validare."""
        output_dir = self.output_dir_entry.get().strip()
        subj_name = self.participant_name_var.get().strip()
        if not output_dir or not subj_name:
            messagebox.showerror("Errore", "Per favore, inserisci il nome del partecipante e la cartella di output.")
            return None, None, None
        
        output_dir_path = Path(output_dir)
        data_dir_path = output_dir_path / 'eyetracking_file'
        return output_dir_path, data_dir_path, subj_name

    def run_core_analysis(self):
        """Esegue l'analisi di base e la preparazione dei dati."""
        output_dir_path, data_dir_path, subj_name = self._get_common_paths()
        if not output_dir_path: return

        try:
            # Crea le cartelle e copia i file
            data_dir_path.mkdir(parents=True, exist_ok=True)
            input_files = {}
            for name, entry in self.file_entries.items():
                path = entry.get().strip()
                if path:
                    shutil.copy(Path(path), data_dir_path / name)
                    input_files[name] = str(data_dir_path / name)
            
            # Salva la configurazione per un uso futuro
            config = {
                "input_files": input_files,
                "unenriched_mode": self.unenriched_var.get(),
                "yolo_mode": self.yolo_var.get()
            }
            with open(output_dir_path / 'config.json', 'w') as f:
                json.dump(config, f, indent=4)

            messagebox.showinfo("In corso", "Avvio dell'analisi principale. Questo potrebbe richiedere del tempo...")
            main_analyzer.run_core_analysis(
                subj_name=subj_name,
                data_dir_str=str(data_dir_path),
                output_dir_str=str(output_dir_path),
                un_enriched_mode=self.unenriched_var.get(),
                run_yolo=self.yolo_var.get()
            )
            messagebox.showinfo("Successo", "Analisi principale completata. Ora puoi generare grafici e video.")
        except Exception as e:
            messagebox.showerror("Errore di Analisi", f"Si è verificato un errore: {e}\n\n{traceback.format_exc()}")

    def run_plot_generation(self):
        """Esegue la generazione dei grafici selezionati."""
        output_dir_path, _, subj_name = self._get_common_paths()
        if not output_dir_path: return

        plot_selections = {key: var.get() for key, var in self.plot_vars.items()}
        
        try:
            messagebox.showinfo("In corso", "Generazione dei grafici selezionati...")
            main_analyzer.generate_selected_plots(
                output_dir_str=str(output_dir_path),
                subj_name=subj_name,
                plot_selections=plot_selections
            )
            messagebox.showinfo("Successo", f"Grafici generati in {output_dir_path / 'plots'}")
        except Exception as e:
            messagebox.showerror("Errore Grafici", f"Si è verificato un errore: {e}\n\n{traceback.format_exc()}")

    def run_video_generation(self):
        """Esegue la generazione del video con le opzioni selezionate."""
        output_dir_path, _, subj_name = self._get_common_paths()
        if not output_dir_path: return

        video_options = {key: var.get() for key, var in self.video_vars.items()}
        video_options['output_filename'] = self.video_filename_var.get().strip()

        if not video_options['output_filename']:
            messagebox.showerror("Errore", "Per favore, specifica un nome per il file video di output.")
            return

        try:
            messagebox.showinfo("In corso", "Generazione del video personalizzato. Questo processo può essere molto lento...")
            main_analyzer.generate_custom_video(
                output_dir_str=str(output_dir_path),
                subj_name=subj_name,
                video_options=video_options
            )
            messagebox.showinfo("Successo", f"Video salvato in {output_dir_path / video_options['output_filename']}")
        except Exception as e:
            messagebox.showerror("Errore Video", f"Si è verificato un errore: {e}\n\n{traceback.format_exc()}")


if __name__ == '__main__':
    root = tk.Tk()
    app = SpeedApp(root)
    root.mainloop()
