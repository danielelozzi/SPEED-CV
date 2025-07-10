# GUI.py

import tkinter as tk
from tkinter import filedialog, messagebox
import os
import shutil
import webbrowser
from pathlib import Path
import traceback

# Importa la funzione orchestratrice principale dal nuovo script
try:
    import main_analyzer
except ImportError:
    messagebox.showerror("Errore Critico", "File 'main_analyzer.py' non trovato. Assicurati che sia nella stessa cartella di GUI.py.")
    exit()


# --- Configurazione ---
REQUIRED_FILES = {
    "events.csv": "events.csv",
    "gaze_enriched.csv": "gaze_enriched.csv",
    "fixations_enriched.csv": "fixations_enriched.csv",
    "gaze.csv": "gaze.csv",
    "fixations.csv": "fixations.csv",
    "3d_eye_states.csv": "3d_eye_states.csv",
    "blinks.csv": "blinks.csv",
    "saccades.csv": "saccades.csv",
    "internal.mp4": "internal camera video",
    "external.mp4": "external camera video",
    "world_timestamps.csv": "world_timestamps.csv",
    "surface_positions.csv": "surface_positions.csv",
}
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
OPTIONAL_FOR_UNENRICHED = ["gaze_enriched.csv", "fixations_enriched.csv"]

class SpeedApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SPEED v2.0 - Cognitive and Behavioral Science Lab")
        self.file_entries = {}
        
        main_frame = tk.Frame(root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Participant Name
        name_frame = tk.Frame(main_frame); name_frame.pack(fill=tk.X, pady=5)
        name_label = tk.Label(name_frame, text="Participant Name:", width=20, anchor='w'); name_label.pack(side=tk.LEFT)
        self.participant_name_var = tk.StringVar(); self.participant_name_var.trace_add("write", self.update_output_dir_default)
        self.name_entry = tk.Entry(name_frame, textvariable=self.participant_name_var); self.name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Output Directory
        output_frame = tk.Frame(main_frame); output_frame.pack(fill=tk.X, pady=5)
        output_label = tk.Label(output_frame, text="Output Folder:", width=20, anchor='w'); output_label.pack(side=tk.LEFT)
        self.output_dir_entry = tk.Entry(output_frame); self.output_dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        output_button = tk.Button(output_frame, text="Browse...", command=self.select_output_dir); output_button.pack(side=tk.RIGHT)

        # Analysis Options
        options_frame = tk.LabelFrame(main_frame, text="Analysis Options", padx=5, pady=5); options_frame.pack(fill=tk.X, pady=(5,0))
        
        self.run_standard_var = tk.BooleanVar(value=True)
        self.standard_checkbox = tk.Checkbutton(options_frame, text="Run Standard Event-Based Analysis", variable=self.run_standard_var)
        self.standard_checkbox.pack(anchor='w')
        
        self.generate_standard_video_var = tk.BooleanVar(value=False)
        self.standard_video_checkbox = tk.Checkbutton(options_frame, text="└─ Generate Standard Summary Video", variable=self.generate_standard_video_var)
        self.standard_video_checkbox.pack(anchor='w', padx=(20, 0))

        self.yolo_analysis_var = tk.BooleanVar(value=False)
        self.yolo_checkbox = tk.Checkbutton(options_frame, text="Run YOLO Object Detection Analysis (requires GPU)", variable=self.yolo_analysis_var, command=self.toggle_yolo_requirements)
        self.yolo_checkbox.pack(anchor='w')
        
        self.surface_yolo_var = tk.BooleanVar(value=False)
        self.surface_yolo_checkbox = tk.Checkbutton(options_frame, text="└─ Run YOLO on Marker-Mapped Surface only (slow)", variable=self.surface_yolo_var, command=self.toggle_yolo_requirements)
        self.surface_yolo_checkbox.pack(anchor='w', padx=(20, 0))

        self.generate_surface_video_var = tk.BooleanVar(value=True)
        self.surface_video_checkbox = tk.Checkbutton(options_frame, text="   └─ Generate Warped Surface Video", variable=self.generate_surface_video_var)
        self.surface_video_checkbox.pack(anchor='w', padx=(40, 0))


        # File Selection
        files_frame = tk.LabelFrame(main_frame, text="Select Data Files", padx=5, pady=5)
        files_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        for std_name, display_label in REQUIRED_FILES.items():
            row_frame = tk.Frame(files_frame); row_frame.pack(fill=tk.X, pady=2)
            label = tk.Label(row_frame, text=f"{display_label}:", width=35, anchor='w'); label.pack(side=tk.LEFT)
            entry = tk.Entry(row_frame); entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            self.file_entries[std_name] = entry
            description = FILE_DESCRIPTIONS[std_name]
            button = tk.Button(row_frame, text="Browse...", command=lambda e=entry, d=description: self.select_file(e, d)); button.pack(side=tk.RIGHT)
        
        self.toggle_yolo_requirements()

        # Run Button & Status
        run_button = tk.Button(main_frame, text="Start Analysis", command=self.run_analysis_process, font=('Helvetica', 10, 'bold')); run_button.pack(pady=10)
        self.status_label = tk.Label(main_frame, text="Ready", fg="blue"); self.status_label.pack(pady=5)
        
        # Links
        link_frame = tk.Frame(main_frame)
        link_frame.pack(side=tk.BOTTOM, pady=10)
        lab_link = tk.Label(link_frame, text="Cognitive and Behavioral Science Lab", fg="blue", cursor="hand2"); lab_link.pack(side=tk.LEFT, padx=10)
        lab_link.bind("<Button-1>", lambda e: self.open_link("https://labscoc.wordpress.com/"))
        github_link = tk.Label(link_frame, text="Daniele Lozzi's Github Page", fg="blue", cursor="hand2")
        github_link.pack(side=tk.LEFT, padx=10)
        github_link.bind("<Button-1>", lambda e: self.open_link("https://github.com/danielelozzi/"))


    def update_output_dir_default(self, *args):
        subj_name = self.participant_name_var.get().strip()
        if subj_name:
            self.output_dir_entry.delete(0, tk.END)
            self.output_dir_entry.insert(0, str(Path(f'./analysis_results_{subj_name}').resolve()))
        else:
            self.output_dir_entry.delete(0, tk.END)

    def select_output_dir(self):
        directory_path = filedialog.askdirectory(title="Select Output Folder")
        if directory_path:
            self.output_dir_entry.delete(0, tk.END)
            self.output_dir_entry.insert(0, directory_path)

    def toggle_yolo_requirements(self):
        run_yolo = self.yolo_analysis_var.get()
        run_surface_yolo = self.surface_yolo_var.get()

        # Abilita/disabilita i checkbox dipendenti
        self.surface_yolo_checkbox.config(state=tk.NORMAL if run_yolo else tk.DISABLED)
        self.surface_video_checkbox.config(state=tk.NORMAL if run_yolo and run_surface_yolo else tk.DISABLED)
        
        if not run_yolo:
            self.surface_yolo_var.set(False)
            run_surface_yolo = False
        if not run_surface_yolo:
            self.generate_surface_video_var.set(False)

        # Evidenzia i file richiesti
        self.file_entries["world_timestamps.csv"].master.winfo_children()[0].config(fg="red" if run_yolo else "black")
        self.file_entries["surface_positions.csv"].master.winfo_children()[0].config(fg="red" if run_surface_yolo else "black")
        self.file_entries["gaze_enriched.csv"].master.winfo_children()[0].config(fg="red" if run_surface_yolo else "black")

    def select_file(self, entry_widget, description):
        file_path = filedialog.askopenfilename(title=description)
        if file_path:
            entry_widget.delete(0, tk.END); entry_widget.insert(0, file_path)

    def open_link(self, url): webbrowser.open_new(url)

    def run_analysis_process(self):
        subj_name = self.name_entry.get().strip()
        output_dir_path = self.output_dir_entry.get().strip()
        if not subj_name or not output_dir_path:
            messagebox.showerror("Error", "Please enter a participant name and select an output folder."); return

        # Raccogli tutte le opzioni dalla GUI
        options = {
            "run_standard": self.run_standard_var.get(),
            "generate_standard_video": self.generate_standard_video_var.get(),
            "run_yolo": self.yolo_analysis_var.get(),
            "run_surface_yolo": self.surface_yolo_var.get(),
            "generate_surface_video": self.generate_surface_video_var.get()
        }
        
        selected_files = {std_name: entry.get().strip() for std_name, entry in self.file_entries.items()}

        # Controllo file mancanti
        missing_files = []
        required_by_logic = {
            "world_timestamps.csv": options["run_yolo"],
            "surface_positions.csv": options["run_surface_yolo"],
            "gaze_enriched.csv": options["run_surface_yolo"]
        }
        for name, path in selected_files.items():
            if required_by_logic.get(name, False) and not path:
                 missing_files.append(REQUIRED_FILES[name])
        
        if missing_files:
            messagebox.showerror("Error", f"Please select all required files for the selected options. Missing: {', '.join(missing_files)}"); return

        try:
            self.status_label.config(text=f"Preparing folders for {subj_name}...", fg="blue"); self.root.update_idletasks()
            base_output_dir = Path(output_dir_path)
            data_dir = base_output_dir / 'eyetracking_file'
            data_dir.mkdir(parents=True, exist_ok=True)

            self.status_label.config(text="Copying files...", fg="blue"); self.root.update_idletasks()
            for std_name, source_path_str in selected_files.items():
                if source_path_str: shutil.copy(Path(source_path_str), data_dir / std_name)

            self.status_label.config(text="Starting analysis... This might take some time.", fg="blue"); self.root.update_idletasks()
            
            # Chiama la funzione orchestratrice con tutte le opzioni
            main_analyzer.run_full_analysis(
                subj_name=subj_name, 
                data_dir_str=str(data_dir), 
                output_dir_str=str(base_output_dir), 
                options=options
            )
            
            self.status_label.config(text="Analysis complete!", fg="green")
            messagebox.showinfo("Success", f"Analysis for {subj_name} has finished.\nResults are in '{base_output_dir}'.")
        except Exception as e:
            self.status_label.config(text=f"An error occurred: {e}", fg="red")
            messagebox.showerror("Error", f"An error occurred during the process:\n{e}")
            traceback.print_exc()

if __name__ == '__main__':
    root = tk.Tk()
    app = SpeedApp(root)
    root.mainloop()