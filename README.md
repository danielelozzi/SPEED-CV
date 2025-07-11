# SPEED (labScoc Processing and Extraction of Eye tracking Data) v2.1

*Eye-Tracking Data Analysis Software*

SPEED is a Python-based tool with a graphical user interface (GUI) for processing and analyzing eye-tracking data from cognitive and behavioral experiments. It is designed to segment data based on predefined events, calculate key metrics, and generate visualizations of eye movements like gaze paths and fixations.

*This is a new version based on the first version of **SPEED**:* [**SPEED v1.0**](https://github.com/danielelozzi/SPEED).

The tool can handle both raw ("un-enriched," pixel-based) and surface-projected ("enriched," normalized) data, making it flexible for various stages of the analysis pipeline.

## Data Acquisition 📋

Before using this software, you need to acquire and prepare the data following a specific procedure with Pupil Labs tools.

* **Video Recording**: Use Pupil Labs Neon glasses to record the session.
* *(optional)* **Surface Definition (AprilTag)**: Place AprilTags at the four corners of a PC screen. These markers allow the Pupil Labs software to track the surface and map gaze coordinates onto it. For more details, see the official documentation: [**Pupil Labs Surface Tracker**](https://docs.pupil-labs.com/neon/neon-player/surface-tracker/).
* **Upload to Pupil Cloud**: Once the recording is complete, upload the data to the Pupil Cloud platform.
* *(optional)* **Enrichment with Marker Mapper**: Inside Pupil Cloud, start the "Marker Mapper" enrichment. This process analyzes the video, detects the AprilTags, generates the `surface_positions.csv` file (which contains the surface coordinates for each frame), and downloads all the data. Marker Mapper Usage Guide: [**Pupil Cloud Marker Mapper**](https://docs.pupil-labs.com/neon/pupil-cloud/enrichments/marker-mapper/#setup).

## Features ✨

* **Graphical User Interface (GUI)**: An intuitive `Tkinter`-based interface for easy file selection and analysis configuration.
* **Event-Based Segmentation**: Analyzes eye-tracking data in segments defined by timestamps in an `events.csv` file.
* **Dual Analysis Mode**: Supports both "un-enriched" (pixel-based) and "enriched" (surface-normalized) data. The user can switch between modes via a checkbox.
* **Path Plot Generation**: Automatically creates and saves PDF plots for fixation paths and raw gaze paths for each event segment.
* **Summary Statistics**: Generates a final CSV file with aggregated metrics for each analysis segment.
* **Automated File Management**: Creates a structured output folder for each participant, copying input files and saving results neatly.
* **Enhanced Pupillometry Plotting**: Generates detailed pupillometry time series plots.
* **Saccade and Blink Analysis**: Visualizes saccade metrics and blink events over time.
* **Density Heatmap Generation**: Creates heatmaps that visualize the areas of highest concentration for fixations and gaze points using a kernel density estimate (KDE).
* **YOLOv8 Object Detection Integration**: Optionally detects and tracks objects in the scene video, correlating gaze data to specific objects.
* **Focused Surface Analysis**: A dedicated mode to run YOLO analysis exclusively on a marker-defined surface. The tool automatically crops and applies perspective correction to each video frame, isolating the area of interest (e.g., a computer screen) for highly accurate object and gaze tracking within that specific zone. This feature can now run in both **enriched** and **un-enriched** modes.

## Environment Setup ⚙️

To run the SPEED analysis tool, you'll need Python 3 and several scientific computing libraries. It's highly recommended to use a virtual environment to manage dependencies.

1.  **Create a virtual environment:**
    ```bash
    python -m venv speed-env
    source speed-env/bin/activate  # On Windows, use `speed-env\Scripts\activate`
    ```

2.  **Install the required libraries:**
    The required libraries depend on the analysis you want to run. Create a `requirements.txt` file with the content below. For the optional YOLO analysis, you will need `torch` and `ultralytics`.
    ```
    pandas
    numpy
    matplotlib
    opencv-python
    scipy
    tqdm
    # Optional for YOLO analysis
    torch
    ultralytics
    ```
    Then, install them using pip:
    ```bash
    pip install -r requirements.txt
    ```
    * **Note on `Tkinter`**: This is part of the Python standard library and does not require a separate installation.
    * **Note on `torch`**: Installing PyTorch can be complex, especially if you want to use a GPU (highly recommended for YOLO). Please refer to the official [PyTorch installation guide](https://pytorch.org/get-started/locally/) for instructions tailored to your system.
    * **Note on YOLO**: To use YOLO, you must download the pre-trained neural network from the following link on the Hugging Face website: <https://huggingface.co/Ultralytics/YOLOv8/blob/main/yolov8n.pt>

## How to Use the Application 🚀

1.  **Launch the GUI**: Run the `GUI.py` script from your terminal.
    ```bash
    python GUI.py
    ```

2.  **Fill in the Information**:
    * **Participant Name**: Enter a unique identifier for the participant.
    * **Output Folder**: The application will automatically suggest an output path based on the participant's name. You can also use the "Browse..." button to select a different location.
    * **Analysis Options**: Select the desired analyses. Note that some options depend on others.
        * `Analyze un-enriched data only`: **This is a key option.** When checked, all analyses (including standard plots and surface video) will be based on raw, pixel-based data (`gaze.csv`, `fixations.csv`), ignoring enriched files. When unchecked, the analysis will use enriched data if available.
        * `Run Standard Event-Based Analysis`: Enables the baseline analysis of fixations, saccades, and pupil metrics segmented by events.
        * `└─ Generate Standard Summary Video`: Creates the summary video for the standard analysis.
        * `Run YOLO Object Detection Analysis`: Master switch to enable YOLO features (a GPU is highly recommended).
        * `└─ Run YOLO on Marker-Mapped Surface only`: When checked, it runs a slower but highly focused analysis on the perspective-corrected surface. This analysis now works in both enriched and un-enriched modes.
        * `   └─ Generate Warped Surface Video`: Creates a video of the cropped and flattened surface with object detections and gaze points drawn on it.

3.  **Select Input Files**:
    * Click the "Browse..." button next to each file type to select the corresponding data file.
    * The labels of the required files will turn **red** based on the selected analysis options.

4.  **Start the Analysis**:
    * Once all required fields are filled and files are selected, click the **"Start Analysis"** button.
    * The status label at the bottom will show the progress.
    * Upon completion, a confirmation message will appear.

## Input Files 📂

The application requires several specific CSV and MP4 files. The required files change dynamically based on the selected options in the GUI.

| Standard Name          | Display Label (in GUI)                 | Description                                                                    | Requirement                                      |
| ---------------------- | -------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------ |
| `events.csv`           | `events.csv`                           | Contains timestamps that define the start and end of experimental segments.    | **Always** |
| `gaze_enriched.csv`    | `gaze_enriched.csv`                    | Gaze data with coordinates normalized to a detected surface.                   | Required if "un-enriched only" is **unchecked**. |
| `fixations_enriched.csv` | `fixations_enriched.csv`               | Fixation data with coordinates normalized to a detected surface.               | Required if "un-enriched only" is **unchecked**. |
| `gaze.csv`             | `un-enriched gaze CSV file`            | Raw gaze data with coordinates in pixels (`px`).                               | **Always** |
| `fixations.csv`        | `un-enriched fixations CSV file`       | Raw fixation data with coordinates in pixels (`px`).                           | **Always** |
| `3d_eye_states.csv`    | `3D eye states CSV file (pupil)`       | Pupil diameter and other 3D eye model data.                                    | **Always** |
| `blinks.csv`           | `blinks CSV file`                      | Data on blink events.                                                          | **Always** |
| `saccades.csv`         | `saccades CSV file`                    | Data on saccadic movements.                                                    | **Always** |
| `internal.mp4`         | `internal camera video`                | The video recording of the participant's eye.                                  | **Always** |
| `external.mp4`         | `external camera video`                | The video recording of the participant's scene/view.                           | **Always** |
| `world_timestamps.csv` | `world_timestamps.csv`                 | Timestamps for each frame of the external video.                               | Required for **all YOLO Analyses**.              |
| `surface_positions.csv`  | `Marker Mapper surface positions CSV`  | Corner coordinates of the marker-defined surface per frame.                    | Required for **Surface YOLO Analysis**.          |

## Output Files 📈

The analysis generates a main folder named `analysis_results_{participant_name}`.

1.  **`eyetracking_file/`**
    * A subfolder containing copies of all the input files used for the analysis.

2.  **Summary Results (`.csv`)**
    * `summary_results_{subj_name}.csv`: A CSV file containing the main quantitative outcomes of the standard analysis, with one row per event segment.

3.  **YOLO Surface Analysis Results (`.csv` and `.mp4`)**
    *These files are generated only when the "Run YOLO on Marker-Mapped Surface" option is enabled.*
    * **`surface_analysis_video_{subj_name}.mp4`**: A video showing the **cropped and perspective-corrected surface**. It displays the detected object bounding boxes and the user's gaze point overlaid on the flattened surface view.
    * **`surface_analysis_data_{subj_name}.csv`**: A detailed frame-by-frame CSV of the surface analysis. It includes the detected objects, their bounding boxes (normalized to the surface dimensions), and whether the user's gaze was within each box for every processed frame.

4.  **Analysis Plots (`.pdf`)**
    * These plots visualize different aspects of the eye-tracking data for each event segment. Depending on the analysis mode, you will get separate plots for `_enriched` and `_not_enriched` data.

5.  **Analysis Video (`.mp4`)**
    * `output_analysis_video.mp4`: (if the option is checked) An MP4 video that synchronizes the internal view, external view, and a real-time plot of pupil diameters.

## ✍️ Authors & Citation

* Dr. Daniele Lozzi
* Dr. Ilaria Di Pompeo
* Martina Marcaccio
* Matias Ademaj
* Dr. Simone Migliore
* Prof. Giuseppe Curcio

*If you use this script in your research or work, please cite the following publications:*

* Lozzi, D.; Di Pompeo, I.; Marcaccio, M.; Ademaj, M.; Migliore, S.; Curcio, G. SPEED: A Graphical User Interface Software for Processing Eye Tracking Data. NeuroSci 2025, 6, 35. <https://doi.org/10.3390/neurosci6020035>
* Lozzi, D.; Di Pompeo, I.; Marcaccio, M.; Alemanno, M.; Krüger, M.; Curcio, G.; Migliore, S. AI-Powered Analysis of Eye Tracker Data in Basketball Game. Sensors 2025, 25, 3572. <https://doi.org/10.3390/s25113572>

*If you also use the Computer Vision YOLO-based feature, please cite the following publication:*

* Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You only look once: Unified, real-time object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 779-788). <https://doi.org/10.1109/CVPR.2016.91>

---

*This tool is developed for the Cognitive and Behavioral Science Lab. For more information, visit [our website](https://labscoc.wordpress.com/).*
