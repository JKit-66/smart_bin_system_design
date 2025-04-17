# Smart Bin System Design

A smart waste sorting bin designed for primary schools, developed as a final year project at the University of Southampton. It leverages computer vision and the YOLOv8 framework to classify and sort waste into five recycling categories: Mixed Recycling, Food Waste, General Waste, Paper, and Prohibited Item. The system uses motor-driven mechanisms controlled by ESP32-S3 microcontrollers to automate sorting, aiming to foster recycling awareness among students through real-time object detection, a Pygame-based GUI, and educational components.

## Features

- **Waste Classification**: Utilizes YOLOv8 for real-time detection and classification of waste items.
- **Automated Sorting**: Employs ESP32-S3 microcontrollers to control motors for sorting waste into designated bins.
- **Interactive GUI**: Includes a Pygame-based interface for monitoring and educational interaction.
- **Educational Focus**: Designed to engage young students in sustainable practices.
- **Comprehensive Pipeline**: Covers model training, data processing, and hardware integration.

## Repository Structure

### `ESP32-Arduino Code`

Contains Arduino code for controlling the sorting box's movement and lid operations in the auto-sorting mechanism.

### `GUI`

- **File**: `main_interactive_game.py`
- **Description**: Run this file to launch the Pygame GUI for monitoring and interaction.
- **Note**: Update the path to the YOLO `.pt` weight file in the script before running.

### `Object Classification Training`

- **File**: `run.sbatch`
- **Description**: Main script for submitting YOLOv8 training jobs to an HPC cluster. It executes `trainYOLO.py` under the `roboblow` folder for model training.

### `Object Classification Pre-Training Processing`

- **Description**: Contains a script to update class indices in label files for the dataset.
- **Dataset**: The `dataset` folder includes all waste items for training, split into `train`, `test`, and `valid` subsets after compilation.

### `Object Classification Post-Training Processing Result`

Contains scripts and data files for analyzing validation results:

- `step1_validation_step_auto_output_to_csv.py`: Validates detection on dataset images, annotates results into a `results` folder, and compiles them into a CSV file.
- `DetectionResults_34dfa47a17634b648b49de72c5d4d376_v7.csv`: Stores detection results from validation.
- `step2_post_process_excel_csv_file.py`: Analyzes the CSV file, calculating metrics like confidence score distributions and true detection rates, saving results to `_v7.txt`.
- `_v7.txt`: Contains analyzed data for each class.
- `step3_summarize_exceldata_to_one_big_CM_matrix.py`: Processes `_v7.txt` into a 27x27 confusion matrix.
- `step4_categorize_big_CM_to_5_categories.py`: Reduces the 27x27 matrix into a 6x6 matrix based on recycling categories.
- `step5_plotting_of_CM_with_matrix.py`: Plots the confusion matrices with labels and titles.
- `step6_get_confidenceScore_distribution_from_excel_csv_file.py`: Analyzes `_v7.txt` to extract and sort confidence score distributions for deeper model insights.


## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/JKit-66/smart_bin_system_design.git
   ```
2. For ESP32 firmware, install the Arduino IDE and required ESP32 board support.

3. Update the YOLO weight file path in `GUI/main_interactive_game.py`.


## Usage

1. **Train the Model**:
   - Submit the `run.sbatch` script to your HPC cluster for YOLOv8 training.
2. **Run the GUI**:
   - Execute `python GUI/main_interactive_game.py` to launch the interactive interface.
3. **Upload Firmware**:
   - Use Arduino IDE to upload the code in `ESP32-Arduino Code` to the ESP32-S3 microcontroller.
4. **Analyze Results**:
   - Run the scripts in `Object Classification Post-Training Processing Result` sequentially to process and visualize model performance.


## Acknowledgments

- Built with YOLOv8 for object detection.
- Utilizes Pygame for the GUI.
- Powered by ESP32-S3 microcontrollers for hardware control.