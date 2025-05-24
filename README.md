# Object Based Video Summarization with SAM2 and YOLO-v8

## Overview

This README provides instructions on setting up and running an object-based video summarization pipeline using the **SAM2 video predictor** and **YOLO-v8** object detection. The script extracts frames from a video, performs object detection, annotates the frames, and generates a final summary video based on user-selected objects.

## Prerequisites

Ensure you have the following installed:

- Python 3.6+ (for Jupyter/Colab)
- Python 3.9+ (for Streamlit UI)
- Jupyter Notebook or Google Colab (preferred)
- VS Code (for Streamlit)
- CUDA-compatible GPU (Recommended: 100 GB GPU for optimal performance)

> ⚠️ macOS is not supported due to lack of NVIDIA GPU support required by SAM2.

## Setup Instructions

### Install Required Libraries

Run the following commands:

```bash
pip install huggingface_hub
pip install ultralytics
pip install opencv-python Pillow ipywidgets
pip install sam2
pip install -q supervision[assets] jupyter_bbox_widget
```

### Set Up Paths

Update the script with the paths to your video and image:

```python
SOURCE_VIDEO = "/path/to/your/demo.mp4"
image_path = "/path/to/your/sample_pics.png"
```

### Frame Extraction

Set the following parameters to control frame extraction:

```python
START_IDX = 0       # Starting frame index
END_IDX = 300       # Ending frame index
SCALE = 1.0         # Frame resize scale
```

## Model Evaluation Steps

### Step 1: Select Objects for Summarization

- Choose at least two objects to include in the summarized video.

### Step 2: Submit Selection

- Once objects are selected, run the summarization script.

### Step 3: Output

- Final annotated and summarized video will be saved in the root directory.

## Running the Streamlit UI

### Install Dependencies

Ensure the following dependencies are installed:

```bash
pip install huggingface_hub
pip install ultralytics
pip install opencv-python Pillow ipywidgets
pip install sam2
```

### Launch the App

Run the Streamlit app with:

```bash
streamlit run app.py
```

### Use the UI

1. Upload a video and image
2. Select objects to summarize
3. Choose the timestamp range
4. View and download the final summarized video

## Output

- The final summarized video is saved in the project root.
- Intermediate outputs (frames, annotations) are saved optionally.

## Contact

For questions, raise an issue or contact the project maintainer.
