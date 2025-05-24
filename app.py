import streamlit as st
from PIL import Image
import cv2
import numpy as np
import base64
from ultralytics import YOLO
import os
from pathlib import Path
from sam2.sam2_video_predictor import SAM2VideoPredictor
import supervision as sv

# Set up directories
HOME = os.getcwd()
SOURCE_FRAMES_DIR = Path(HOME) / "source_frames"
SOURCE_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATED_FRAMES_DIR = Path(HOME) / "annotated_frames"
ANNOTATED_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

# Load SAM2 model
sam2_model = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large")

# Helper functions
def detect_objects(image_path):
    model = YOLO("yolov8n.pt")
    results = model(image_path)
    detected_objects = []

    object_count = {}
    for result in results:
        for obj in result.boxes.data:
            class_id = int(obj[5].item())
            class_name = model.names[class_id]

            if class_name in object_count:
                object_count[class_name] += 1
            else:
                object_count[class_name] = 1

            unique_label = f"{class_name}_{object_count[class_name]}"
            detected_objects.append(unique_label)

    return detected_objects

def encode_image(filepath):
    with open(filepath, 'rb') as f:
        image_bytes = f.read()
    return "data:image/jpg;base64," + base64.b64encode(image_bytes).decode('utf-8')

# Streamlit App
st.title("Video Object Summarization")

uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_video and uploaded_image:
    # Save uploaded files
    video_path = os.path.join(HOME, "uploaded_video.mp4")
    with open(video_path, "wb") as f:
        f.write(uploaded_video.getbuffer())

    image_path = os.path.join(HOME, "uploaded_image.jpg")
    with open(image_path, "wb") as f:
        f.write(uploaded_image.getbuffer())

    # Detect objects in the image
    st.write("Detecting objects in the uploaded image...")
    detected_objects = detect_objects(image_path)
    st.write("Detected objects:", detected_objects)

    # User selects objects to summarize
    selected_objects = st.multiselect("Select Objects to Summarize", detected_objects)
    start_time = st.number_input("Start Time (seconds)", min_value=0, step=1)
    end_time = st.number_input("End Time (seconds)", min_value=1, step=1)

    if st.button("Process Video"):
        st.write("Processing video...")

        # Video summarization process
        video_info = sv.VideoInfo.from_video_path(video_path)
        SCALE_FACTOR = 0.5
        frames_generator = sv.get_video_frames_generator(video_path, start=int(start_time * video_info.fps), end=int(end_time * video_info.fps))
        
        images_sink = sv.ImageSink(
            target_dir_path=SOURCE_FRAMES_DIR.as_posix(),
            overwrite=True,
            image_name_pattern="{:05d}.jpeg"
        )

        with images_sink:
            for frame in frames_generator:
                frame = sv.scale_image(frame, SCALE_FACTOR)
                images_sink.save_image(frame)

        SOURCE_FRAME_PATHS = sorted(sv.list_files_with_extensions(SOURCE_FRAMES_DIR.as_posix(), extensions=["jpeg"]))
        inference_state = sam2_model.init_state(video_path=SOURCE_FRAMES_DIR.as_posix())

        sam2_model.reset_state(inference_state)

        FRAME_IDX = 0
        widget_boxes = [{'x': 100, 'y': 100, 'width': 0, 'height': 0, 'label': obj} for obj in selected_objects]

        for object_id, label in enumerate(selected_objects, start=1):
            points = np.array([[box['x'], box['y']] for box in widget_boxes if box['label'] == label], dtype=np.float32)
            labels = np.ones(len(points))

            _, object_ids, mask_logits = sam2_model.add_new_points(
                inference_state=inference_state,
                frame_idx=FRAME_IDX,
                obj_id=object_id,
                points=points,
                labels=labels
            )

        TARGET_VIDEO = Path(HOME) / "final_annotated_video.mp4"
        with sv.VideoSink(TARGET_VIDEO.as_posix(), video_info=video_info) as sink:
            for frame_idx, object_ids, mask_logits in sam2_model.propagate_in_video(inference_state):
                frame_path = SOURCE_FRAME_PATHS[frame_idx]
                frame = cv2.imread(frame_path)
                masks = (mask_logits > 0.0).cpu().numpy()
                masks = np.squeeze(masks).astype(bool)

                if np.any(masks):
                    detections = sv.Detections(
                        xyxy=sv.mask_to_xyxy(masks=masks),
                        mask=masks,
                        class_id=np.array(object_ids)
                    )

                    annotated_frame = sv.MaskAnnotator().annotate(scene=frame.copy(), detections=detections)
                    sink.write_frame(annotated_frame)

                    annotated_frame_path = ANNOTATED_FRAMES_DIR / f"{frame_idx:05d}.jpeg"
                    cv2.imwrite(str(annotated_frame_path), annotated_frame)

        st.video(str(TARGET_VIDEO))
        st.write("Video processing complete. You can download the summarized video.")

        # Provide download link for the video
        with open(TARGET_VIDEO, "rb") as video_file:
            video_bytes = video_file.read()
            b64_video = base64.b64encode(video_bytes).decode('utf-8')
            download_link = f'<a href="data:video/mp4;base64,{b64_video}" download="summarized_video.mp4">Download Summarized Video</a>'
            st.markdown(download_link, unsafe_allow_html=True)