import subprocess
import imageio_ffmpeg
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import cv2

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RoadGuardian AI",
    page_icon="🚗",
    layout="wide"
)

# ============================================================
# LOAD MODEL
# ============================================================

#MODEL_PATH = "RoadGuardian_best_gpu.pt"
#MODEL_PATH = "RoadGuardian_final_merged_50ep.pt"
MODEL_PATH = "RoadGuardian_best_new_dataset.pt"

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()

# ============================================================
# HEADER
# ============================================================

st.title("🚗 RoadGuardian AI")
st.subheader("Smart Road Hazard Detection and Alert System")

st.write(
    "AI-powered detection of road hazards using YOLOv8n."
)

st.divider()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Detection Settings")

confidence = st.sidebar.slider(
    "Detection Confidence",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05
)

st.sidebar.markdown("---")

st.sidebar.write("### 🎯 Detectable Hazards")

st.sidebar.write("🕳️ Pothole")
st.sidebar.write("〰️ Crack")
st.sidebar.write("⚫ Manhole")
st.sidebar.write("🧱 Road Debris")
st.sidebar.write("⚠️ Speed Breaker")

st.sidebar.markdown("---")

st.sidebar.info(
    "Model: YOLOv8n\n\n"
    "Input: Image / Video\n\n"
    "Device: CPU"
)

# ============================================================
# MAIN TABS
# ============================================================

image_tab, video_tab = st.tabs(
    ["📷 Image Detection", "🎥 Video Detection"]
)

# ============================================================
# IMAGE DETECTION
# ============================================================

with image_tab:

    st.header("📷 Road Image Analysis")

    uploaded_file = st.file_uploader(
        "Upload a road image",
        type=["jpg", "jpeg", "png"],
        key="image_upload"
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        # ----------------------------------------------------
        # INPUT IMAGE
        # ----------------------------------------------------

        with col1:

            st.subheader("Input Image")

            st.image(
                image,
                use_container_width=True
            )

        # ----------------------------------------------------
        # DETECTION
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
        ) as temp_file:

            image.save(temp_file.name)
            temp_path = temp_file.name

        with st.spinner("🔍 Analyzing road image..."):

            results = model.predict(
                source=temp_path,
                imgsz=416,
                conf=confidence,
                device="cpu",
                verbose=False
            )

        annotated_image = results[0].plot()

        # ----------------------------------------------------
        # OUTPUT IMAGE
        # ----------------------------------------------------

        with col2:

            st.subheader("Detection Result")

            st.image(
                annotated_image,
                channels="BGR",
                use_container_width=True
            )

        # ----------------------------------------------------
        # DETECTION DETAILS
        # ----------------------------------------------------

        boxes = results[0].boxes

        st.divider()

        if boxes is not None and len(boxes) > 0:

            st.error("⚠️ ROAD HAZARD DETECTED")

            st.subheader("📋 Detection Details")

            total = len(boxes)

            st.metric(
                "Total Hazards Detected",
                total
            )

            detected_hazards = []

            for box in boxes:

                class_id = int(box.cls[0])
                conf = float(box.conf[0])

                class_name = model.names[class_id]

                detected_hazards.append(
                    (class_name, conf)
                )

            # Display detections
            for class_name, conf in detected_hazards:

                st.warning(
                    f"⚠️ **{class_name}**  |  "
                    f"Confidence: **{conf * 100:.1f}%**"
                )

        else:

            st.success(
                "✅ No road hazard detected in this image."
            )

        os.remove(temp_path)


# ============================================================
# VIDEO DETECTION
# ============================================================



# ============================================================
# VIDEO DETECTION
# ============================================================

with video_tab:

    st.header("🎥 Road Video Analysis")

    uploaded_video = st.file_uploader(
        "Upload a road video",
        type=["mp4", "avi", "mov"],
        key="video_upload"
    )

    if uploaded_video is not None:

        st.info(
            "Video processing may take some time "
            "because YOLO is running on CPU."
        )

        # Save input video
        input_video = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        input_video.write(uploaded_video.read())
        input_video.close()

        # Temporary AVI output
        avi_path = tempfile.mktemp(suffix=".avi")

        # Final browser-compatible MP4
        mp4_path = tempfile.mktemp(suffix=".mp4")

        cap = cv2.VideoCapture(input_video.name)

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0:
            fps = 25

        if width <= 0 or height <= 0:
            st.error("Unable to read video dimensions.")
            cap.release()
            st.stop()

        # Use AVI with MJPG codec
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")

        writer = cv2.VideoWriter(
            avi_path,
            fourcc,
            fps,
            (width, height)
        )

        if not writer.isOpened():
            cap.release()
            st.error("Unable to create output video.")
            st.stop()

        frame_count = 0
        hazard_count = 0

        progress = st.progress(0)
        status = st.empty()

        # Process each frame
        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            results = model.predict(
                source=frame,
                imgsz=416,
                conf=confidence,
                device="cpu",
                verbose=False
            )

            annotated_frame = results[0].plot()

            if (
                results[0].boxes is not None
                and len(results[0].boxes) > 0
            ):
                hazard_count += len(results[0].boxes)

            writer.write(annotated_frame)

            if total_frames > 0:
                progress.progress(
                    min(frame_count / total_frames, 1.0)
                )

            if frame_count % 10 == 0:
                status.write(
                    f"Processing frame {frame_count}/{total_frames}"
                )

        cap.release()
        writer.release()

        # Convert AVI to H.264 MP4
        status.write("Converting video to browser-compatible MP4...")

        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

        command = [
            ffmpeg_path,
            "-y",
            "-i", avi_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            mp4_path
        ]

        conversion = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        progress.progress(1.0)

        if conversion.returncode != 0:
            st.error("Video conversion failed.")
            st.code(
                conversion.stderr.decode(
                    errors="ignore"
                )
            )
        else:

            status.success("Video processing completed!")

            st.subheader("🎬 Processed Video")

            with open(mp4_path, "rb") as video_file:
                video_bytes = video_file.read()

            st.video(video_bytes)

            st.divider()

            st.metric(
                "Hazard Detections",
                hazard_count
            )

            if hazard_count > 0:
                st.error(
                    "⚠️ ROAD HAZARDS DETECTED IN VIDEO"
                )
            else:
                st.success(
                    "✅ No road hazards detected."
                )

        # Cleanup
        for file_path in [
            input_video.name,
            avi_path,
            mp4_path
        ]:
            if os.path.exists(file_path):
                os.remove(file_path)