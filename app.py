import cv2
import json
import time
import tempfile
import zipfile
from pathlib import Path
from collections import Counter
import pandas as pd
import streamlit as st
from ultralytics import YOLO

APP_TITLE = "TrafficVision"
SUPPORTED_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

st.set_page_config(page_title="TrafficVision", page_icon="🚦", layout="wide")
st.title("🚦 TrafficVision")
st.caption("Traffic intelligence from uploaded video")
st.write("Upload a traffic video to detect, track and count vehicles crossing a virtual line.")

@st.cache_resource
def load_model(model_name: str):
    return YOLO(model_name)

def process_video(input_path, output_path, events_path, summary_path, model_name, line_ratio, conf, imgsz):
    model = load_model(model_name)
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open the uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    line_y = int(height * line_ratio)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not initialize output video writer.")

    # Track each ID's previous center and count at most once per track.
    previous_y = {}
    counted_ids = set()
    events = []
    per_class = Counter()
    processed = 0
    start = time.time()
    progress = st.progress(0.0, text="Preparing video analysis…")
    status = st.empty()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            processed += 1
            result = model.track(
                frame, persist=True, tracker="bytetrack.yaml",
                conf=conf, imgsz=imgsz, verbose=False
            )[0]

            # Draw the virtual line.
            cv2.line(frame, (0, line_y), (width - 1, line_y), (0, 0, 255), 2)
            cv2.putText(frame, f"Counting line y={line_y}", (12, max(25, line_y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

            boxes = result.boxes
            if boxes is not None and len(boxes) and boxes.id is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                ids = boxes.id.int().cpu().tolist()
                cls_ids = boxes.cls.int().cpu().tolist()
                confs = boxes.conf.cpu().tolist()
                for box, track_id, cls_id, score in zip(xyxy, ids, cls_ids, confs):
                    if cls_id not in SUPPORTED_CLASSES:
                        continue
                    x1, y1, x2, y2 = [int(v) for v in box]
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    label = SUPPORTED_CLASSES[cls_id]
                    color = (60, 200, 60) if track_id not in counted_ids else (255, 180, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} ID:{track_id}", (x1, max(20, y1 - 7)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                    old_y = previous_y.get(track_id)
                    # Count only a center crossing from above the line to at/below it.
                    if (track_id not in counted_ids and old_y is not None
                            and old_y < line_y <= center_y):
                        counted_ids.add(track_id)
                        per_class[label] += 1
                        events.append({
                            "frame": processed,  # 1-based frame numbering
                            "timestamp_seconds": round((processed - 1) / fps, 3),
                            "track_id": int(track_id),
                            "vehicle_class": label,
                            "direction": "top_to_bottom",
                            "confidence": round(float(score), 4),
                            "center_x": int(center_x),
                            "center_y": int(center_y)
                        })
                    previous_y[track_id] = center_y

            total = sum(per_class.values())
            cv2.putText(frame, f"Counted crossings: {total}", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            writer.write(frame)

            if frame_count > 0:
                progress.progress(min(processed / frame_count, 1.0),
                                  text=f"Processing frame {processed}/{frame_count}")
            status.write(f"Frames processed: {processed} · Unique crossings: {sum(per_class.values())}")

    finally:
        cap.release()
        writer.release()

    elapsed = round(time.time() - start, 3)
    pd.DataFrame(events, columns=[
        "frame", "timestamp_seconds", "track_id", "vehicle_class",
        "direction", "confidence", "center_x", "center_y"
    ]).to_csv(events_path, index=False)

    total = sum(per_class.values())
    class_counts = {name: int(per_class.get(name, 0))
                    for name in ["car", "motorcycle", "bus", "truck"]}
    summary = {
        "application": APP_TITLE,
        "model": model_name,
        "tracker": "ByteTrack",
        "source_video": Path(input_path).name,
        "video": {
            "frames_processed": processed,
            "fps": round(float(fps), 3),
            "width": width,
            "height": height,
            "duration_seconds": round(processed / fps, 3) if fps else None
        },
        "counting": {
            "line_ratio": line_ratio,
            "line_y_pixels": line_y,
            "direction": "top_to_bottom",
            "total_unique_crossings": total,
            "vehicle_class_counts": class_counts
        },
        "processing": {"elapsed_seconds": elapsed},
        "notes": [
            "Counts are unique tracked crossings from top to bottom at the configured virtual line.",
            "Only COCO classes car, motorcycle, bus and truck are included.",
            "Tracking IDs can switch or fragment; results should be validated for each camera and scene.",
            "This report does not measure road-wide traffic volume unless the camera and counting line are configured appropriately."
        ]
    }
    Path(summary_path).write_text(json.dumps(summary, indent=2))
    progress.progress(1.0, text="Analysis complete.")
    return summary, pd.DataFrame(events), output_path

with st.sidebar:
    st.header("Analysis settings")
    model_name = st.selectbox("Detection model", ["yolo11n.pt", "yolo11s.pt"], index=0)
    line_ratio_pct = st.slider("Counting line height (%)", 10, 90, 60, 1)
    conf = st.slider("Detection confidence", 0.10, 0.90, 0.25, 0.05)
    imgsz = st.select_slider("Inference image size", options=[320, 416, 512, 640, 768], value=640)
    st.caption("Larger models and image sizes may improve detection but take longer and use more memory.")

uploaded = st.file_uploader("Upload a traffic video", type=["mp4", "avi", "mov", "mkv"])
if uploaded:
    st.video(uploaded)
    st.info("For best results, use a stable camera view where vehicles cross the counting line clearly.")
    if st.button("Analyze traffic", type="primary", use_container_width=True):
        workdir = Path(tempfile.mkdtemp(prefix="trafficvision_"))
        input_path = workdir / uploaded.name
        input_path.write_bytes(uploaded.getbuffer())
        output_path = workdir / "trafficvision_annotated.mp4"
        events_path = workdir / "crossing_events.csv"
        summary_path = workdir / "traffic_summary.json"
        try:
            summary, events_df, annotated_path = process_video(
                input_path, output_path, events_path, summary_path,
                model_name, line_ratio_pct / 100.0, conf, imgsz
            )
            st.session_state["tv_result"] = {
                "summary": summary,
                "events": events_df.to_dict("records"),
                "video": str(annotated_path),
                "csv": str(events_path),
                "json": str(summary_path)
            }
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.exception(exc)

result = st.session_state.get("tv_result")
if result:
    summary = result["summary"]
    counts = summary["counting"]["vehicle_class_counts"]
    st.divider()
    st.header("Traffic analysis report")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Unique crossings", summary["counting"]["total_unique_crossings"])
    m2.metric("Cars", counts["car"])
    m3.metric("Motorcycles", counts["motorcycle"])
    m4.metric("Buses + trucks", counts["bus"] + counts["truck"])

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Vehicle class counts")
        chart_df = pd.DataFrame({"Vehicle class": list(counts.keys()), "Count": list(counts.values())})
        st.bar_chart(chart_df.set_index("Vehicle class"))
        st.dataframe(chart_df, use_container_width=True, hide_index=True)
    with right:
        st.subheader("Video and processing details")
        st.json({
            "Frames processed": summary["video"]["frames_processed"],
            "FPS": summary["video"]["fps"],
            "Resolution": f'{summary["video"]["width"]} × {summary["video"]["height"]}',
            "Duration (seconds)": summary["video"]["duration_seconds"],
            "Processing time (seconds)": summary["processing"]["elapsed_seconds"],
            "Counting line (pixels)": summary["counting"]["line_y_pixels"]
        })

    st.subheader("Annotated video")
    st.video(result["video"])
    events_df = pd.DataFrame(result["events"])
    st.subheader("Crossing events")
    if events_df.empty:
        st.warning("No top-to-bottom crossings were counted. Check the camera angle, counting line, confidence threshold, and video quality.")
    else:
        st.dataframe(events_df, use_container_width=True, hide_index=True)
    with open(result["csv"], "rb") as f:
        st.download_button("Download crossing events (CSV)", f, file_name="trafficvision_crossing_events.csv", mime="text/csv")
    with open(result["json"], "rb") as f:
        st.download_button("Download summary report (JSON)", f, file_name="trafficvision_summary.json", mime="application/json")

    bundle_path = Path(result["video"]).parent / "trafficvision_report_bundle.zip"
    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(result["video"], "annotated_video.mp4")
        zf.write(result["csv"], "crossing_events.csv")
        zf.write(result["json"], "traffic_summary.json")
    with open(bundle_path, "rb") as f:
        st.download_button("Download complete report bundle (ZIP)", f, file_name="trafficvision_report_bundle.zip", mime="application/zip")

st.divider()
st.caption("TrafficVision is a prototype. Validate results against manually reviewed footage before operational use.")
