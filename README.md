# TrafficVision Web Application

A Streamlit interface for uploading traffic videos and generating vehicle crossing reports using Ultralytics YOLO and ByteTrack.

## Features
- Upload MP4, AVI, MOV or MKV videos.
- Detects COCO classes: car, motorcycle, bus and truck.
- Tracks vehicles with ByteTrack IDs.
- Counts each tracked ID once when its bounding-box center crosses a virtual line from top to bottom.
- Configurable line position, confidence threshold, model and inference size.
- Annotated video, crossing-events CSV, JSON summary, and ZIP bundle.

## Run locally
Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the selected pretrained YOLO weights. Internet access is required unless weights are already cached.

## Important behavior and limitations
- The line is horizontal and placed at the selected percentage of frame height.
- Only top-to-bottom crossings are counted. Reverse-direction and horizontal counting are not currently included.
- Supported output classes are car, motorcycle, bus, truck. COCO model class "motorcycle" is displayed as motorcycle/bike.
- The current crossing test uses the tracked bounding-box center and previous frame center. Track ID switches or lost tracks can cause misses or duplicate counts.
- A fixed line is not appropriate for every camera angle; tune the line per video.
- This prototype processes uploaded files on the server and has no authentication, multi-user job queue, persistent database, or upload-size limits. Do not expose it publicly without adding those controls.
- Counts are not independently validated ground truth. Test on representative footage before operational use.
- The default output uses OpenCV's MP4V encoder; browser playback support can vary by environment. For production deployment, transcode outputs to H.264/AAC using FFmpeg.
