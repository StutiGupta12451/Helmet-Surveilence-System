import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
import numpy as np

@st.cache_resource
def load_model():
    return YOLO(r"Helmet-Surveilence-System\helmet survilence\no_helmet.pt")

model = load_model()
names = model.names


def inside(inner, outer):
    x1, y1, x2, y2 = inner
    X1, Y1, X2, Y2 = outer
    return x1 > X1 and y1 > Y1 and x2 < X2 and y2 < Y2



def process_frame(frame):

    results = model(frame)

    motorcyclists = []
    no_helmets = []
    plates = []

    for r in results:
        boxes = r.boxes.xyxy
        classes = r.boxes.cls
        confs = r.boxes.conf

        for box, cls, conf in zip(boxes, classes, confs):
            label = names[int(cls)]
            x1, y1, x2, y2 = map(int, box)

            if conf < 0.5:
                continue

            if label == "motorcyclist":
                motorcyclists.append((x1, y1, x2, y2))

            elif label == "no-helmet":
                no_helmets.append((x1, y1, x2, y2))

            elif label == "plate":
                plates.append((x1, y1, x2, y2))

    for rider in motorcyclists:

        rider_has_no_helmet = False
        rider_plate = None

        for nh in no_helmets:
            if inside(nh, rider):
                rider_has_no_helmet = True
                break

        if not rider_has_no_helmet:
            continue

        for p in plates:
            if inside(p, rider):
                rider_plate = p
                break

        if rider_plate is None:
            continue

        x1, y1, x2, y2 = rider
        px1, py1, px2, py2 = rider_plate

        # Draw rider
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
        cv2.putText(frame, "NO HELMET RIDER", (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # Draw plate
        cv2.rectangle(frame, (px1,py1), (px2,py2), (0,255,0), 2)

    return frame


st.title("🚦 Helmet Violation Detection System")

uploaded_file = st.file_uploader(
    "Upload Image or Video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

if uploaded_file is not None:

    file_type = uploaded_file.type


    if "image" in file_type:

        file_bytes = uploaded_file.read()
        np_arr = bytearray(file_bytes)
        frame = cv2.imdecode(
            np.frombuffer(np_arr, dtype="uint8"), cv2.IMREAD_COLOR
        )

        output = process_frame(frame)

        st.image(output, channels="BGR", caption="Processed Image")

        output_path = "output.jpg"
        cv2.imwrite(output_path, output)

        with open(output_path, "rb") as f:
            st.download_button(
                "Download Image",
                f,
                file_name="processed_image.jpg"
            )


    else:

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        output_path = "output_video.mp4"
        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed = process_frame(frame)
            out.write(processed)

            stframe.image(processed, channels="BGR")

        cap.release()
        out.release()

        st.success("Video processing complete!")

        with open(output_path, "rb") as f:
            st.download_button(
                "Download Video",
                f,
                file_name="processed_video.mp4"
            )
