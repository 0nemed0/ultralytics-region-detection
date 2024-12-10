# Modifications from the streamlit_07.py

import cv2
import tempfile
import streamlit as st
import numpy as np
from ultralytics import YOLO, solutions
import settings

# Set up the Streamlit page layout
st.set_page_config(layout="wide")

# Load the pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")

# Streamlit app title
st.title("TRAFFIC MONITORING SYSTEM")

# Define options for video input
option = st.selectbox('Select Input Type:', ('Recorded Video', 'Live Webcam', 'External Camera'))

# Function to process video
def process_video(cap, polygons, single_line):
    if cap and cap.isOpened():
        # Initialize single-element container
        stframe = st.empty()

        # Specify classes to count
        classes_to_count = [1, 2, 3, 5, 7]  # Class IDs

        # Initialize Object Counters for Polygon 1
        counters1 = [
            solutions.ObjectCounter(
                view_img=False,
                reg_pts=polygons.get("Polygon 1"),
                classes_names=model.names,
                draw_tracks=True,
                line_thickness=2,
            )
        ]

        # Initialize Object Counters for Polygon 2 if not in "Single Line" mode
        counters2 = []
        if not single_line:
            counters2 = [
                solutions.ObjectCounter(
                    view_img=False,
                    reg_pts=polygons.get("Polygon 2"),
                    classes_names=model.names,
                    draw_tracks=True,
                    line_thickness=2,
                )
            ]

        # Process video frames in a loop
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                st.info("Video processing is not success.")
                break

            # Perform object tracking on the current frame, filtering by specified classes
            tracks = model.track(frame, persist=True, show=False, classes=classes_to_count)

            # Apply object counting for each polygon
            for counter1 in counters1:
                frame = counter1.start_counting(frame, tracks, 50, 50)

            if not single_line:
                for counter2 in counters2:
                    frame = counter2.start_counting(frame, tracks, 900, 50)

            # Stream the annotated frame to the app
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

        # Release resources
        cap.release()
        cv2.destroyAllWindows()
    else:
        st.info("Please select a valid input and ensure the source is accessible.")

# Settings sliders
left_column, right_column = st.columns([1, 2])
with left_column:
    st.header("Settings")
    polygons = {"Polygon 1": [], "Polygon 2": []}

    # Default values for each polygon
    default_values = {
        "Polygon 1": [(280, 200), (0, 300), (580, 300), (600, 200)],
        "Polygon 2": [(650, 200), (650, 300), (1300, 300), (1000, 200)],
    }

    # Add a checkbox for "Single Line" functionality
    single_line = st.checkbox("Single Line")
    opposite = st.checkbox("Opposite")

    # Adjust values if "Single Line" is checked
    if single_line:
        default_values["Polygon 2"] = [(0, 0), (0, 0), (0, 0), (0, 0)]

    # Add a checkbox to trigger the coordinate swap
    if opposite:
        for polygon_name, points in default_values.items():
            points[0], points[2] = points[2], points[0]  # Swap (X1, Y1) with (X3, Y3)
            points[1], points[3] = points[3], points[1]  # Swap (X2, Y2) with (X4, Y4)

    # Polygons X, Y coordinates value
    for polygon_name, points in default_values.items():
        if polygon_name == "Polygon 2" and single_line:
            continue  # Skip Polygon 2 settings if "Single Line" is checked

        st.subheader(polygon_name)
        for j, (default_x, default_y) in enumerate(points, start=1):
            x = st.number_input(f"{polygon_name}  X{j}", value=default_x)
            y = st.number_input(f"{polygon_name}  Y{j}", value=default_y)
            polygons[polygon_name].append((x, y))

with right_column:
    st.header("Video Display")

    if option == 'Recorded Video':
        # File uploader for the video
        video = st.selectbox("Video File...", settings.VIDEOS_DICT.keys())

        if video is not None:
            # Open the video capture file using OpenCV
            cap = cv2.VideoCapture(str(video))

            # Process the video and record the output
            processed_video_path = process_video(cap, polygons, single_line)
            
            if processed_video_path:
                st.video(processed_video_path)

    elif option == 'Live Webcam':
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Error accessing the webcam.")
        else:
            processed_video_path = process_video(cap, polygons, single_line)
            if processed_video_path:
                st.video(processed_video_path)

    elif option == 'External Camera':
        external_camera_url = st.text_input("Enter the URL or ID of the external camera:")
        if external_camera_url:
            cap = cv2.VideoCapture(external_camera_url)
            if not cap.isOpened():
                st.error("Error accessing the external camera.")
            else:
                processed_video_path = process_video(cap, polygons, single_line)
                if processed_video_path:
                    st.video(processed_video_path)

    else:
        st.info("Please select a valid input and ensure the source is accessible.")
