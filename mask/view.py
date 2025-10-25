import cv2
import numpy as np
from PIL import Image

def annotate(video_frames):
    """
    Multi-object video annotation tool with pastel colors and robust undo.
    - Left-click: add point for active object
    - Number keys (1–9): switch active object
    - Backspace: undo last point for active object
    - Click/drag timeline: change frame
    - Arrow keys: navigate
    - Enter: save & exit
    """

    num_frames = len(video_frames)
    frame_idx = 0
    active_obj = 1
    dragging = False

    # Nested annotations: {frame_idx: {obj_id: [(x, y), ...]}}
    annotations = {i: {} for i in range(num_frames)}

    # Generate soft pastel color palette dynamically
    def get_color(obj_id):
        # Generate color from hue rotation
        hue = (obj_id * 35) % 180  # HSV hue
        color = np.uint8([[[hue, 120, 255]]])
        bgr = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)[0, 0]
        return tuple(int(c) for c in bgr)

    def get_frame(idx):
        img = Image.fromarray(video_frames[idx]).convert("RGB")
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def draw_overlay(img, idx):
        overlay = img.copy()
        h, w = overlay.shape[:2]

        # Gradient top bar
        for y in range(28):
            c = 30 + int(20 * (y / 28))
            cv2.line(overlay, (0, y), (w, y), (c, c, c), 1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, overlay)

        # Info text
        text = f"Frame {idx+1}/{num_frames} | Active Obj: {active_obj} | [Backspace]: Undo | [Enter]: Save"
        cv2.putText(overlay, text, (14, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.47,
                    (240, 240, 240), 1, cv2.LINE_AA)
        return overlay

    def draw_slider(current_idx, width, height=45):
        bar = np.full((height, width, 3), 40, dtype=np.uint8)
        for y in range(height):
            shade = 40 + int(15 * (y / height))
            cv2.line(bar, (0, y), (width, y), (shade, shade, shade), 1)
        progress_x = int((current_idx / (num_frames - 1)) * width)
        cv2.rectangle(bar, (0, 0), (progress_x, height), (100, 150, 255), -1, cv2.LINE_AA)
        handle_x = progress_x
        cv2.circle(bar, (handle_x, height // 2), 7, (255, 190, 100), -1, cv2.LINE_AA)
        cv2.circle(bar, (handle_x, height // 2), 11, (255, 210, 130), 1, cv2.LINE_AA)
        return bar

    def animate_click(x, y, color):
        for r in range(4, 16, 2):
            temp = frame.copy()
            cv2.circle(temp, (x, y), r, color, 2, cv2.LINE_AA)
            overlay = draw_overlay(temp.copy(), frame_idx)
            draw_all_annotations(overlay)
            h, w = overlay.shape[:2]
            combined = np.vstack([overlay, draw_slider(frame_idx, w)])
            cv2.imshow("Video Annotator", combined)
            cv2.waitKey(10)
        cv2.circle(frame, (x, y), 5, color, -1)

    def draw_all_annotations(display):
        """Draw all objects’ points for the current frame."""
        if frame_idx not in annotations:
            return
        for obj_id, points in annotations[frame_idx].items():
            color = get_color(obj_id)
            for (x, y) in points:
                cv2.circle(display, (x, y), 5, color, -1)

    def update_display():
        display = draw_overlay(frame.copy(), frame_idx)
        draw_all_annotations(display)
        h, w = display.shape[:2]
        combined = np.vstack([display, draw_slider(frame_idx, w)])
        cv2.imshow("Video Annotator", combined)

    def click_event(event, x, y, flags, param):
        nonlocal frame_idx, frame, dragging
        h, w = frame.shape[:2]
        slider_y_start = h

        if event == cv2.EVENT_LBUTTONDOWN:
            if y > slider_y_start:  # Click slider
                dragging = True
                frame_idx = int((x / w) * (num_frames - 1))
                frame = get_frame(frame_idx)
                update_display()
            else:
                color = get_color(active_obj)
                annotations[frame_idx].setdefault(active_obj, []).append((x, y))
                animate_click(x, y, color)
                update_display()

        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            frame_idx = np.clip(int((x / w) * (num_frames - 1)), 0, num_frames - 1)
            frame = get_frame(frame_idx)
            update_display()

        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False

    # --- Init ---
    frame = get_frame(frame_idx)
    cv2.namedWindow("Video Annotator", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Video Annotator", click_event)
    update_display()

    while True:
        key = cv2.waitKey(30) & 0xFF

        if key == 13:  # ENTER
            break

        elif key in (8, 127):  # BACKSPACE
            # true undo: remove last point of active object
            if frame_idx in annotations and active_obj in annotations[frame_idx] and annotations[frame_idx][active_obj]:
                annotations[frame_idx][active_obj].pop()
            frame = get_frame(frame_idx)
            update_display()

        elif key == 81:  # ←
            frame_idx = max(0, frame_idx - 1)
            frame = get_frame(frame_idx)
            update_display()

        elif key == 83:  # →
            frame_idx = min(num_frames - 1, frame_idx + 1)
            frame = get_frame(frame_idx)
            update_display()

        elif ord('1') <= key <= ord('5'):  # switch object
            active_obj = key - ord('0')
            update_display()

    cv2.destroyAllWindows()
    return annotations

from transformers.video_utils import load_video
import json
video_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/bedroom.mp4"
video_frames, _ = load_video(video_url)

annotations = annotate(video_frames)

with open("annotations.json", "w") as f:
    json.dump({"annotations": annotations}, f)