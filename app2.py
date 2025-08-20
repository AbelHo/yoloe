import torch
import numpy as np
import gradio as gr
import supervision as sv
import cv2
from ultralytics import YOLOE
from huggingface_hub import hf_hub_download
from PIL import Image
import tempfile
import os

def init_model(model_id):
    filename = f"{model_id}-seg.pt"
    path = hf_hub_download(repo_id="jameslahm/yoloe", filename=filename)
    model = YOLOE(path)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model

def extract_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError("Could not read first frame from video.")
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def annotate_image(image, detections):
    resolution_wh = image.size
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh)
    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence in zip(detections['class_name'], detections.confidence)
    ]
    annotated_image = image.copy()
    annotated_image = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX, opacity=0.4).annotate(scene=annotated_image, detections=detections)
    annotated_image = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX, thickness=thickness).annotate(scene=annotated_image, detections=detections)
    annotated_image = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX, text_scale=text_scale, smart_position=True).annotate(scene=annotated_image, detections=detections, labels=labels)
    return annotated_image

def yoloe_inference(image, texts, model_id, image_size, conf_thresh, iou_thresh):
    model = init_model(model_id)
    model.set_classes(texts, model.get_text_pe(texts))
    results = model.predict(source=image, imgsz=image_size, conf=conf_thresh, iou=iou_thresh)
    detections = sv.Detections.from_ultralytics(results[0])
    return annotate_image(image, detections)

def process_first_frame(video, texts, model_id, image_size, conf_thresh, iou_thresh):
    first_frame = extract_first_frame(video)
    annotated = yoloe_inference(first_frame, [t.strip() for t in texts.split(',')], model_id, image_size, conf_thresh, iou_thresh)
    return annotated

def process_entire_video(video, texts, model_id, image_size, conf_thresh, iou_thresh):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    out = cv2.VideoWriter(temp_out.name, fourcc, fps, (width, height))
    model = init_model(model_id)
    model.set_classes([t.strip() for t in texts.split(',')], model.get_text_pe([t.strip() for t in texts.split(',')]))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = model.predict(source=pil_img, imgsz=image_size, conf=conf_thresh, iou=iou_thresh)
        detections = sv.Detections.from_ultralytics(results[0])
        annotated = annotate_image(pil_img, detections)
        annotated_np = cv2.cvtColor(np.array(annotated), cv2.COLOR_RGB2BGR)
        out.write(annotated_np)
    cap.release()
    out.release()
    return temp_out.name

def app2():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Upload Video", interactive=True)
                texts = gr.Textbox(label="Input Texts", value='person,bus', placeholder='person,bus', interactive=True)
                model_id = gr.Dropdown(
                    label="Model",
                    choices=["yoloe-v8s", "yoloe-v8m", "yoloe-v8l", "yoloe-11s", "yoloe-11m", "yoloe-11l"],
                    value="yoloe-v8l",
                )
                image_size = gr.Slider(label="Image Size", minimum=320, maximum=1280, step=32, value=640)
                conf_thresh = gr.Slider(label="Confidence Threshold", minimum=0.0, maximum=1.0, step=0.05, value=0.25)
                iou_thresh = gr.Slider(label="IoU Threshold", minimum=0.0, maximum=1.0, step=0.05, value=0.70)
                run_first_frame = gr.Button(value="Run on First Frame")
                run_full_video = gr.Button(value="Run on Entire Video")
            with gr.Column():
                output_image = gr.Image(type="numpy", label="Annotated First Frame")
                output_video = gr.Video(label="Annotated Video")
        run_first_frame.click(
            fn=process_first_frame,
            inputs=[video_input, texts, model_id, image_size, conf_thresh, iou_thresh],
            outputs=[output_image],
        )
        run_full_video.click(
            fn=process_entire_video,
            inputs=[video_input, texts, model_id, image_size, conf_thresh, iou_thresh],
            outputs=[output_video],
        )

gradio_app2 = gr.Blocks()
with gradio_app2:
    gr.Markdown("""
    # YOLOE Video Text Prompt Demo
    Upload a video and enter text prompts (e.g., 'person,bus').
    - Run on First Frame: Shows annotated first frame.
    - Run on Entire Video: Processes all frames and outputs annotated video.
    """)
    app2()

if __name__ == '__main__':
    gradio_app2.launch(allowed_paths=["figures"], server_name="0.0.0.0", server_port=7860)#, share=True
