import torch
import numpy as np
import gradio as gr
import supervision as sv
from scipy.ndimage import binary_fill_holes
from ultralytics import YOLOE
from ultralytics.utils.torch_utils import smart_inference_mode
from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor
from gradio_image_prompter import ImagePrompter
from huggingface_hub import hf_hub_download
import cv2
import tempfile
import os

import argparse

def init_model(model_id, is_pf=False):
    filename = f"{model_id}-seg.pt" if not is_pf else f"{model_id}-seg-pf.pt"
    path = hf_hub_download(repo_id="jameslahm/yoloe", filename=filename)
    model = YOLOE(path)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model

@smart_inference_mode()
def yoloe_inference(image, prompts, target_image, model_id, image_size, conf_thresh, iou_thresh, prompt_type):
    model = init_model(model_id)
    kwargs = {}
    if prompt_type == "Text":
        texts = prompts["texts"]
        model.set_classes(texts, model.get_text_pe(texts))
    elif prompt_type == "Visual":
        kwargs = dict(
            prompts=prompts,
            predictor=YOLOEVPSegPredictor
        )
        if target_image:
            model.predict(source=image, imgsz=image_size, conf=conf_thresh, iou=iou_thresh, return_vpe=True, **kwargs)
            model.set_classes(["object0"], model.predictor.vpe)
            model.predictor = None  # unset VPPredictor
            image = target_image
            kwargs = {}
    elif prompt_type == "Prompt-free":
        vocab = model.get_vocab(prompts["texts"])
        model = init_model(model_id, is_pf=True)
        model.set_vocab(vocab, names=prompts["texts"])
        model.model.model[-1].is_fused = True
        model.model.model[-1].conf = 0.001
        model.model.model[-1].max_det = 1000

    results = model.predict(source=image, imgsz=image_size, conf=conf_thresh, iou=iou_thresh, **kwargs)
    detections = sv.Detections.from_ultralytics(results[0])

    resolution_wh = image.size
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh)

    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(detections['class_name'], detections.confidence)
    ]

    annotated_image = image.copy()
    annotated_image = sv.MaskAnnotator(
        color_lookup=sv.ColorLookup.INDEX,
        opacity=0.4
    ).annotate(scene=annotated_image, detections=detections)
    annotated_image = sv.BoxAnnotator(
        color_lookup=sv.ColorLookup.INDEX,
        thickness=thickness
    ).annotate(scene=annotated_image, detections=detections)
    annotated_image = sv.LabelAnnotator(
        color_lookup=sv.ColorLookup.INDEX,
        text_scale=text_scale,
        smart_position=True
    ).annotate(scene=annotated_image, detections=detections, labels=labels)

    return annotated_image

def video_to_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError("Could not read video file.")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    from PIL import Image
    return Image.fromarray(frame_rgb)

def process_video(video_path, prompts, target_image, model_id, image_size, conf_thresh, iou_thresh, prompt_type, visual_prompt_type, visual_usage_type):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    out = cv2.VideoWriter(temp_out.name, fourcc, fps, (width, height))
    from PIL import Image
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)
        # Use same prompts for all frames
        annotated = yoloe_inference(pil_frame, prompts, target_image, model_id, image_size, conf_thresh, iou_thresh, prompt_type)
        annotated_np = np.array(annotated)
        annotated_bgr = cv2.cvtColor(annotated_np, cv2.COLOR_RGB2BGR)
        out.write(annotated_bgr)
    cap.release()
    out.release()
    return temp_out.name

def app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    video_file = gr.Video(label="Video File", visible=True, interactive=True)
                    first_frame_image = gr.Image(type="pil", label="First Frame", visible=True, interactive=True)
                    box_image = ImagePrompter(type="pil", label="DrawBox", visible=True, interactive=True)
                    mask_image = gr.ImageEditor(type="pil", label="DrawMask", visible=True, interactive=True, layers=False, canvas_size=(640, 640))
                    target_image = gr.Image(type="pil", label="Target Image", visible=False, interactive=True)

                yoloe_infer_first = gr.Button(value="Detect & Segment First Frame")
                yoloe_infer_video = gr.Button(value="Detect & Segment Entire Video")
                prompt_type = gr.Textbox(value="Visual", visible=False)
                visual_prompt_type = gr.Dropdown(choices=["bboxes", "masks"], value="bboxes", label="Visual Type", interactive=True)
                visual_usage_type = gr.Radio(choices=["Intra-Image", "Cross-Image"], value="Intra-Image", label="Intra/Cross Image", interactive=True)
                model_id = gr.Dropdown(
                    label="Model",
                    choices=[
                        "yoloe-v8s",
                        "yoloe-v8m",
                        "yoloe-v8l",
                        "yoloe-11s",
                        "yoloe-11m",
                        "yoloe-11l",
                    ],
                    value="yoloe-v8l",
                )
                image_size = gr.Slider(
                    label="Image Size",
                    minimum=320,
                    maximum=1280,
                    step=32,
                    value=640,
                )
                conf_thresh = gr.Slider(
                    label="Confidence Threshold",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.25,
                )
                iou_thresh = gr.Slider(
                    label="IoU Threshold",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.70,
                )
            with gr.Column():
                output_image = gr.Image(type="numpy", label="Annotated First Frame", visible=True)
                output_video = gr.Video(label="Annotated Video", visible=True)

        def extract_first_frame_and_populate(video_file):
            if video_file is None:
                return None, gr.update(value=None), gr.update(value=None)
            first_frame = video_to_first_frame(video_file)
            # Automatically populate box_image and mask_image with the first frame
            box_val = {"image": first_frame, "points": []}
            # For mask, show the first frame as background and let user draw mask on it
            mask_val = {"background": first_frame, "layers": [], "composite": first_frame}
            return first_frame, gr.update(value=box_val), gr.update(value=mask_val)

        video_file.change(
            fn=extract_first_frame_and_populate,
            inputs=[video_file],
            outputs=[first_frame_image, box_image, mask_image]
        )

        def run_inference_first_frame(video_file, box_image, mask_image, target_image, model_id, image_size, conf_thresh, iou_thresh, prompt_type, visual_prompt_type, visual_usage_type):
            image = video_to_first_frame(video_file)
            prompts = None
            if visual_prompt_type == "bboxes":
                points = box_image["points"]
                points = np.array(points)
                if len(points) == 0:
                    gr.Warning("No boxes are provided. No image output.", visible=True)
                    return gr.update(value=None)
                bboxes = np.array([p[[0, 1, 3, 4]] for p in points if p[2] == 2])
                prompts = {
                    "bboxes": bboxes,
                    "cls": np.array([0] * len(bboxes))
                }
            elif visual_prompt_type == "masks":
                    mask_layer = mask_image["layers"][0]
                    # Accept both PIL and numpy
                    if hasattr(mask_layer, "convert"):
                        mask_arr = np.array(mask_layer.convert("L"))
                    else:
                        mask_arr = np.array(mask_layer)
                    mask_arr = binary_fill_holes(mask_arr).astype(np.uint8)
                    mask_arr[mask_arr > 0] = 1
                    if mask_arr.sum() == 0:
                        gr.Warning("No masks are provided. No image output.", visible=True)
                        return gr.update(value=None)
                    prompts = {
                        "masks": mask_arr[None],
                        "cls": np.array([0])
                    }
            return yoloe_inference(image, prompts, target_image, model_id, image_size, conf_thresh, iou_thresh, prompt_type)

        yoloe_infer_first.click(
            fn=run_inference_first_frame,
            inputs=[video_file, box_image, mask_image, target_image, model_id, image_size, conf_thresh, iou_thresh, prompt_type, visual_prompt_type, visual_usage_type],
            outputs=[output_image],
        )

        def run_inference_video(video_file, box_image, mask_image, target_image, model_id, image_size, conf_thresh, iou_thresh, prompt_type, visual_prompt_type, visual_usage_type):
            first_frame = video_to_first_frame(video_file)
            prompts = None
            if visual_prompt_type == "bboxes":
                points = box_image["points"]
                points = np.array(points)
                if len(points) == 0:
                    gr.Warning("No boxes are provided. No video output.", visible=True)
                    return gr.update(value=None)
                bboxes = np.array([p[[0, 1, 3, 4]] for p in points if p[2] == 2])
                prompts = {
                    "bboxes": bboxes,
                    "cls": np.array([0] * len(bboxes))
                }
            elif visual_prompt_type == "masks":
                    mask_layer = mask_image["layers"][0]
                    if hasattr(mask_layer, "convert"):
                        mask_arr = np.array(mask_layer.convert("L"))
                    else:
                        mask_arr = np.array(mask_layer)
                    mask_arr = binary_fill_holes(mask_arr).astype(np.uint8)
                    mask_arr[mask_arr > 0] = 1
                    if mask_arr.sum() == 0:
                        gr.Warning("No masks are provided. No video output.", visible=True)
                        return gr.update(value=None)
                    prompts = {
                        "masks": mask_arr[None],
                        "cls": np.array([0])
                    }
            video_path = process_video(video_file, prompts, target_image, model_id, image_size, conf_thresh, iou_thresh, prompt_type, visual_prompt_type, visual_usage_type)
            return video_path

        yoloe_infer_video.click(
            fn=run_inference_video,
            inputs=[video_file, box_image, mask_image, target_image, model_id, image_size, conf_thresh, iou_thresh, prompt_type, visual_prompt_type, visual_usage_type],
            outputs=[output_video],
        )

if __name__ == '__main__':
    gradio_app = gr.Blocks()
    with gradio_app:
        gr.HTML(
            """
        <h1 style='text-align: center'>
        <img src="/file=figures/logo.png" width="2.5%" style="display:inline;padding-bottom:4px">
        YOLOE: Real-Time Seeing Anything (Video Visual Prompt)
        </h1>
        """
        )
        gr.Markdown(
            """
            This demo allows you to run YOLOE with visual prompts on video files. You can draw bounding boxes or masks on the first frame, and run inference on the first frame or the entire video.
            """
        )
        with gr.Row():
            with gr.Column():
                app()

    parser = argparse.ArgumentParser(description="Launch YOLOE Gradio app.")
    parser.add_argument("--port", type=int, default=7860, help="Port number to run the app on.")
    args = parser.parse_args()

    gradio_app.launch(allowed_paths=["figures"], server_name="0.0.0.0", server_port=args.port)#, share=True
