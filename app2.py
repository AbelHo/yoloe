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


def _get_det_field(detections, key): #FIXME find the right one and remove try/catch
    """Robustly get a field from detections as a plain Python list.
    Try attribute access, then item access, then fall back to empty list.
    """
    # try attribute
    try:
        val = getattr(detections, key)
    except Exception:
        try:
            val = detections[key]
        except Exception:
            return []
    # convert to plain Python list if possible
    try:
        if hasattr(val, 'tolist'):
            return val.tolist()
        return list(val)
    except Exception:
        return val

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
    result_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
    mask_video_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mkv')
    import json
    all_results = []
    model = init_model(model_id)
    model.set_classes([t.strip() for t in texts.split(',')], model.get_text_pe([t.strip() for t in texts.split(',')]))
    mask_out = cv2.VideoWriter(mask_video_file.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height), isColor=True)
    frame_idx = 0
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
        # Save mask as grayscale frame in mask video
        if hasattr(detections, 'mask') and detections.mask is not None and len(detections.mask.shape) == 3:
            mask_frame = np.any(detections.mask, axis=0).astype(np.uint8) * 255
        elif hasattr(detections, 'mask') and detections.mask is not None:
            mask_frame = detections.mask.astype(np.uint8) * 255
        else:
            mask_frame = np.zeros((height, width), dtype=np.uint8)
        # Ensure mask_frame is 3D (height, width, 1) for grayscale video
        if mask_frame.ndim == 2:
            mask_frame = np.expand_dims(mask_frame, axis=-1)
        # Convert to 3 channels for compatibility
        mask_frame = np.repeat(mask_frame, 3, axis=-1)
        mask_out.write(mask_frame)
        # Save bboxes, class names, confidences as JSON
        frame_result = {
            "frame": frame_idx,
            "bboxes": _get_det_field(detections, 'xyxy'),
            "class_name": _get_det_field(detections, 'class_name'),
            "confidence": _get_det_field(detections, 'confidence'),
        }
        all_results.append(frame_result)
        frame_idx += 1
    cap.release()
    out.release()
    mask_out.release()
    with open(result_file.name, 'w') as f:
        json.dump(all_results, f)
    return temp_out.name, result_file.name, mask_video_file.name
    cap.release()
    out.release()
    with open(result_file.name, 'wb') as f:
        f.write(msgpack.packb(all_results))
    return temp_out.name, result_file.name

def app2():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Upload Video", interactive=True)
                with gr.Tab("Text"):
                    texts = gr.Textbox(label="Input Texts", value='person,bus', placeholder='person,bus', interactive=True)
                    run_first_frame_text = gr.Button(value="Run on First Frame (Text)")
                    run_full_video_text = gr.Button(value="Run on Entire Video (Text)")
                with gr.Tab("Visual"):
                    visual_prompt_type = gr.Dropdown(choices=["bboxes", "masks"], value="bboxes", label="Visual Type", interactive=True)
                    run_first_frame_visual = gr.Button(value="Run on First Frame (Visual)")
                    run_full_video_visual = gr.Button(value="Run on Entire Video (Visual)")
                with gr.Tab("Prompt-Free"):
                    run_first_frame_pf = gr.Button(value="Run on First Frame (Prompt-Free)")
                    run_full_video_pf = gr.Button(value="Run on Entire Video (Prompt-Free)")
                model_id = gr.Dropdown(
                    label="Model",
                    choices=["yoloe-v8s", "yoloe-v8m", "yoloe-v8l", "yoloe-11s", "yoloe-11m", "yoloe-11l"],
                    value="yoloe-v8l",
                )
                image_size = gr.Slider(label="Image Size", minimum=320, maximum=1280, step=32, value=640)
                conf_thresh = gr.Slider(label="Confidence Threshold", minimum=0.0, maximum=1.0, step=0.05, value=0.25)
                iou_thresh = gr.Slider(label="IoU Threshold", minimum=0.0, maximum=1.0, step=0.05, value=0.70)
            with gr.Column():
                output_image = gr.Image(type="numpy", label="Annotated First Frame")
                output_video = gr.Video(label="Annotated Video")
                output_json = gr.File(label="Download Results File (JSON)")
                output_mask_video = gr.File(label="Download Mask Video (MKV)")

        # Text prompt handlers
        run_first_frame_text.click(
            fn=process_first_frame,
            inputs=[video_input, texts, model_id, image_size, conf_thresh, iou_thresh],
            outputs=[output_image],
        )
        run_full_video_text.click(
            fn=process_entire_video,
            inputs=[video_input, texts, model_id, image_size, conf_thresh, iou_thresh],
            outputs=[output_video, output_json, output_mask_video],
        )

        # Visual prompt handlers
        def process_first_frame_visual(video, visual_prompt_type, model_id, image_size, conf_thresh, iou_thresh):
            first_frame = extract_first_frame(video)
            # For demo, just run with dummy bbox/mask prompt
            if visual_prompt_type == "bboxes":
                # Dummy bbox: whole image
                w, h = first_frame.size
                bboxes = np.array([[0, 0, w, h]])
                prompts = {"bboxes": bboxes, "cls": np.array([0])}
            else:
                mask = np.ones((first_frame.size[1], first_frame.size[0]), dtype=np.uint8)
                prompts = {"masks": mask[None], "cls": np.array([0])}
            model = init_model(model_id)
            results = model.predict(source=first_frame, imgsz=image_size, conf=conf_thresh, iou=iou_thresh, prompts=prompts)
            detections = sv.Detections.from_ultralytics(results[0])
            return annotate_image(first_frame, detections)

        def process_entire_video_visual(video, visual_prompt_type, model_id, image_size, conf_thresh, iou_thresh):
            cap = cv2.VideoCapture(video)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            temp_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            out = cv2.VideoWriter(temp_out.name, fourcc, fps, (width, height))
            result_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
            mask_video_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mkv')
            import json
            all_results = []
            model = init_model(model_id)
            mask_out = cv2.VideoWriter(mask_video_file.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height), isColor=True)
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if visual_prompt_type == "bboxes":
                    w, h = pil_img.size
                    bboxes = np.array([[0, 0, w, h]])
                    prompts = {"bboxes": bboxes, "cls": np.array([0])}
                else:
                    mask = np.ones((pil_img.size[1], pil_img.size[0]), dtype=np.uint8)
                    prompts = {"masks": mask[None], "cls": np.array([0])}
                results = model.predict(source=pil_img, imgsz=image_size, conf=conf_thresh, iou=iou_thresh, prompts=prompts)
                detections = sv.Detections.from_ultralytics(results[0])
                annotated = annotate_image(pil_img, detections)
                annotated_np = cv2.cvtColor(np.array(annotated), cv2.COLOR_RGB2BGR)
                out.write(annotated_np)
                # Save mask as grayscale frame in mask video
                if hasattr(detections, 'mask') and detections.mask is not None and len(detections.mask.shape) == 3:
                    mask_frame = np.any(detections.mask, axis=0).astype(np.uint8) * 255
                elif hasattr(detections, 'mask') and detections.mask is not None:
                    mask_frame = detections.mask.astype(np.uint8) * 255
                else:
                    mask_frame = np.zeros((height, width), dtype=np.uint8)
                if mask_frame.ndim == 2:
                    mask_frame = np.expand_dims(mask_frame, axis=-1)
                mask_frame = np.repeat(mask_frame, 3, axis=-1)
                mask_out.write(mask_frame)
                frame_result = {
                    "frame": frame_idx,
                    "bboxes": _get_det_field(detections, 'xyxy'),
                    "class_name": _get_det_field(detections, 'class_name'),
                    "confidence": _get_det_field(detections, 'confidence'),
                }
                all_results.append(frame_result)
                frame_idx += 1
            cap.release()
            out.release()
            mask_out.release()
            with open(result_file.name, 'w') as f:
                json.dump(all_results, f)
            return temp_out.name, result_file.name, mask_video_file.name

        run_first_frame_visual.click(
            fn=process_first_frame_visual,
            inputs=[video_input, visual_prompt_type, model_id, image_size, conf_thresh, iou_thresh],
            outputs=[output_image],
        )
        run_full_video_visual.click(
            fn=process_entire_video_visual,
            inputs=[video_input, visual_prompt_type, model_id, image_size, conf_thresh, iou_thresh],
            outputs=[output_video, output_json, output_mask_video],
        )

        # Prompt-Free handlers
        def process_first_frame_pf(video, model_id, image_size, conf_thresh, iou_thresh):
            first_frame = extract_first_frame(video)
            with open('tools/ram_tag_list.txt', 'r') as f:
                texts = [x.strip() for x in f.readlines()]
            annotated = yoloe_inference(first_frame, texts, model_id, image_size, conf_thresh, iou_thresh)
            return annotated

        def process_entire_video_pf(video, model_id, image_size, conf_thresh, iou_thresh):
            cap = cv2.VideoCapture(video)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            temp_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            out = cv2.VideoWriter(temp_out.name, fourcc, fps, (width, height))
            result_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
            mask_video_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mkv')
            import json
            all_results = []
            with open('tools/ram_tag_list.txt', 'r') as f:
                texts = [x.strip() for x in f.readlines()]
            model = init_model(model_id)
            model.set_classes(texts, model.get_text_pe(texts))
            mask_out = cv2.VideoWriter(mask_video_file.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height), isColor=True)
            frame_idx = 0
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
                # Save mask as grayscale frame in mask video
                if hasattr(detections, 'mask') and detections.mask is not None and len(detections.mask.shape) == 3:
                    mask_frame = np.any(detections.mask, axis=0).astype(np.uint8) * 255
                elif hasattr(detections, 'mask') and detections.mask is not None:
                    mask_frame = detections.mask.astype(np.uint8) * 255
                else:
                    mask_frame = np.zeros((height, width), dtype=np.uint8)
                if mask_frame.ndim == 2:
                    mask_frame = np.expand_dims(mask_frame, axis=-1)
                mask_frame = np.repeat(mask_frame, 3, axis=-1)
                mask_out.write(mask_frame)
                frame_result = {
                    "frame": frame_idx,
                    "bboxes": _get_det_field(detections, 'xyxy'),
                    "class_name": _get_det_field(detections, 'class_name'),
                    "confidence": _get_det_field(detections, 'confidence'),
                }
                all_results.append(frame_result)
                frame_idx += 1
            cap.release()
            out.release()
            mask_out.release()
            with open(result_file.name, 'w') as f:
                json.dump(all_results, f)
            return temp_out.name, result_file.name, mask_video_file.name

        run_first_frame_pf.click(
            fn=process_first_frame_pf,
            inputs=[video_input, model_id, image_size, conf_thresh, iou_thresh],
            outputs=[output_image],
        )
        run_full_video_pf.click(
            fn=process_entire_video_pf,
            inputs=[video_input, model_id, image_size, conf_thresh, iou_thresh],
            outputs=[output_video, output_json, output_mask_video],
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
