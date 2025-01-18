import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from argparse import ArgumentParser

"""
Hyper parameters
"""
# TEXT_PROMPT = "tiger. deer."
# IMG_PATH = "notebooks/images/t_d_a_0.jpg"
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# OUTPUT_DIR = Path("outputs/grounded_sam2_local_demo")
DUMP_JSON_RESULTS = True


def main(args):
    DEVICE = args.device
    OUTPUT_DIR = Path(args.output_dir)
    STR_IMG_PATH = "../" + args.structure_image
    # APP_IMG_PATH = "../" + args.appearance_image
    OBJECTS = args.objects

    # create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # environment settings
    # use bfloat16

    # build SAM2 image predictor
    sam2_checkpoint = SAM2_CHECKPOINT
    model_cfg = SAM2_MODEL_CONFIG
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    # build grounding dino model
    grounding_model = load_model(
        model_config_path=GROUNDING_DINO_CONFIG, 
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
        device=DEVICE
    )
    
    for i, app_img_path in enumerate(args.appearance_images):
        APP_IMG_PATH = "../" + app_img_path
        
        app_output_dir = OUTPUT_DIR / f"appearance_image_{i}"
        app_output_dir.mkdir(parents=True, exist_ok=True)
        segmentation(OBJECTS, APP_IMG_PATH, sam2_predictor, grounding_model, app_output_dir)
    
    str_output_dir = OUTPUT_DIR / "structure_image"
    str_output_dir.mkdir(parents=True, exist_ok=True)
    segmentation(OBJECTS, STR_IMG_PATH, sam2_predictor, grounding_model, str_output_dir)
    
    # app_output_dir = OUTPUT_DIR / "appearance_image"
    # app_output_dir.mkdir(parents=True, exist_ok=True)
    # segmentation(OBJECTS, APP_IMG_PATH, sam2_predictor, grounding_model, app_output_dir)


# setup the input image and text prompt for SAM 2 and Grounding DINO
# VERY important: text queries need to be lowercased + end with a dot
# text = TEXT_PROMPT
# img_path = IMG_PATH

def segmentation(objects, img_path, sam2_predictor, grounding_model, output_dir):
    image_source, image = load_image(img_path)
    sam2_predictor.set_image(image_source)

    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=objects,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    )

    # process the box prompt for SAM 2
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()


    # FIXME: figure how does this influence the G-DINO model
    # torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    """
    Post-process the output of the model to get the masks, scores, and logits for visualization
    """
    # convert the shape to (n, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)


    confidences = confidences.numpy().tolist()
    class_names = labels

    class_ids = np.array(list(range(len(class_names))))

    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(class_names, confidences)
    ]

    """
    Visualize image with supervision useful API
    """
    img = cv2.imread(img_path)
    detections = sv.Detections(
        xyxy=input_boxes,  # (n, 4)
        mask=masks.astype(bool),  # (n, h, w)
        class_id=class_ids
    )

    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    cv2.imwrite(os.path.join(output_dir, "groundingdino_annotated_image.jpg"), annotated_frame)

    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    cv2.imwrite(os.path.join(output_dir, "grounded_sam2_annotated_image_with_mask.jpg"), annotated_frame)

    """
    Dump the results in standard format and save as json files
    """
    if DUMP_JSON_RESULTS:
        # convert mask into rle format
        mask_rles = [single_mask_to_rle(mask) for mask in masks]

        input_boxes = input_boxes.tolist()
        scores = scores.tolist()
        # save the results in standard format
        results = {
            "image_path": img_path,
            "annotations" : [
                {
                    "class_name": class_name,
                    "bbox": box,
                    "segmentation": mask_rle,
                    "score": score,
                }
                for class_name, box, mask_rle, score in zip(class_names, input_boxes, mask_rles, scores)
            ],
            "box_format": "xyxy",
            "img_width": w,
            "img_height": h,
        }
        
        with open(os.path.join(output_dir, "grounded_sam2_local_image_demo_results.json"), "w") as f:
            json.dump(results, f, indent=4)
            

def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    # print(rle)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle
        

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--structure_image", "-si", type=str, default=None)
    # parser.add_argument("--appearance_image", "-ai", type=str, default=None)
    parser.add_argument("--appearance_images", nargs='+')

    parser.add_argument("--objects", type=str, default="tiger. deer.", help="Specify the distinct objects you want to transfer the appearance from the appearance image to the structure image. VERY important: text queries need to be lowercased + end with a dot.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the inference on.")

    parser.add_argument("--output_dir", type=str, default="outputs/grounded_sam2_local_demo")
    args = parser.parse_args()
    main(args)
