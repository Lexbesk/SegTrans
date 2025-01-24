from argparse import ArgumentParser
from datetime import datetime
from os import makedirs, path
from time import time

from diffusers import DDIMScheduler, StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image
import torch
import yaml
import json
from pathlib import Path
import pycocotools.mask as mask_util

from ctrl_x.pipelines.pipeline_sdxl import CtrlXStableDiffusionXLPipeline
from ctrl_x.pipelines.pipeline_sdxl import CtrlXMultiplePipeline
from ctrl_x.utils import *
from ctrl_x.utils.sdxl import *



def sam_load_image(image_path: str) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.open(image_path).convert("RGB")
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed


@torch.no_grad()
def inference(
        pipe, refiner, device,
        structure_image, appearance_images,
        prompt, structure_prompt, appearance_prompt,
        positive_prompt, negative_prompt,
        guidance_scale, structure_guidance_scale, appearance_guidance_scale,
        num_inference_steps, eta, seed,
        width, height,
        structure_schedule, appearance_schedule, objects, cross_maps=None
):
    seed_everything(seed)

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    control_config = get_control_config(structure_schedule, appearance_schedule)
    print(f"\nUsing the following control config:\n{control_config}\n")

    config = yaml.safe_load(control_config)

    config = {'control_schedule': {
                'encoder': {0: [[], [], []], 1: [[], [], [0.6, 0.6]], 2: [[], [], [0.6, 0.6]]},
                'middle': [[], [], []],
                'decoder': {0: [[0.6], [0.6, 0.6, 0.6], [0.0, 0.6, 0.6]], 1: [[], [], [0.6, 0.6]], 2: [[], [], []]}
                },
              'control_target': [['output_tensor'], ['query', 'key'], ['before']],
              'self_recurrence_schedule': [[0.1, 0.5, 2]]}

    register_control(
        model=pipe,
        timesteps=timesteps,
        control_schedule=config["control_schedule"],
        control_target=config["control_target"],
        device=device,
        cross_maps=cross_maps,
        multiple=True
    )

    pipe.safety_checker = None
    pipe.requires_safety_checker = False

    self_recurrence_schedule = get_self_recurrence_schedule(config["self_recurrence_schedule"], num_inference_steps)

    pipe.set_progress_bar_config(desc="Ctrl-X inference")
    result, structure, appearances = pipe(
        prompt=prompt,
        structure_prompt=structure_prompt,
        appearance_prompt=appearance_prompt,
        structure_image=structure_image,
        appearance_images=appearance_images,
        num_inference_steps=num_inference_steps,
        negative_prompt=negative_prompt,
        positive_prompt=positive_prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        structure_guidance_scale=structure_guidance_scale,
        appearance_guidance_scale=appearance_guidance_scale,
        eta=eta,
        output_type="pil",
        return_dict=False,
        control_schedule=config["control_schedule"],
        self_recurrence_schedule=self_recurrence_schedule,
        objects=objects
    )

    if refiner is not None:
        refiner.set_progress_bar_config(desc="Refiner")
        result_refiner = refiner(
            image=pipe.refiner_args["latents"],
            prompt=pipe.refiner_args["prompt"],
            negative_prompt=pipe.refiner_args["negative_prompt"],
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            guidance_rescale=0.7,
            num_images_per_prompt=1,
            eta=eta,
            output_type="pil",
        ).images

    else:
        result_refiner = [None]

    del pipe.refiner_args

    return result[0], result_refiner[0], structure[0]


def get_cross_image_attention(args):
    segmentation_output_dir = Path("Grounded-SAM-2-main/" + args.segmentation_output_dir)
    str_segmentation_output_dir = segmentation_output_dir / "structure_image"
    app_segmentation_output_dirs = []
    for i, app_img_path in enumerate(args.appearance_images):
        app_segmentation_output_dir = segmentation_output_dir / f"appearance_image_{i}"
        app_segmentation_output_dirs.append(app_segmentation_output_dir)
    
    with open(os.path.join(str_segmentation_output_dir, "grounded_sam2_local_image_demo_results.json"), 'r') as f:
        loaded_dict = json.load(f)
    str_annotations = loaded_dict.get('annotations')
    
    app_annotations_list = []
    for app_segmentation_output_dir in app_segmentation_output_dirs:
        with open(os.path.join(app_segmentation_output_dir, "grounded_sam2_local_image_demo_results.json"), 'r') as f:
            loaded_dict = json.load(f)
        app_annotations = loaded_dict.get('annotations')
        app_annotations_list.append(app_annotations)
    
    str_masks = []
    app_masks_list = []
    app_classnames = []
    
    for annotation in str_annotations:
        rle = annotation.get('segmentation')
        classname = annotation.get('class_name')
        score = annotation.get('score')
        rle_string = rle["counts"]
        rle_size = rle["size"]
        str_mask = mask_util.decode(rle)
        str_mask = torch.tensor(str_mask, dtype=torch.float32, device=args.device)
        str_masks.append((str_mask, classname, score))
      
    for app_annotations in app_annotations_list:  
        app_masks = []
        classnames = []
        for annotation in app_annotations:
            rle = annotation.get('segmentation')
            classname = annotation.get('class_name')
            score = annotation.get('score')
            rle_string = rle["counts"]
            rle_size = rle["size"]
            app_mask = mask_util.decode(rle)
            app_mask = torch.tensor(app_mask, dtype=torch.float32, device=args.device)
            app_masks.append((app_mask, classname, score))
            app_masks.sort(key=lambda x: x[2], reverse=True)
            if classname not in classnames:
                classnames.append(classname)
        app_masks_list.append(app_masks)
        app_classnames.append(classnames)

    str_masks.sort(key=lambda x: x[2], reverse=True)
    classnames = []
    for mask in str_masks:
        if mask[1] not in classnames:
            classnames.append(mask[1])
    # for mask in app_masks:
    #     if mask[1] not in classnames:
    #         classnames.append(mask[1])
    print("Structure Image Classes:", classnames)
    print("Appearance Image Classes (Multiple):", app_classnames)
    # print(str_masks[0][0].shape)
    cross_maps = {4096: None, 1024: None}
    cross_maps[4096], labels1 = create_cross_mask_multiple(str_masks, app_masks_list, app_classnames, 64, classnames, args.device)
    cross_maps[1024], labels2 = create_cross_mask_multiple(str_masks, app_masks_list, app_classnames, 32, classnames, args.device)
    # print(cross_maps[4096].shape, cross_maps[1024].shape)
    # print(cross_maps[1024][12])
    
    
    return cross_maps


@torch.no_grad()
def main(args):
    structure_image = None
    if args.structure_image is not None:
        structure_image = load_image(args.structure_image)

    appearance_images = []
    for image_path in args.appearance_images:
        appearance_image = load_image(image_path)
        appearance_images.append(appearance_image)
        
    
    cross_maps = get_cross_image_attention(args)

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
    refiner_id_or_path = "stabilityai/stable-diffusion-xl-refiner-1.0"
    device = args.device if torch.cuda.is_available() else "cpu"
    variant = "fp16" if device == args.device else "fp32"

    scheduler = DDIMScheduler.from_config(model_id_or_path,
                                          subfolder="scheduler")  # TODO: Support schedulers beyond DDIM

    if args.model is None:
        pipe = CtrlXMultiplePipeline.from_pretrained(
            model_id_or_path, scheduler=scheduler, torch_dtype=torch_dtype, variant=variant, use_safetensors=True,
        )
    else:
        print(f"Using weights {args.model} for SDXL base model.")
        pipe = CtrlXMultiplePipeline.from_single_file(args.model, scheduler=scheduler, torch_dtype=torch_dtype)

    refiner = None
    if not args.disable_refiner:
        refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            refiner_id_or_path, scheduler=scheduler, text_encoder_2=pipe.text_encoder_2, vae=pipe.vae,
            torch_dtype=torch_dtype, variant=variant, use_safetensors=True,
        )

    if args.model_offload or args.sequential_offload:
        try:
            import accelerate  # Checking if accelerate is installed for Model/CPU offloading
        except:
            raise ModuleNotFoundError("`accelerate` must be installed for Model/CPU offloading.")

        if args.sequential_offload:
            pipe.enable_sequential_cpu_offload()
            if refiner is not None:
                refiner.enable_sequential_cpu_offload()
        elif args.model_offload:
            pipe.enable_model_cpu_offload()
            if refiner is not None:
                refiner.enable_model_cpu_offload()

    else:
        pipe = pipe.to(device)
        if refiner is not None:
            refiner = refiner.to(device)

    model_load_print = "Base model "
    if not args.disable_refiner:
        model_load_print += "+ refiner "
    if args.sequential_offload:
        model_load_print += "loaded with sequential CPU offloading."
    elif args.model_offload:
        model_load_print += "loaded with model CPU offloading."
    else:
        model_load_print += "loaded."
    print(f"{model_load_print} Running on device: {device}.")

    t = time()

    result, result_refiner, structure= inference(
        pipe=pipe,
        refiner=refiner,
        device=device,
        structure_image=structure_image,
        appearance_images=appearance_images,
        prompt=args.prompt,
        structure_prompt=args.structure_prompt,
        appearance_prompt=args.appearance_prompt,
        positive_prompt=args.positive_prompt,
        negative_prompt=args.negative_prompt,
        guidance_scale=args.guidance_scale,
        structure_guidance_scale=args.structure_guidance_scale,
        appearance_guidance_scale=args.appearance_guidance_scale,
        num_inference_steps=args.num_inference_steps,
        eta=args.eta,
        seed=args.seed,
        width=args.width,
        height=args.height,
        structure_schedule=args.structure_schedule,
        appearance_schedule=args.appearance_schedule,
        objects=args.objects,
        cross_maps=cross_maps
    )

    makedirs(args.output_folder, exist_ok=True)
    prefix = "ctrlx__" + datetime.now().strftime("%Y%m%d_%H%M%S")
    structure.save(path.join(args.output_folder, f"{prefix}__structure.jpg"), quality=JPEG_QUALITY)
    # appearance.save(path.join(args.output_folder, f"{prefix}__appearance.jpg"), quality=JPEG_QUALITY)
    result.save(path.join(args.output_folder, f"{prefix}__result.jpg"), quality=JPEG_QUALITY)
    if result_refiner is not None:
        result_refiner.save(path.join(args.output_folder, f"{prefix}__result_refiner.jpg"), quality=JPEG_QUALITY)

    if args.benchmark:
        inference_time = time() - t
        peak_memory_usage = torch.cuda.max_memory_reserved()
        print(f"Inference time: {inference_time:.2f}s")
        print(f"Peak memory usage: {peak_memory_usage / pow(1024, 3):.2f}GiB")

    print("Done.")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--structure_image", "-si", type=str, default=None)
    # parser.add_argument("--appearance_image", "-ai", type=str, default=None)
    parser.add_argument("--appearance_images", nargs='+')

    parser.add_argument("--prompt", "-p", type=str, required=True)
    parser.add_argument("--structure_prompt", "-sp", type=str, default="")
    parser.add_argument("--appearance_prompt", "-ap", type=str, default="")

    parser.add_argument("--positive_prompt", "-pp", type=str, default="high quality")
    parser.add_argument("--negative_prompt", "-np", type=str, default="ugly, blurry, dark, low res, unrealistic")

    parser.add_argument("--guidance_scale", "-g", type=float, default=5.0)
    parser.add_argument("--structure_guidance_scale", "-sg", type=float, default=5.0)
    parser.add_argument("--appearance_guidance_scale", "-ag", type=float, default=5.0)

    parser.add_argument("--num_inference_steps", "-n", type=int, default=50)
    parser.add_argument("--eta", "-e", type=float, default=1.0)
    parser.add_argument("--seed", "-s", type=int, default=90095)

    parser.add_argument("--width", "-W", type=int, default=1024)
    parser.add_argument("--height", "-H", type=int, default=1024)

    parser.add_argument("--structure_schedule", "-ss", type=float, default=0.6)
    parser.add_argument("--appearance_schedule", "-as", type=float, default=0.6)

    parser.add_argument("--output_folder", "-o", type=str, default="./results")

    parser.add_argument(
        "-mo", "--model_offload", action="store_true",
        help="Model CPU offload, lowers memory usage with slight runtime increase. `accelerate` must be installed.",
    )
    parser.add_argument(
        "-so", "--sequential_offload", action="store_true",
        help=(
            "Sequential layer CPU offload, significantly lowers memory usage with massive runtime increase."
            "`accelerate` must be installed. If both model_offload and sequential_offload are set, then use the latter."
        ),
    )
    parser.add_argument("-r", "--disable_refiner", action="store_true")
    parser.add_argument("-m", "--model", type=str, default=None, help="Optionally, load model safetensors.")
    parser.add_argument("-b", "--benchmark", action="store_true", help="Show inference time and max memory usage.")
    parser.add_argument("--objects", type=str, default="tiger. deer.", help="Specify the distinct objects you want to transfer the appearance from the appearance image to the structure image. VERY important: text queries need to be lowercased + end with a dot.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the inference on.")
    parser.add_argument("--segmentation_output_dir", type=str, default="outputs")

    args = parser.parse_args()
    main(args)
