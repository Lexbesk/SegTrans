# python run_ctrlx.py \
#     --structure_image assets/images/horse__point_cloud.jpg \
#     --appearance_image assets/images/horse.jpg \
#     --prompt "a photo of a horse standing on grass" \
#     --structure_prompt "a 3D point cloud of a horse"


# Below is the command to run our cool pipeline'=

str="assets/ex2/animals/camel_people_sketch.jpg"
# app="assets/ex1/elephant_zebra/ele_ze_4.jpg" "assets/cat_dog/cat_2.jpg" "assets/cat_dog/dog_2.jpg" "assets/cat_dog/cat_0.jpg" "assets/cat_dog/dog_0.jpg"
apps=("assets/ex2/animals/camel.jpg" "assets/ex2/animals/trump.jpg")
objects="camel. person."
device="cuda:5"

cd Grounded-SAM-2-main
python grounded_sam2_local_demo_multiple.py \
    --structure_image "$str" \
    --appearance_images "${apps[@]}" \
    --objects "$objects" \
    --device "$device" \
    --output_dir "outputs"
cd ..

python run_ctrlx_multiple.py \
    --structure_image "$str" \
    --appearance_images "${apps[@]}" \
    --prompt "" \
    --structure_prompt "" \
    --objects "$objects" \
    --device "$device" \
    --segmentation_output_dir "outputs" 
