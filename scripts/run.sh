# python run_ctrlx.py \
#     --structure_image assets/images/horse__point_cloud.jpg \
#     --appearance_image assets/images/horse.jpg \
#     --prompt "a photo of a horse standing on grass" \
#     --structure_prompt "a 3D point cloud of a horse"


# Below is the command to run our cool pipeline'=

str="assets/images/man4.jpeg"
# app="assets/ex1/elephant_zebra/ele_ze_4.jpg"
app="assets/images/horse.jpg"
objects="horse. man."
device="cuda:5"

cd Grounded-SAM-2-main
python grounded_sam2_local_demo.py \
    --structure_image "$str" \
    --appearance_image "$app" \
    --objects "$objects" \
    --device "$device" \
    --output_dir "outputs"
cd ..

python run_ctrlx.py \
    --structure_image "$str" \
    --appearance_image "$app" \
    --prompt "" \
    --structure_prompt "" \
    --objects "$objects" \
    --device "$device" \
    --segmentation_output_dir "outputs" 
