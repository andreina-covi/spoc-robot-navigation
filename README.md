# SPOC-Extended: Spatial-Semantic Navigation and Manipulation

This repository is an extended version of [SPOC](https://github.com/allenai/spoc-robot-training) (AllenAI, 2024).
It introduces new modules for spatial-semantic grounding, map-based navigation, and multimodal integration.


Script base:

This script is for generating csv files with spoc

python -m training.offline.online_eval --shuffle --eval_subset minival --output_basedir /home/andreina/Documents/Programs/Dataset/logs \
 --test_augmentation --task_type RoomVisit \
 --eval_set_size 1 \
 --input_sensors raw_navigation_camera raw_manipulation_camera last_actions an_object_is_in_hand \
 --house_set objaverse --wandb_logging False --num_workers 1 \
 --gpu_devices 0 --training_run_id SigLIP-ViTb-3-CHORES-S --local_checkpoint_dir /home/andreina/Documents/Programs/Dataset/checkpoints

This script is for processing and creating spatial descriptions of the csv files

python -m spatial_description_generation \
 --csv_path_navigation
 --csv_path_objects
 --json_path_navigation
 --json_path_spatial_rels
 --json_path_trajectories
 --episode_key

Usage example:

python -m spatial_description_generation \
 --csv_path_navigation /home/andreina/Documents/Programs/Dataset/Generated/navigation/12_05_2025_16_54_13_515802/navigation.csv \
 --csv_path_objects /home/andreina/Documents/Programs/Dataset/Generated/navigation/12_05_2025_16_54_13_515802/objects.csv \
 --json_path_navigation /home/andreina/Documents/Programs/Dataset/Generated/navigation/12_05_2025_16_54_13_515802/jsons/nav_1.json \
 --json_path_spatial_rels /home/andreina/Documents/Programs/Dataset/Generated/navigation/12_05_2025_16_54_13_515802/jsons/spa_rels_1.json \
 --json_path_trajectories /home/andreina/Documents/Programs/Dataset/Generated/navigation/12_05_2025_16_54_13_515802/jsons/traj_1.json \
 --episode_key episode_1