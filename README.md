# SPOC-Extended: Spatial-Semantic Navigation and Manipulation

This repository is an extended version of [SPOC](https://github.com/allenai/spoc-robot-training) (AllenAI, 2024).
It introduces new modules for spatial-semantic grounding, map-based navigation, and multimodal integration.


Script base:

This script is for generating csv files with spoc:

```bash
python -m training.offline.online_eval --shuffle --eval_subset minival --output_basedir /home/andreina/Documents/Programs/Dataset/logs \
 --test_augmentation --task_type RoomVisit \
 --eval_set_size 1 \
 --input_sensors raw_navigation_camera raw_manipulation_camera last_actions an_object_is_in_hand \
 --house_set objaverse --wandb_logging False --num_workers 1 \
 --gpu_devices 0 --training_run_id SigLIP-ViTb-3-CHORES-S --local_checkpoint_dir /home/andreina/Documents/Programs/Dataset/checkpoints
```


This script is for processing and creating spatial descriptions of the csv files

```bash
python -m spatial_description_generation \
 --csv_path_navigation filename \
 --csv_path_objects filename \
 --json_path_navigation filename \
 --json_path_spatial_rels filename \
 --json_path_trajectories filename
```
# --episode_key string

Usage example:

```bash
python -m spatial_description_generation \
 --csv_path_navigation /home/andreina/Documents/Programs/Dataset/Generated/navigation/03_16_2026_15_15_36_850440/navigation.csv \
 --csv_path_objects /home/andreina/Documents/Programs/Dataset/Generated/navigation/03_16_2026_15_15_36_850440/objects.csv \
 --json_path_navigation /home/andreina/Documents/Programs/Dataset/Generated/navigation/03_16_2026_15_15_36_850440/jsons/nav_1.json \
 --json_path_spatial_rels /home/andreina/Documents/Programs/Dataset/Generated/navigation/03_16_2026_15_15_36_850440/jsons/spa_rels_1.json \
 --json_path_trajectories /home/andreina/Documents/Programs/Dataset/Generated/navigation/03_16_2026_15_15_36_850440/jsons/traj_1.json \
 --image_path /home/andreina/Documents/Programs/Dataset/Generated/navigation/03_16_2026_15_15_36_850440/image.png
```

# --episode_key episode_1
