# SPOC-Extended: Spatial-Semantic Navigation and Manipulation

This repository is an extended version of [SPOC](https://github.com/allenai/spoc-robot-training) (AllenAI, 2024).
It introduces new modules for spatial-semantic grounding, map-based navigation, and multimodal integration.


Script base:

python -m training.offline.online_eval --shuffle --eval_subset minival --output_basedir /home/andreina/Documents/Programs/Dataset/logs \
 --test_augmentation --task_type RoomVisit \
 --eval_set_size 1 \
 --input_sensors raw_navigation_camera raw_manipulation_camera last_actions an_object_is_in_hand \
 --house_set objaverse --wandb_logging False --num_workers 1 \
 --gpu_devices 0 --training_run_id SigLIP-ViTb-3-CHORES-S --local_checkpoint_dir /home/andreina/Documents/Programs/Dataset/checkpoints