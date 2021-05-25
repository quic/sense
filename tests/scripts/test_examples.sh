#!/bin/bash

# Run all example scripts on a short video file
test_video_path="tests/resources/test_video.mp4"
calorie_estimation_args="--weight=65 --age=30 --height=170 --gender=female --path_in=$test_video_path"

echo "Test Action Recognition demo ::"
PYTHONPATH=./ python examples/run_action_recognition.py --path_in=$test_video_path
PYTHONPATH=./ python examples/run_action_recognition.py --path_in=$test_video_path --model_version=lite
PYTHONPATH=./ python examples/run_action_recognition.py --path_in=$test_video_path --model_name=StridedInflatedMobileNetV2
PYTHONPATH=./ python examples/run_action_recognition.py --path_in=$test_video_path --model_name=StridedInflatedMobileNetV2 --model_version=lite

echo "Test Gesture Control demo ::"
PYTHONPATH=./ python examples/run_gesture_control.py --path_in=$test_video_path
PYTHONPATH=./ python examples/run_gesture_control.py --path_in=$test_video_path --model_version=lite

echo "Test Fitness Tracker demo ::"
PYTHONPATH=./ python examples/run_fitness_tracker.py $calorie_estimation_args
PYTHONPATH=./ python examples/run_fitness_tracker.py $calorie_estimation_args --model_version=lite
PYTHONPATH=./ python examples/run_fitness_tracker.py $calorie_estimation_args --model_name=StridedInflatedEfficientNet
PYTHONPATH=./ python examples/run_fitness_tracker.py $calorie_estimation_args --model_name=StridedInflatedEfficientNet --model_version=lite

echo "Test Calorie Estimation demo ::"
PYTHONPATH=./ python examples/run_calorie_estimation.py $calorie_estimation_args
PYTHONPATH=./ python examples/run_calorie_estimation.py $calorie_estimation_args --model_version=lite
PYTHONPATH=./ python examples/run_calorie_estimation.py $calorie_estimation_args --model_name=StridedInflatedEfficientNet
PYTHONPATH=./ python examples/run_calorie_estimation.py $calorie_estimation_args --model_name=StridedInflatedEfficientNet --model_version=lite

echo "Test Fitness Rep Counter demo ::"
PYTHONPATH=./ python examples/run_fitness_rep_counter.py --path_in=$test_video_path
