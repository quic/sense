#!/bin/bash

# Run test on all demo files
echo "Test Calorie Estimation demo ::"
PYTHONPATH=./ python examples/run_calorie_estimation.py --weight=65 --age=30 --height=170 --gender=female --path_in=tests/resources/test_video.mp4

echo "Test Fitness Rep Counter demo ::"
PYTHONPATH=./ python examples/run_fitness_rep_counter.py --path_in=tests/resources/test_video.mp4

echo "Test Fitness Tracker demo ::"
PYTHONPATH=./ python examples/run_fitness_tracker.py --weight=65 --age=30 --height=170 --gender=female --path_in=tests/resources/test_video.mp4

echo "Test Gesture Recognition demo ::"
PYTHONPATH=./ python examples/run_gesture_recognition.py --path_in=tests/resources/test_video.mp4
