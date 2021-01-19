#!/bin/bash

# ------------------------------------------------------------------------------
# Sense SHORTCUTS
# ------------------------------------------------------------------------------
shopt -s expand_aliases
# Demo scripts test run for each files
alias demo_calorie_estimation='PYTHONPATH=./ python scripts/run_calorie_estimation.py --weight=65 --age=30 --height=170 --gender=female --path_in=tests/resources/test_video.mp4'

alias demo_fitness_rep_counter='PYTHONPATH=./ python scripts/run_fitness_rep_counter.py --path_in=tests/resources/test_video.mp4'

alias demo_fitness_tracker='PYTHONPATH=./ python scripts/run_fitness_tracker.py --weight=65 --age=30 --height=170 --gender=female --path_in=tests/resources/test_video.mp4'

alias demo_gesture_recognition='PYTHONPATH=./ python scripts/run_gesture_recognition.py --path_in=tests/resources/test_video.mp4'

# ------------------------------------------------------------------------------
# CHAINED COMMANDS
# ------------------------------------------------------------------------------

# Run all demo files at once
test_demos() {
    echo "Test Calorie Estimation demo ::"
    demo_calorie_estimation

    echo "Test Fitness Rep Counter demo ::"
    demo_fitness_rep_counter

    echo "Test Fitness Tracker demo ::"
    demo_fitness_tracker

    echo "Test Gesture Recognition demo ::"
    demo_gesture_recognition
}

# Run the created function
test_demos
echo "All tests passed!"
