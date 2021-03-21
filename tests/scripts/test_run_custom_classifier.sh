#!/bin/bash

echo "Test Running Custom Classifier ::"
PYTHONPATH=./ python tools/run_custom_classifier.py --custom_classifier=tests/resources/checkpoint_dir --path_in=tests/resources/test_video.mp4
