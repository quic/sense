################
#   Original   #
################

# LAB2INT = {
#     "\"Drinking\" gesture": 0,
#     "\"Sleeping\" gesture": 1,
#     "Calling someone closer": 2,
#     "Covering ears": 3,
#     "Covering eyes": 4,
#     "Dabbing": 5,
#     "Doing nothing": 6,
#     "Doing other things": 7,
#     "Facepalming": 8,
#     "No person visible": 9,
#     "Nodding": 10,
#     "Pointing left": 11,
#     "Pointing right": 12,
#     "Pointing to the camera": 13,
#     "Putting finger to mouth": 14,
#     "Rolling hand": 15,
#     "Scratching": 16,
#     "Shaking head": 17,
#     "Showing the middle finger": 18,
#     "Swiping down": 19,
#     "Swiping down (with two hands)": 20,
#     "Swiping left": 21,
#     "Swiping right": 22,
#     "Swiping up": 23,
#     "Swiping up (with two hands)": 24,
#     "Thumb down": 25,
#     "Thumb up": 26,
#     "Waving": 27,
#     "Zooming in": 28,
#     "Zooming out": 29
# }
#
# INT2LAB = {value: key for key, value in LAB2INT.items()}
#
# LAB_THRESHOLDS = {
#     "\"Drinking\" gesture": .2,
#     "\"Sleeping\" gesture": .7,
#     "Calling someone closer": .5,
#     "Covering ears": .5,
#     "Covering eyes": .5,
#     "Dabbing": .5,
#     "Facepalming": .5,
#     "Nodding": .5,
#     "Pointing left": .3,
#     "Pointing right": .3,
#     "Pointing to the camera": .5,
#     "Putting finger to mouth": .5,
#     "Rolling hand": .5,
#     "Scratching": .5,
#     "Shaking head": .5,
#     "Showing the middle finger": .3,
#     "Swiping down": .65,
#     "Swiping down (with two hands)": .5,
#     "Swiping left": .5,
#     "Swiping right": .5,
#     "Swiping up": .5,
#     "Swiping up (with two hands)": .5,
#     "Thumb down": .5,
#     "Thumb up": .5,
#     "Waving": .5,
#     "Zooming in": .5,
#     "Zooming out": .5
# }

################
#   Baseline   #
################

LAB2INT_reactive = {
    "counting - clockwise_rotation=end": 1,
    "counting - clockwise_rotation=start": 2,
    "counting - counter-clockwise_rotation=end": 3,
    "counting - counter-clockwise_rotation=start": 4,
    "counting - draw_clockwise_rotation=end": 5,
    "counting - draw_clockwise_rotation=start": 6,
    "counting - draw_counter-clockwise_rotation=end": 7,
    "counting - draw_counter-clockwise_rotation=start": 8,
    "counting - invalid_gesture=end": 9,
    "counting - invalid_gesture=start": 10,
    "counting - swipe_down=end": 11,
    "counting - swipe_down=start": 12,
    "counting - swipe_left_with_left_hand=end": 13,
    "counting - swipe_left_with_left_hand=start": 14,
    "counting - swipe_left_with_right_hand=end": 15,
    "counting - swipe_left_with_right_hand=start": 16,
    "counting - swipe_right_with_left_hand=end": 17,
    "counting - swipe_right_with_left_hand=start": 18,
    "counting - swipe_right_with_right_hand=end": 19,
    "counting - swipe_right_with_right_hand=start": 20,
    "counting - swipe_up=end": 21,
    "counting - swipe_up=start": 22,
    "counting - thumb_down=end": 23,
    "counting - thumb_down=start": 24,
    "counting - thumb_up=end": 25,
    "counting - thumb_up=start": 26,
    "counting - zoom_in_with_full_hand=end": 27,
    "counting - zoom_in_with_full_hand=start": 28,
    "counting - zoom_out_with_full_hand=end": 29,
    "counting - zoom_out_with_full_hand=start": 30,
    "counting - background": 0
}

INT2LAB_reactive = {value: key for key, value in LAB2INT_reactive.items()}

LAB_THRESHOLDS_reactive = {
    "counting - clockwise_rotation=end": 0.5,
    "counting - clockwise_rotation=start": 0.1,
    "counting - counter-clockwise_rotation=end": 0.5,
    "counting - counter-clockwise_rotation=start": 0.1,
    "counting - draw_clockwise_rotation=end": 0.4,
    "counting - draw_clockwise_rotation=start": 0.1,
    "counting - draw_counter-clockwise_rotation=end": 0.3,
    "counting - draw_counter-clockwise_rotation=start": 0.1,
    "counting - invalid_gesture=end": 0.5,
    "counting - invalid_gesture=start": 0.1,
    "counting - swipe_down=end": 0.2,
    "counting - swipe_down=start": 0.1,
    "counting - swipe_left_with_left_hand=end": 0.2,
    "counting - swipe_left_with_left_hand=start": 0.1,
    "counting - swipe_left_with_right_hand=end": 0.2,
    "counting - swipe_left_with_right_hand=start": 0.1,
    "counting - swipe_right_with_left_hand=end": 0.2,
    "counting - swipe_right_with_left_hand=start": 0.1,
    "counting - swipe_right_with_right_hand=end": 0.2,
    "counting - swipe_right_with_right_hand=start": 0.1,
    "counting - swipe_up=end": 0.35,
    "counting - swipe_up=start": 0.1,
    "counting - thumb_down=end": 0.4,
    "counting - thumb_down=start": 0.1,
    "counting - thumb_up=end": 0.2,
    "counting - thumb_up=start": 0.1,
    "counting - zoom_in_with_full_hand=end": 0.3,
    "counting - zoom_in_with_full_hand=start": 0.1,
    "counting - zoom_out_with_full_hand=end": 0.25,
    "counting - zoom_out_with_full_hand=start": 0.1,
    "counting - background": 0.9
}

##################
#   Fine-tuned   #
##################

LAB2INT_reactive9 = {
    "counting - clockwise_rotation=end": 1,
    "counting - clockwise_rotation=start": 2,
    "counting - counter-clockwise_rotation=end": 3,
    "counting - counter-clockwise_rotation=start": 4,
    "counting - draw_clockwise_rotation=end": 5,
    "counting - draw_clockwise_rotation=start": 6,
    "counting - draw_counter-clockwise_rotation=end": 7,
    "counting - draw_counter-clockwise_rotation=start": 8,
    "counting - invalid_gesture=end": 9,
    "counting - invalid_gesture=start": 10,
    "counting - swipe_down=end": 11,
    "counting - swipe_down=start": 12,
    "counting - swipe_left_=end": 13,
    "counting - swipe_left_=start": 14,
    "counting - swipe_right_=end": 15,
    "counting - swipe_right_=start": 16,
    "counting - swipe_up=end": 17,
    "counting - swipe_up=start": 18,
    "counting - thumb_down=end": 19,
    "counting - thumb_down=start": 20,
    "counting - thumb_up=end": 21,
    "counting - thumb_up=start": 22,
    "counting - zoom_in_with_full_hand=end": 23,
    "counting - zoom_in_with_full_hand=start": 24,
    "counting - zoom_out_with_full_hand=end": 25,
    "counting - zoom_out_with_full_hand=start": 26,
    "counting - background": 0
}

INT2LAB_reactive9 = {value: key for key, value in LAB2INT_reactive9.items()}

LAB_THRESHOLDS_reactive9 = {
    "counting - clockwise_rotation=end": 0.5,                   # Doesn't activate
    "counting - clockwise_rotation=start": 0.2,
    "counting - counter-clockwise_rotation=end": 0.5,           # Doesn't activate
    "counting - counter-clockwise_rotation=start": 0.2,
    "counting - draw_clockwise_rotation=end": 0.1,              # Very low activation
    "counting - draw_clockwise_rotation=start": 0.1,
    "counting - draw_counter-clockwise_rotation=end": 0.2,      # Low activation
    "counting - draw_counter-clockwise_rotation=start": 0.1,
    "counting - invalid_gesture=end": 0.5,                      # Unknown gesture? Different from background?
    "counting - invalid_gesture=start": 0.2,
    "counting - swipe_down=end": 0.3,
    "counting - swipe_down=start": 0.2,
    "counting - swipe_left_=end": 0.4,
    "counting - swipe_left_=start": 0.2,
    "counting - swipe_right_=end": 0.4,
    "counting - swipe_right_=start": 0.2,
    "counting - swipe_up=end": 0.5,
    "counting - swipe_up=start": 0.2,
    "counting - thumb_down=end": 0.4,
    "counting - thumb_down=start": 0.2,
    "counting - thumb_up=end": 0.4,
    "counting - thumb_up=start": 0.2,
    "counting - zoom_in_with_full_hand=end": 0.4,
    "counting - zoom_in_with_full_hand=start": 0.2,
    "counting - zoom_out_with_full_hand=end": 0.4,
    "counting - zoom_out_with_full_hand=start": 0.2,
    "counting - background": 0.8
}
