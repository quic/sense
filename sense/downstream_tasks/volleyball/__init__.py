LAB2INT_CLASSIFICATION = {
    "holding_ball": 0,
    "forearm_passing": 1,
    "pokey": 2,
    "dropping_ball": 3,
    "doing_nothing": 4,
    "bouncing_ball": 5,
    "one_arm_passing": 6,
    "leaving_screen": 7,
    "overhead_passing": 8
}

LAB2INT_COUNTING = {
    "counting_background": 0,
    "holding_ball_position_1": 1,
    "holding_ball_position_2": 2,
    "forearm_passing_position_1": 3,
    "forearm_passing_position_2": 4,
    "pokey_position_1": 5,
    "pokey_position_2": 6,
    "dropping_ball_position_1": 7,
    "dropping_ball_position_2": 8,
    "doing_nothing_position_1": 9,
    "doing_nothing_position_2": 10,
    "bouncing_ball_position_1": 11,
    "bouncing_ball_position_2": 12,
    "one_arm_passing_position_1": 13,
    "one_arm_passing_position_2": 14,
    "leaving_screen_position_1": 15,
    "leaving_screen_position_2": 16,
    "overhead_passing_position_1": 17,
    "overhead_passing_position_2": 18
}

INT2LAB_CLASSIFICATION = {value: key for key, value in LAB2INT_CLASSIFICATION.items()}
INT2LAB_COUNTING = {value: key for key, value in LAB2INT_COUNTING.items()}

CLASSIFICATION_THRESHOLDS = {key: 0.5 for key in LAB2INT_CLASSIFICATION}
