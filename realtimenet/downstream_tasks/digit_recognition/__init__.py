LAB2INT = {
    "Doing nothing": 0,
    "Doing other things": 1,
    "0": 2,
    "1": 3,
    "2": 4,
    "3": 5,
    "4": 6,
    "5": 7,
    "6": 8,
    "7": 9,
    "8": 10,
    "9": 11,
    "Drawing": 12,
}

INT2LAB = {value: key for key, value in LAB2INT.items()}

# v11_mix
LAB2THRESHOLD = {
    "0": 0.7,
    "1": 0.7,
    "2": 0.7,
    "3": 0.3,
    "4": 0.3,
    "5": 0.5,
    "6": 0.4,
    "7": 0.5,
    "8": 0.4,
    "9": 0.3,
}
