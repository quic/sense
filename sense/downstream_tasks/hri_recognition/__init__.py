LAB2INT = {
  "Come forward": 0,
  "Do Nothing": 1,
  "Doing something else": 2,
  "Handwave": 3,
  "Move backward": 4,
  "Move to the left": 5,
  "Move to the right": 6,
  "Pause": 7,
  "Pointing: Object": 8,
  "Repeat: Rotate arm clockwise": 9,
  "Resume": 10,
  "Start": 11,
  "Stop": 12,
  "Thumbs down": 13,
  "Thumbs up": 14,
  "Undo: Rotate arm anti-clockwise": 15,
  "Watch out": 16
}

INT2LAB = {value: key for key, value in LAB2INT.items()}

GESTURE_THRESHOLDS = {
    "Pause": 0.6,
    "Move to the left": 0.6,
    "Undo: Rotate arm anti-clockwise": 0.3, 
    "Thumbs down": 0.6,  
    "Watch out": 0.5, 
    "Thumbs up": 0.9, 
    "Resume": 0.7, 
    "Repeat: Rotate arm clockwise": 0.3, 
    "Come forward": 0.5, 
    "Move to the right": 0.6, 
    "Move backward": 0.8, 
    "Handwave": 0.4, 
    "Start": 0.5, 
    "Pointing: Object": 0.5, 
    "Stop": 0.95
}

