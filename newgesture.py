"""
Acquire a video for creating a gesture. Video frames will be saved.
"""

import sys
import gesturerecog as GR

gest=GR.CreateGesture(sys.argv[1])

gest.getframes()

