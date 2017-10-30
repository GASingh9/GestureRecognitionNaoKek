"""
Generate a trained MLP model with the data stored.
"""

import gesturerecog as GR

train=GR.TrainNao()

print("{} gestures found".format(train.gest_count))
print(train.gest_identifier)
