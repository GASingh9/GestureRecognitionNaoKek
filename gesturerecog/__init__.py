#!/usr/bin/env python
"""
Create and train gestures for gesture recognition on the Nao Robot.
"""
from .creategesture import CreateGesture
from .trainnao import TrainNao

print("gesturerecog package initialised")

__author__ = "Gurashish Singh Bhatia, Gwendolyn Foo, David Ingram and Junho Jung"
__version__ = "1.0"
__status__ = "Development"

__all__ = (
    'CreateGesture',
    'TrainNao'
)
