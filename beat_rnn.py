# -*- coding: utf-8 -*-
import collections
import datetime

import pygame
import numpy as np
import pathlib
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_datasets as tfds
from IPython import display
from matplotlib import pyplot as plt
from typing import Dict, List, Optional, Sequence, Tuple

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)




# Load the full GMD with MIDI only (no audio) as a tf.data.Dataset
dataset = tfds.load(
    name="groove/full-midionly",
    split=tfds.Split.TRAIN,
    try_gcs=True)

# Build your input pipeline
dataset = dataset.shuffle(1024).batch(32).prefetch(
    tf.data.experimental.AUTOTUNE)

for features in dataset.take(1):
  # Access the features you are interested in
  midi, genre = features["midi"], features["style"]["primary"]
  break;


def play_music(midi_filename):
  '''Stream music_file in a blocking manner'''
  clock = pygame.time.Clock()
  pygame.mixer.music.load(midi_filename)
  pygame.mixer.music.play()
  while pygame.mixer.music.get_busy():
    clock.tick(30) # check if playback has finished
    

# mixer config
freq = 44100  # audio CD quality
bitsize = -16   # unsigned 16 bit
channels = 2  # 1 is mono, 2 is stereo
buffer = 1024   # number of samples
pygame.mixer.init(freq, bitsize, channels, buffer)

# optional volume 0 to 1.0
pygame.mixer.music.set_volume(0.8)

# listen for interruptions
try:
  # use the midi file you just saved
  play_music(midi)
except KeyboardInterrupt:
  # if user hits Ctrl/C then exit
  # (works only in console mode)
  pygame.mixer.music.fadeout(1000)
  pygame.mixer.music.stop()
  raise SystemExit
  