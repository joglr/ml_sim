"""Constants file to keep all together"""

# Max pixels per second
V_MAX = 10
MAX_WHEEL_SPEED = 500
USE_VISUALIZATION = True
DRAW_LIDAR = True
DRAW_PARTICLES = True
DRAW_ROBOT = True

WIDTH = 1200
HEIGHT = 800

BEAM_LENGTH = 120
BEAM_COUNT = 3

AXLE_LENGTH = 10
WHEEL_RADIUS = 2.2
IMG_PATH = 'thymio_small.png'
GRID_SIZE = 20
NUM_PARTICLES = 200
NOISE_FACTOR = 0.01

OCCUPANCY_DELTA = 0.05
DEOCCUPANCY_DELTA = 0.025


POPULATION_SIZE = 80  # Total number of robots
# TOP_N = POPULATION_SIZE // 2  # Number of top-performing robots to keep
TOP_N = 4  # Number of top-performing robots to keep
EPISODE_DURATION = 15_000  # Duration of each episode in seconds
HIDDEN_LAYER_SIZE = 10  # Size of the hi4dden layer in the generative model
