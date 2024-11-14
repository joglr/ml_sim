import math
import pygame
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress  # For trend line calculations
from lidar import LidarSensor
from environment import Environment
from model import GenerativeModel
from robot import DifferentialDriveRobot
from ending import BoomAnimation
from constants import BEAM_LENGTH, DRAW_LIDAR, DRAW_ROBOT, EPISODE_DURATION, HEIGHT, HIDDEN_LAYER_SIZE, MAX_WHEEL_SPEED, BEAM_COUNT, POPULATION_SIZE, TOP_N, USE_VISUALIZATION, WIDTH
import random

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))

pygame.display.set_caption("Robotics Simulation")

env = Environment(WIDTH, HEIGHT)

# Initialize population of robots
def create_robot_with_model(model: GenerativeModel = None):
    offsetX = 250
    offsetY = 100
    return (DifferentialDriveRobot(WIDTH / 2 + offsetX, HEIGHT / 2 + offsetY, random.random() * 360),
            model if model is not None else GenerativeModel(hidden_size=HIDDEN_LAYER_SIZE))

robots_with_models = [create_robot_with_model() for _ in range(POPULATION_SIZE)]
lidar = LidarSensor()
boom_animation = BoomAnimation(WIDTH // 2, HEIGHT // 2)
showEnd = True
last_time = pygame.time.get_ticks()
episode_start_time = pygame.time.get_ticks()  # Track the start time of each episode

robot_scores = {}
mean_fitness_history = {}  # Stores mean fitness over time for visualization

# Initialize the mean, min, max fitness plot
plt.ion()  # Enable interactive mode for live-updating
fig, ax = plt.subplots()
ax.set_title("Fitness Over Time")
ax.set_xlabel("Episode")
ax.set_ylabel("Fitness")
mean_line, = ax.plot([], [], 'b-', label="Mean Fitness")  # Mean fitness line
min_line, = ax.plot([], [], 'g-', label="Min Fitness")    # Min fitness line
max_line, = ax.plot([], [], 'r-', label="Max Fitness")    # Max fitness line
trend_line, = ax.plot([], [], 'k--', label="5-episode Trend")  # Trend line for the last 5 episodes

episodes, mean_fitness_values, min_fitness_values, max_fitness_values = [], [], [], []  # Lists to store data for plotting
ax.legend()  # Show legend for lines

def fitness_function(robot: DifferentialDriveRobot) -> float:
    rotations = robot.theta // (2 * math.pi)
    rotation_penalty = -20 * rotations

    # Check if the robot has collided with the environment
    if env.check_collision(robot):
        return rotation_penalty - 50
    return rotation_penalty + 10

def update_trend_line():
    # Calculate trend line for the last 5 episodes if there are enough data points
    if len(mean_fitness_values) >= 5:
        recent_episodes = episodes[-5:]
        recent_means = mean_fitness_values[-5:]
        slope, intercept, _, _, _ = linregress(recent_episodes, recent_means)
        trend_x = np.array([recent_episodes[0], recent_episodes[-1]])
        trend_y = intercept + slope * trend_x
        trend_line.set_xdata(trend_x)
        trend_line.set_ydata(trend_y)
    else:
        trend_line.set_xdata([])
        trend_line.set_ydata([])

if __name__ == "__main__":
    running = True
    episode_count = 0  # Track episode count separately
    while running:
        current_time = pygame.time.get_ticks()

        if (current_time - episode_start_time) >= EPISODE_DURATION:
            models = [mod for _, mod in robots_with_models]
            fitness_scores = [robot_scores[model] if model in robot_scores else 0 for robot, model in robots_with_models]

            # Calculate mean, min, max fitness values
            mean_fitness = np.mean(fitness_scores)
            min_fitness = np.min(fitness_scores)
            max_fitness = np.max(fitness_scores)

            # Update live plot data
            episodes.append(episode_count)
            mean_fitness_values.append(mean_fitness)
            min_fitness_values.append(min_fitness)
            max_fitness_values.append(max_fitness)

            mean_line.set_xdata(episodes)
            mean_line.set_ydata(mean_fitness_values)
            min_line.set_xdata(episodes)
            min_line.set_ydata(min_fitness_values)
            max_line.set_xdata(episodes)
            max_line.set_ydata(max_fitness_values)

            # Update trend line for the last 5 episodes
            update_trend_line()

            # Rescale axes based on data
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.01)  # Pause to update the figure

            episode_count += 1
            robot_scores = {}
            sorted_indices = np.argsort(fitness_scores)

            rank_weights = np.zeros(POPULATION_SIZE)
            rank = 2
            for i in sorted_indices:
                rank_weights[i] = 1 / rank
                rank += 1
            rank_weights /= np.sum(rank_weights)

            top_n_indices = sorted_indices[-TOP_N:]
            top_n_models = [models[i] for i in top_n_indices]
            rest_robots = POPULATION_SIZE - TOP_N
            # robots_weighted_selection = np.random.choice(models, size=rest_robots, p=rank_weights)

            new_models = top_n_models

            for model in new_models:
                model.mutate()

            i = 0
            while i < rest_robots:
                two_robots = np.random.choice(models, size=2, p=rank_weights)
                robot1_model = two_robots[0]
                robot2_model = two_robots[1]

                new_robot_1 = robot1_model.crossover(robot2_model)
                new_robot_2 = robot2_model.crossover(robot1_model)

                new_robot_1.mutate()
                new_robot_2.mutate()

                new_models.append(new_robot_1)
                new_models.append(new_robot_2)

                i += 2

            robots_with_models = [create_robot_with_model(mod) for mod in new_models]
            episode_start_time = current_time

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        time_step = (current_time - last_time) / 1000
        last_time = current_time

        print("FPS: ", 1/time_step)

        if USE_VISUALIZATION:
            screen.fill((0, 0, 0))
            env.draw(screen)

        for robot, model in robots_with_models:
            total_reward = 0
            robot_pose = robot.predict(time_step)
            lidar_scans, _intersect_points = lidar.generate_scans(robot_pose, env.get_environment())
            lidar_scans = [random.random() for _ in range(BEAM_COUNT)]

            left_wheel, right_wheel = np.random.rand(), np.random.rand()
            robot.set_motor_speeds(left_wheel * MAX_WHEEL_SPEED, right_wheel * MAX_WHEEL_SPEED)

            reward = fitness_function(robot)
            robot_scores[model] = robot_scores[model] if model in robot_scores else 0
            robot_scores[model] += reward

            if USE_VISUALIZATION:
                lidar.draw(robot_pose, _intersect_points, screen)
                robot.draw(screen)

        pygame.display.flip()

    pygame.quit()
    plt.ioff()  # Turn off interactive mode
    plt.show()  # Keep the plot open after Pygame window closes
