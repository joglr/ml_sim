import math
import pygame
import numpy as np
import matplotlib.pyplot as plt
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

# Initialize the mean fitness plot
plt.ion()  # Enable interactive mode for live-updating
fig, ax = plt.subplots()
ax.set_title("Mean Fitness Over Time")
ax.set_xlabel("Episode")
ax.set_ylabel("Mean Fitness")
line, = ax.plot([], [], 'b-')  # Placeholder for mean fitness history
episodes, mean_fitness_values = [], []  # Lists to store data for plotting

def fitness_function(robot: DifferentialDriveRobot) -> float:
    rotations = robot.theta // (2 * math.pi)
    rotation_penalty = -20 * rotations

    # Check if the robot has collided with the environment
    if env.check_collision(robot):
        return rotation_penalty - 50
    return rotation_penalty + 10

if __name__ == "__main__":
    running = True
    episode_count = 0  # Track episode count separately
    while running:
        current_time = pygame.time.get_ticks()

        if (current_time - episode_start_time) >= EPISODE_DURATION:
            models = [mod for _, mod in robots_with_models]
            fitness_scores = [robot_scores[model] if model in robot_scores else 0 for robot, model in robots_with_models]
            mean_fitness = int(np.mean(fitness_scores))
            mean_fitness_history[episode_count] = mean_fitness  # Record mean fitness

            # Update live plot data
            episodes.append(episode_count)
            mean_fitness_values.append(mean_fitness)
            line.set_xdata(episodes)
            line.set_ydata(mean_fitness_values)
            ax.relim()
            ax.autoscale_view()  # Rescale axes based on data
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
            robots_weighted_selection = np.random.choice(models, size=rest_robots, p=rank_weights)

            new_models = top_n_models
            i = 0
            while i < rest_robots - 1:
                j = i + 1
                robot1_model = robots_weighted_selection[i]
                robot2_model = robots_weighted_selection[j]

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
