import pygame
import numpy as np
from lidar import LidarSensor
from environment import Environment
from model import GenerativeModel 
from robot import DifferentialDriveRobot
from ending import BoomAnimation
from constants import BEAM_LENGTH, DRAW_LIDAR, DRAW_ROBOT, HEIGHT, MAX_WHEEL_SPEED, NUM_BEAMS, USE_VISUALIZATION, WIDTH
import random 

POPULATION_SIZE = 20  # Total number of robots
TOP_N = POPULATION_SIZE // 2  # Number of top-performing robots to keep
EPISODE_DURATION = 4  # Duration of each episode in seconds
HIDDEN_LAYER_SIZE = 10  # Size of the hidden layer in the generative model

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Robotics Simulation")

env = Environment(WIDTH, HEIGHT)

# Initialize population of robots
robots_with_models = [(DifferentialDriveRobot(WIDTH / 2, HEIGHT / 2, random.random() * 360), GenerativeModel(hidden_size=HIDDEN_LAYER_SIZE)) for _ in range(POPULATION_SIZE)]
lidar = LidarSensor()
boom_animation = BoomAnimation(WIDTH // 2, HEIGHT // 2)
showEnd = True
last_time = pygame.time.get_ticks()
episode_start_time = pygame.time.get_ticks()  # Track the start time of each episode

def evaluate_robot(robot_with_model: tuple[DifferentialDriveRobot, GenerativeModel]) -> float:
    rob, mod = robot_with_model
    # rob.reset(WIDTH / 2, HEIGHT / 2, random.random() * 360)
    # mod.reset()
    total_reward = 0
    for _ in range(100):
        robot_pose = rob.predict(0.1)
        lidar_scans, _ = lidar.generate_scans(robot_pose, env.get_environment())
        left_wheel, right_wheel = mod.train(lidar_scans)
        left_wheel, right_wheel = left_wheel.item(), right_wheel.item()
        rob.set_motor_speeds(left_wheel * MAX_WHEEL_SPEED, right_wheel * MAX_WHEEL_SPEED)
        reward = reward_function(rob)
        total_reward += reward
        if reward < 0:
            break
    return total_reward

def reward_function(robot: DifferentialDriveRobot) -> float:
    # Check if the robot has collided with the environment
    if env.check_collision(robot):
        return 0

    # # Check if the robot has reached the goal
    # if env.check_goal(robot):
    return 100

if __name__ == "__main__":
    running = True
    while running:
        current_time = pygame.time.get_ticks()
        
        # Check if the episode duration has been reached
        if (current_time - episode_start_time) / 1000 >= EPISODE_DURATION:
            models = [mod for _, mod in robots_with_models]
            # Evaluate and update population
            fitness_scores = [evaluate_robot(robot) for robot in robots_with_models]
            
            # Calculate ranking for each robot based on fitness score
            sorted_indices = np.argsort(fitness_scores)

            rank_weights = np.zeros(POPULATION_SIZE)
            rank = 2
            for i in sorted_indices:
                rank_weights[i] = 1 / rank
                rank += 1

            # Normalize rank weights
            rank_weights /= np.sum(rank_weights)

            # Elitism: Select top_n robots
            top_n = 3
            top_n_indices = sorted_indices[-top_n:]
            top_n_models = [models[i] for i in top_n_indices]

            rest_robots = POPULATION_SIZE - top_n

            # Selection 
            print(len(robots_with_models), len(rank_weights))
            print(np.array(robots_with_models).shape, np.array(rank_weights).shape)
            # print(robots_with_models)
            # print(rank_weights)
            robots_weighted_selection = np.random.choice(models, size=rest_robots, p=rank_weights)
            
            # Crossover
            new_models = top_n_models

            i = 0
            while len(new_models) < rest_robots:
                j = i +1
                
                robot1_model = robots_weighted_selection[i][1]
                robot2_model = robots_weighted_selection[j][1]

                new_robot_1 = robot1_model.crossover(robot2_model)
                new_robot_2 = robot2_model.crossover(robot1_model)

                # Mutation
                new_robot_1.mutate()
                new_robot_2.mutate()

                new_models.append(new_robot_1)
                new_models.append(new_robot_2)

                i += 2

            # Set the robots to the new population
            robots_with_models = [(DifferentialDriveRobot(WIDTH / 2, HEIGHT / 2, random.random() * 360), mod) for mod in new_models]
            # Reset episode start time
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
            # print(model)
            robot_pose = robot.predict(time_step)
            lidar_scans, _intersect_points  = lidar.generate_scans(robot_pose, env.get_environment())
            left_wheel, right_wheel = model.predict(lidar_scans)
            left_wheel, right_wheel = left_wheel.item(), right_wheel.item()
            # left_wheel, right_wheel = np.random.rand(), np.random.rand()
            robot.set_motor_speeds(left_wheel * MAX_WHEEL_SPEED, right_wheel * MAX_WHEEL_SPEED)
            if USE_VISUALIZATION:
                lidar.draw(robot_pose, _intersect_points, screen)
                robot.draw(screen)


        pygame.display.flip()

    pygame.quit()
