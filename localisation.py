import statistics
import numpy as np
from shapely.ops import nearest_points
from shapely.geometry import Point, LineString
import random
import math
import pygame
from math_utils import clamp, orthogonal_projection
from robot import RobotPose
from copy import deepcopy
import cv2
from constants import BEAM_LENGTH, DEOCCUPANCY_DELTA, GRID_SIZE, HEIGHT, IMG_PATH, MAX_WHEEL_SPEED, NUM_PARTICLES, OCCUPANCY_DELTA, V_MAX, WIDTH, DRAW_PARTICLES, AXLE_LENGTH, WHEEL_RADIUS

class ParticleFilterLocalization:
    def __init__(self, motion_model_stddev, environment, landmarks):
        self.motion_model_stddev = motion_model_stddev
        self.environment = environment
        self.particles = []
        self.landmarks = landmarks.landmarks
        self.num_particles = NUM_PARTICLES
        self.grid_size = GRID_SIZE

        self.weights = np.ones(self.num_particles) / self.num_particles

        # Initialize particles
        self._initialize_particles()

        #this is just for potential visualization, REMEMBER VISUALIZATION IS NOT THE SIMULATION
        self.image = pygame.image.load(IMG_PATH)
        self.rect = self.image.get_rect()

        self.map = self._initialize_map()

        self.steps = 11

    def _initialize_map(self):
        occupancy_map = np.zeros((self.environment.width // self.grid_size + 1, self.environment.height // self.grid_size + 1))
                # Add noise to all cells of the map
        for i in range(self.environment.width // self.grid_size + 1):
            for j in range(self.environment.height // self.grid_size + 1):
                occupancy_map[i, j] = random.gauss(0.5, 0.03)
                occupancy_map[i, j] = clamp(occupancy_map[i, j])

        return occupancy_map

    def _initialize_particles(self):
        particles = []

        for i in range(self.num_particles):
            x = random.randint(0, WIDTH)
            y = random.randint(0, HEIGHT)
            theta = random.uniform(0, 2 * math.pi)
            particles.append(Particle(x, y, theta))

        self.particles = particles

    def _add_noise_to_pose(self, pose):
        """
        Add a small amount of noise to the particle's pose.

        Parameters:
            - pose (Point): The particle's current pose.

        Returns:
            - Point: Updated particle pose with noise.
        """
        noise_x = random.gauss(0, self.motion_model_stddev)
        noise_y = random.gauss(0, self.motion_model_stddev)
        noise_theta = random.gauss(0, self.motion_model_stddev)

        # Ensure particles stay within bounds
        particle_x = max(0, min(pose.x + noise_x, WIDTH))
        particle_y = max(0, min(pose.y + noise_y, HEIGHT))
        particle_theta = (pose.theta + noise_theta) % (2 * math.pi)

        return Particle(particle_x, particle_y, particle_theta)

    def _measurement_update(self, landmark_sightings):
        self.weights = np.zeros(len(self.particles))
        sigma_d = 1.0  # Adjust based on your sensor's noise characteristics
        sigma_theta = 0.1

        for i, particle in enumerate(self.particles):
            total_likelihood = 1.0
            particle_point = Point(particle.x, particle.y)

            for sighting in landmark_sightings:
                landmark_id = sighting['landmark_id']
                orthogonal_distance = sighting['orthogonal_distance']
                bearing_angle = sighting['bearing_angle']

                landmark_pose = self.landmarks[landmark_id]
                wall_segment = landmark_pose['wall_segment']

                nearest_point_on_wall = wall_segment.interpolate(wall_segment.project(particle_point))
                p_distance = particle_point.distance(nearest_point_on_wall)
                _, bearing = orthogonal_projection(particle, wall_segment)
                bearing = bearing % (2 * math.pi)

                dist_error = p_distance - orthogonal_distance
                theta_error = (bearing - bearing_angle + math.pi) % (2 * math.pi) - math.pi

                likelihood_d = np.exp(-0.5 * (dist_error / sigma_d) ** 2)
                likelihood_theta = np.exp(-0.5 * (theta_error / sigma_theta) ** 2)

                total_likelihood *= likelihood_d * likelihood_theta

            self.weights[i] = total_likelihood

        self.weights += 1e-300  # Prevent division by zero
        self.weights /= np.sum(self.weights)

    def resample_particles(self):
        #number of random particles so it doesn't get stuck
        n = self.num_particles//20
        indices = np.random.choice(len(self.particles), self.num_particles-n, p=self.weights)
        new_particles = [deepcopy(self.particles[i]) for i in indices]

        for i in range(n):
            x = random.randint(0, WIDTH)
            y = random.randint(0, HEIGHT)
            theta = random.uniform(0, 2 * math.pi)
            new_particles.append(Particle(x, y, theta))

        self.particles = [self._add_noise_to_pose(particle) for particle in new_particles]

    def update(self, delta_time, landmark_sightings, current_motor_speeds, intersect_points):
        self.steps += 1

        # Update particles based on motion model
        self.particles = self._motion_model(self.particles, current_motor_speeds, delta_time)

        # Calculate weights based on measurement model
        self._measurement_update(landmark_sightings)

         #calculates the current estimated position
        self.calculate_estimated_position()

        weights_std_dev = statistics.stdev(self.weights)


        # Resample particles based on weights
        self.resample_particles()
        # Update the map if particles have converged
        if weights_std_dev > 0.02:
            self.update_map(self.get_estimate(), intersect_points)
        else:
            print("Particles have not converged yet, standard deviation of weights: ", weights_std_dev)

        # Reset weights to uniform after resampling
        self.weights = np.ones(self.num_particles) / self.num_particles

        # Visualization step
        if self.steps % 5 == 0 or True:
            # Normalize the map values to 0-255
            map_copy = self.map.copy()
            robot_x = int(self.latest_pose.x // self.grid_size)
            robot_y = int(self.latest_pose.y // self.grid_size)
            map_copy[robot_x, robot_y] = 1

            map_display = cv2.normalize(map_copy, None, 0, 255, cv2.NORM_MINMAX)
            map_display = map_display.astype(np.uint8)

            # Apply a color map for better visualization (optional)
            # map_display_color = cv2.applyColorMap(map_display, cv2.COLORMAP_JET)
            map_display = cv2.resize(map_display, (HEIGHT, WIDTH), interpolation=cv2.INTER_NEAREST)

            # Flip and rotate the map for correct orientation
            img_flipped = cv2.flip(map_display, 0)
            rotated = cv2.rotate(img_flipped, cv2.ROTATE_90_CLOCKWISE)

            # Display the occupancy grid map
            cv2.namedWindow('image_window', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image_window', 800, 600)
            cv2.imshow('image_window', rotated)
            cv2.waitKey(1)  # This keeps the window open

    def get_estimate(self):
        return self.latest_pose

    def calculate_estimated_position(self):
        top_20_num_particles = int(0.2 * self.num_particles)

        # Sort particles by weight and select top 20%
        sorted_indices = np.argsort(self.weights)[::-1]

        # Calculate average of top 20% particles
        x = 0
        y = 0
        sin_sum = 0
        cos_sum = 0

        for i in sorted_indices[:top_20_num_particles]:
            particle_weight = self.weights[i]
            x += self.particles[i].x * particle_weight
            y += self.particles[i].y * particle_weight
            sin_sum += math.sin(self.particles[i].theta * particle_weight)
            cos_sum += math.cos(self.particles[i].theta * particle_weight)

        theta = math.atan2(sin_sum, cos_sum)

        self.latest_pose = RobotPose(x, y, theta)

    def _motion_model(self, particles, motor_speeds, delta_time):
        for i in range(len(self.particles)):
            particle = particles[i]
            # Assume maximum linear velocity at motor speed 500
            v_max = V_MAX  # pixels/second
            # Calculate the linear velocity of each wheel
            left_motor_speed, right_motor_speed = motor_speeds
            left_wheel_velocity = (left_motor_speed / MAX_WHEEL_SPEED) * v_max
            right_wheel_velocity = (right_motor_speed / MAX_WHEEL_SPEED) * v_max

            v_x = math.cos(particle.theta) * (WHEEL_RADIUS * (left_wheel_velocity + right_wheel_velocity) / 2)
            v_y = math.sin(particle.theta) * (WHEEL_RADIUS * (left_wheel_velocity + right_wheel_velocity) / 2)
            omega = (WHEEL_RADIUS * (left_wheel_velocity - right_wheel_velocity)) / AXLE_LENGTH

            particle.x += v_x * delta_time
            particle.y += v_y * delta_time
            particle.theta += omega * delta_time

            # Ensure the orientation is within the range [0, 2*pi)
            particle.theta = particle.theta % (2 * math.pi)

            particles[i] = particle
        return particles


    def update_map(self, average_particle, intersect_points):
        x0 = int(average_particle.x // self.grid_size)
        y0 = int(average_particle.y // self.grid_size)

        # Decrease occupancy at the robot's position (free space)
        if 0 <= x0 < self.map.shape[0] and 0 <= y0 < self.map.shape[1]:
            self.map[x0, y0] = clamp(self.map[x0, y0] - 1)

        for point in intersect_points:
            x1 = int(point.x // self.grid_size)
            y1 = int(point.y // self.grid_size)

            # Find cells along the line from (x0, y0) to (x1, y1)
            cells = self.get_line(x0, y0, x1, y1)

            # Decrease occupancy for cells along the beam (free space)
            for cell in cells[:-1]:  # Exclude the last cell (the obstacle)
                xi, yi = cell
                if 0 <= xi < self.map.shape[0] and 0 <= yi < self.map.shape[1]:
                    self.map[xi, yi] = clamp(self.map[xi, yi] - DEOCCUPANCY_DELTA)  # Decrease occupancy

            # Increase occupancy for the cell at the obstacle
            dist_to_obstacle = math.sqrt((average_particle.x - point.x) ** 2 + (average_particle.y - point.y) ** 2)

            if dist_to_obstacle > BEAM_LENGTH - 50:
                # Skip updating the map if the obstacle is too far away
                continue

            if 0 <= x1 < self.map.shape[0] and 0 <= y1 < self.map.shape[1]:
                self.map[x1, y1] = clamp(self.map[x1, y1] + OCCUPANCY_DELTA)  # Increase occupancy

    def get_line(self, x1, y1, x2, y2):
        #bresenham's line algorithm
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        cells = []
        while True:
            cells.append((x1, y1))
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

        return cells

    def draw(self, surface):
        if DRAW_PARTICLES:
            for i, particle in enumerate(self.particles):
                weight =  self.weights[i]
                pygame.draw.circle(surface, (0,100,100),(particle.x,particle.y),5,2)


class Particle:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
