from shapely.geometry import Point, LineString
import math
import pygame
import numpy as np
from constants import BEAM_COUNT, BEAM_LENGTH

class LidarSensor:
    def __init__(self):
        self.num_beams = BEAM_COUNT
        self.max_distance_cm = BEAM_LENGTH

    def generate_scans(self, robot_pose, obstacles):
        x = robot_pose.x
        y = robot_pose.y
        theta = robot_pose.theta

        # Calculate the starting angle based on the robot's heading
        starting_angle = theta

        # Initialize a list to store distances for each beam
        distances = []
        intersect_points = []

        for angle in np.linspace(0, 2 * math.pi, self.num_beams, False):
            # Correct angle considering the starting angle and Pygame coordinate system
            corrected_angle = starting_angle + angle
            x2, y2 = (x + self.max_distance_cm * math.cos(corrected_angle), y + self.max_distance_cm * math.sin(corrected_angle))

            # Calculate the end point of the beam
            end_point = Point(x2, y2)

            # Create a LineString representing the Lidar beam
            lidar_beam = LineString([(x, y), end_point])

            # Check for intersections with obstacles
            intersection = self._check_intersections(lidar_beam, obstacles)

            # Calculate distance based on intersection or max distance
            if intersection:
                distance = Point(x, y).distance(intersection)
                intersect_points.append(intersection)
            else:
                distance = self.max_distance_cm
                intersect_points.append(end_point)
            distances.append(distance)

        return distances, intersect_points


    def _calculate_end_point(self, robot_pose, angle):
        """
        Calculate the end point of the Lidar beam based on the robot's pose and angle.

        Parameters:
            - robot_pose (Point): The current pose (location) of the robot with x, y, and theta.
            - angle (float): The angle of the Lidar beam in degrees.

        Returns:
            - Point: The end point of the Lidar beam.
        """
        x = robot_pose.x
        y = robot_pose.y
        theta = robot_pose.theta + angle  # Add angle to initial theta

        # Ensure the angle is within the valid range (0 to 360 degrees)
        theta %= 360

        # Convert angle to radians
        angle_rad = math.radians(theta)

        # Calculate the end point coordinates
        x_end = x + self.max_distance_cm * math.cos(angle_rad)
        y_end = y + self.max_distance_cm * math.sin(angle_rad)

        return Point(x_end, y_end)

    def _check_intersections(self, lidar_beam, obstacles):
        """
        Check for intersections between the Lidar beam and obstacles.

        Parameters:
            - lidar_beam (LineString): LineString representing the Lidar beam.
            - obstacles (list of LineString): List of LineString objects representing obstacles.

        Returns:
            - Point or None: The closest intersection point if there is one, otherwise None.
        """
        intersection_points = [lidar_beam.intersection(obstacle) for obstacle in obstacles]
        # Filter valid points and ensure they are of type Point
        valid_intersections = [point for point in intersection_points if not point.is_empty and isinstance(point, Point)]
        if valid_intersections:
            closest_intersection = min(valid_intersections, key=lambda point: lidar_beam.project(point))
            return closest_intersection
        else:
            return None


    def draw(self, robot_pose, intersect_points, screen):
        x = robot_pose.x
        y = robot_pose.y
        theta = robot_pose.theta
         # Draw the walls
        num_rays = BEAM_COUNT  # Number of desired rays
        step_size = len(intersect_points) // num_rays  # Calculate the step size

        for i in range(0, len(intersect_points), step_size):
            endpoint = intersect_points[i]
            pygame.draw.line(screen,(255,0,0),(x,y),(endpoint.x,endpoint.y),1)
        #draw front lidar beam in green
        endpoint = intersect_points[0]
        pygame.draw.line(screen,(0,255,0),(x,y),(endpoint.x,endpoint.y),2)


