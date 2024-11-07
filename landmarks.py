import pygame
from shapely import Point, LineString
from shapely.ops import nearest_points
import math
import sys

from math_utils import orthogonal_projection

class LandmarkHandler:
    def __init__(self) -> None:
        self.landmarks = {}
        self.landmark_id = 0
        self.segments = []

    def find_landmarks(self, intersect_points, walls, current_pose, threshold=5):
        landmark_sightings = []
        line_segments = []
        current_segment = []

        for i in range(1, len(intersect_points) - 1):
            p1 = intersect_points[i - 1]
            p2 = intersect_points[i]
            p3 = intersect_points[i + 1]

            if self.points_belong_to_straight_line(p1, p2, p3):
                if not current_segment:
                    current_segment.extend([p1, p2])
                current_segment.append(p3)
            else:
                if current_segment and len(current_segment) >= 3:
                    # Create LineString from the current segment
                    line_segment = LineString([(p.x, p.y) for p in current_segment])
                    line_segments.append(line_segment)
                current_segment = []


        if current_segment and len(current_segment) >= 2:
            line_segment = LineString([(p.x, p.y) for p in current_segment])
            line_segments.append(line_segment)

        # Match observed line segments to walls and create landmarks
        walls_matched = set()
        for line_segment in line_segments:
            min_distance = float('inf')
            matched_wall = None

            for wall in walls:
                distance = line_segment.distance(wall)
                if distance < threshold and distance < min_distance:
                    min_distance = distance
                    matched_wall = wall
                    self.segments.append(line_segment)

            if matched_wall and matched_wall not in walls_matched:
                walls_matched.add(matched_wall)
                landmark_id = self.create_landmark(matched_wall, threshold)
                landmark_sighting = self.create_landmark_sighting(landmark_id, current_pose)
                landmark_sightings.append(landmark_sighting)

        return landmark_sightings


    #creates and stores a new landmark if it has not already been spottet
    #in real life, this would also be where you would check if any line segments belonged to the same landmark/wall.
    def create_landmark(self, wall, threshold=5):
        for existing_landmark_key in self.landmarks:
            existing_landmark = self.landmarks[existing_landmark_key]
            if existing_landmark['wall_segment'].distance(wall) < threshold:
                if not self.are_lines_perpendicular(existing_landmark['wall_segment'], wall):
                    return existing_landmark['id']
        self.landmark_id += 1
        new_landmark = {
                'wall_segment':wall,
                'id':self.landmark_id
        }
        self.landmarks[self.landmark_id] = new_landmark
        return new_landmark['id']


    def are_lines_perpendicular(self, line1, line2, tolerance=0.1):
        # direction vectors of the lines
        x1, y1 = line1.coords[0]
        x2, y2 = line1.coords[-1]
        x3, y3 = line2.coords[0]
        x4, y4 = line2.coords[-1]

        # vector of line1
        dx1 = x2 - x1
        dy1 = y2 - y1
        #vector of line2
        dx2 = x4 - x3
        dy2 = y4 - y3

        # dot product
        dot_product = dx1 * dx2 + dy1 * dy2

        return abs(dot_product) < tolerance




    #creates a landmark sighting
    def create_landmark_sighting(self, landmark_id, current_pose):
        wall = self.landmarks[landmark_id]['wall_segment']
        semi_orthogonal_point, bearing = orthogonal_projection(current_pose, wall)
        current_pose_as_point = Point(current_pose.x, current_pose.y)
        distance = current_pose_as_point.distance(semi_orthogonal_point)
        landmark_sighting = {
                "landmark_id": landmark_id,
                "point_of_intercept": (semi_orthogonal_point.x, semi_orthogonal_point.y),
                "orthogonal_distance": distance,
                "bearing_angle": bearing
            }
        return landmark_sighting

    def points_belong_to_straight_line(self, p1, p2, p3):
        # Extract coordinates
        x1, y1 = p1.x, p1.y
        x2, y2 = p2.x, p2.y
        x3, y3 = p3.x, p3.y

        # Calculate the cross product of vectors (p1p2) and (p1p3)
        cross_product = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)

        # If cross product is zero, the points are collinear
        return cross_product == 0

