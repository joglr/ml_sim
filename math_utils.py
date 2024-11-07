from shapely import Point
from shapely.ops import nearest_points

import math

#NOTE: this doesn't really create the orthogonal projection if the point lies outside of the line segment
#but it is easy to understand and will be identical for landmarks found with other lidar rays that intersect with
#the same wall.
def orthogonal_projection(current_pose, linestring):
    point = Point(current_pose.x, current_pose.y)
    theta =  current_pose.theta

    # Find the nearest point on the LineString to the given point
    nearest_point_on_line = nearest_points(point, linestring)[1]

    # Calculate the bearing angle between the robot's current position and the orthogonal intercept point
    bearing = math.atan2(nearest_point_on_line.y - point.y, nearest_point_on_line.x - point.x) - theta

    return nearest_point_on_line, bearing

def clamp(n, minn=0, maxn=1):
    return max(min(maxn, n), minn)
