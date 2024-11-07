import pygame
import random, math
from shapely import LineString, Point
from lidar import LidarSensor
from pygame.locals import QUIT, KEYDOWN
from environment import Environment
from robot import DifferentialDriveRobot,RobotPose
from localisation import ParticleFilterLocalization
from landmarks import LandmarkHandler
from ending import BoomAnimation
from constants import DRAW_LIDAR,DRAW_ROBOT, HEIGHT, USE_VISUALIZATION, WIDTH

# Initialize Pygame
pygame.init()

# Set up environment
env = Environment(WIDTH, HEIGHT)

#initialize lidar and robot
robot = DifferentialDriveRobot(WIDTH/2, HEIGHT/2, 0,)

# Create a Lidar sensor with 60 beams and a max distance of 500 units
lidar = LidarSensor()

#create landmarkhandler
landmark_handler = LandmarkHandler()

# Create ParticleFilterLocalization instance
particle_filter = ParticleFilterLocalization(motion_model_stddev=3, environment=env, landmarks = landmark_handler)

#for potential visualization

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Robotics Simulation")

#timestep counter in milliseconds
last_time = pygame.time.get_ticks()

#collision notifier, that show you if you run into a wall
boom_animation = BoomAnimation(WIDTH // 2, HEIGHT // 2)
showEnd = True

if __name__ == "__main__":
    # Game loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

        #calculate timestep
        time_step = (pygame.time.get_ticks() - last_time)/1000
        last_time = pygame.time.get_ticks()

        #this is the odometry where we use the wheel size and speed to calculate
        #where we approximately end up.
        robot_pose = robot.predict(time_step)

         # Generate Lidar scans - for these exercises, you wil be given these.
        lidar_scans, _intersect_points = lidar.generate_scans(robot_pose, env.get_environment())

        #This is what you will use in landmark detection
        landmark_sightings = landmark_handler.find_landmarks(_intersect_points, env.get_environment(), robot_pose)


        #this is where the particle filtering localisation happens
        particle_filter.update(time_step, landmark_sightings, robot.getMotorspeeds(), _intersect_points)

        # this is where we get the estimated position from the particles.
        particle_estimated_pose = particle_filter.get_estimate()

        #mix the update, sense data into one position
        mixed_pose = RobotPose((0.1 * robot_pose.x + 0.9 * particle_estimated_pose.x), \
                               (0.1 * robot_pose.y + 0.9 * particle_estimated_pose.y), \
                               (robot_pose.theta))

        #EXERCISE 6.1: make the robot move and navigate the environment based on our current sensor information and our current map.
        robot.explore_environment(lidar_scans)

        if USE_VISUALIZATION:
            screen.fill((0, 0, 0))
            #env.draw(screen) #draw walls

            if DRAW_LIDAR:
                lidar.draw(robot_pose, _intersect_points, screen)

            particle_filter.draw(screen)

            for landmark in landmark_handler.landmarks.values():
                # print(landmark)
                wall_segment = landmark['wall_segment']
                start_point = wall_segment.coords[0]
                end_point = wall_segment.coords[1]

                # Draw the wall segment representing the landmark
                pygame.draw.line(screen, (0, 255, 100), start_point, end_point, 5)


                # Optionally, draw the landmark ID
                mid_point = ((start_point[0] + end_point[0]) / 2, (start_point[1] + end_point[1]) / 2)
                font = pygame.font.SysFont(None, 24)
                text_surface = font.render(f"{landmark['id']}", True, (255, 255, 255))
                screen.blit(text_surface, mid_point)
                # pygame.draw.circle(screen, (0, 255, 0), (int(landmark_sighting['point_of_intercept'][0]), int(landmark_sighting['point_of_intercept'][1])), 5)

            for line in landmark_handler.segments:
                pygame.draw.line(screen, color=(255,0,255), start_pos=line.coords[0], end_pos=line.coords[-1], width=3)

            if DRAW_ROBOT:
                robot.draw(screen)
               # Update the display
            collided = env.checkColision(robot_pose)
            if collided:
                # Draw the animation
                boom_animation.draw(screen)
                ended = boom_animation.update()
                if ended:
                    pygame.quit()

            pygame.display.flip()
            pygame.display.update()

    # Quit Pygame
    pygame.quit()
