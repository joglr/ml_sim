import pygame
from pygame.locals import QUIT
from shapely.geometry import LineString, Polygon, Point

class Environment:
    def __init__(self, width, height) -> None:
        self.width = width
        self.height = height
        self.walls = []
        self.create_floorplan()
    
    def get_dimensions(self):
        return (self.width,self.height)
    def create_floorplan(self):
        # Create LineString objects for the walls
        left_wall = LineString([(0, 0), (0, self.height)])
        bottom_wall = LineString([(0, self.height), (self.width, self.height)])
        right_wall = LineString([(self.width, self.height), (self.width, 0)])
        top_wall = LineString([(self.width, 0), (0, 0)])


        #kitchen
        kitchen_wall1 = LineString([(300, 0), (300, 150)])
        kitchen_wall2 = LineString([(0, 500), (300, 500)])
        kitchen_wall3 = LineString([(300, 400), (300, 500)])
        #office
        office_wall1 = LineString([(300, 700), (300, 800)])
        #closet
        closet_wall1 = LineString([(400, 700), (400, 800)])
        #living room
        living_room_right_wall1 = LineString([(800, 0), (800, 300)])
        living_room_right_wall2 = LineString([(800, 400), (800, 800)])
        #top right room
        top_right_wall1 = LineString([(800, 280), (self.width-100, 280)])
        #walk in closet
        walk_in_wall1 = LineString([(800, 680), (950, 680)])
        walk_in_wall2 = LineString([(1050, 680), (self.width, 680)])
        # Create interior walls
        #vertical_wall = LineString([(room_self.width, 0), (room_self.width, self.height)])
        #horizontal_wall = LineString([(0, 300), (self.width, 300)])

        #furniture
        sofa1 = LineString([(500,750),(600,750)])
        sofa2 = LineString([(600,750),(600,500)])
        sofa3 = LineString([(600,500),(500,500)])
        sofa4 = LineString([(500,500),(500,750)])

        table1 = LineString([(400,0),(400,100)])
        table2 = LineString([(400,100),(600,100)])
        table3 = LineString([(600,100),(600,0)])

        bed1 = LineString([(self.width,400),(self.width-250,400)])
        bed2 = LineString([(self.width-250,400),(self.width-250,600)])
        bed3 = LineString([(self.width-250,600),(self.width,600)])

        shower1 = LineString([(0,710),(90,710)])
        shower2 = LineString([(90,710),(90,self.height)])

        kitchen_table1 = LineString([(200,0),(200,100)])
        kitchen_table2 = LineString([(200,100),(100,100)])
        kitchen_table3 = LineString([(100,100),(100,300)])
        kitchen_table4 = LineString([(100,300),(0,300)])


        self.walls = [left_wall,bottom_wall,right_wall,top_wall,kitchen_wall1,\
            kitchen_wall2,kitchen_wall3,office_wall1,closet_wall1,\
                living_room_right_wall1,living_room_right_wall2,top_right_wall1,\
                    walk_in_wall1,walk_in_wall2,sofa1,sofa2,sofa3,sofa4,\
                        table1,table2,table3,bed1,bed2,bed3,shower1,shower2,\
                            kitchen_table1,kitchen_table2,kitchen_table3,kitchen_table4]
    def get_environment(self):
        return self.walls
    
    def checkColision(self, pos):
        for line in self.walls:
            if line.distance(Point(pos.x, pos.y)) <= 5:
                return True
    
    def draw(self,screen):
        # Draw the walls
        for wall in self.walls:
            pygame.draw.line(screen,(255,0,0),(int(wall.xy[0][0]),int(wall.xy[1][0])),(int(wall.xy[0][1]),int(wall.xy[1][1])),4)# (screen, (255, 0, 0), False, wall.xy, 4)



