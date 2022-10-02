import math
from time import sleep
from turtle import back
import numpy as np
import pygame

class Car:
    def __init__(self, points=(0, 0), dir=0):
        self.point = points
        self.dir = dir

    def drive(self, left, right):
        self.point = (self.point[0] + math.cos(self.dir), self.point[1] + math.sin(self.dir))
        self.dir += left - right
    
    def draw(self, surface, color):
        pygame.draw.circle(surface, color, center(self.point), 3)
        pygame.draw.line(surface, color, center(self.point), center(extend(self.point, self.dir, 1)))

class Road:
    def __init__(self, outerbound = None, innerbound = None):
        self.outerbound = outerbound # np array of points
        self.innerbound = innerbound # np array of points
        if (self.outerbound and self.innerbound):
            self.calcStart()
    
    def calcStart(self):
        self.start = ((self.innerbound[0][0] + self.outerbound[0][0]) / 2, (self.innerbound[0][1] + self.outerbound[0][1]) / 2)
        self.dir = math.atan2(self.innerbound[1][1] - self.innerbound[0][1], self.innerbound[1][0] - self.innerbound[0][0])
        return (self.start, self.dir)
    
    def generate_rand(self, points):
        # road = []
        # loc = (0, 0)
        # dir = 0
        # for i in range(points):
        #     road.append(loc)

        #     dir += np.random.normal()
        #     loc = (loc[0] + math.cos(dir), loc[1] + math.sin(dir))
        
        # width = 1

        self.outerbound = []
        self.innerbound = []

        # last_point = road[-1]
        # for point in road:
        #     midpoint = ((point[0] + last_point[0]) / 2, (point[1] + last_point[1]) / 2)
            
        #     slope = 1

        #     self.outerbound.append((midpoint[0], midpoint[1]))
        #     self.innerbound.append((midpoint[0], midpoint[1]))

        _2pi = 2 * math.pi
        for i in range(points):
            theta = _2pi * i / points
            val = 3 * math.sin(theta * 4) + 13

            self.outerbound.append((math.sin(theta) * val, math.cos(theta) * val))
            self.innerbound.append((math.sin(theta) * (val + 4), math.cos(theta) * (val + 4)))

        # start is in the middle of first two points in outer/inner bound
        self.calcStart()
        return self

    def draw(self, surface, color):
        for bound in [self.innerbound, self.outerbound]:
            last_point = bound[-1]
            for point in bound:
                pygame.draw.line(surface, color, center(point), center(last_point))
                last_point = point
    
    def sensors(self, car):
        
        sens = [math.inf, math.inf, math.inf]

        sens_dir_offset = [-1, 0, 1]

        for bound in [self.innerbound, self.outerbound]:
            last_point = bound[-1]
            for point in bound:
                for i in range(len(sens_dir_offset)):
                    d = rayLineIntersect((car.point, car.dir + sens_dir_offset[i]), (last_point, point))
                    if (d and d < sens[i]):
                        sens[i] = d
                last_point = point

        return (sens[0], sens[1], sens[2])

def rayLineIntersect(ray, line):
    '''
    return distance when ray intersects line, or None if does not intersect
    ray = ((x, y), dir)
    line = ((x, y), (x, y))
    '''
    dy = math.sin(ray[1])
    dx = math.cos(ray[1])
    
    lineXdiff = line[1][0] - line[0][0]
    lineYdiff = line[1][1] - line[0][1]

    rayXdiff = line[0][0] - ray[0][0]
    rayYdiff = line[0][1] - ray[0][1]
    
    a_num = lineXdiff * rayYdiff - lineYdiff * rayXdiff
    b_num = dx * rayYdiff - dy * rayXdiff

    denum = lineXdiff * dy - lineYdiff * dx

    if (denum == 0):
        return None

    a = a_num / denum # dist from ray to intersect
    b = b_num / denum # dist from point a of line
    
    return a if (a > 0 and b > 0 and b < 1) else None

def extend(point, dir, size):
    return (point[0] + math.cos(dir) * size, point[1] + math.sin(dir) * size) 

def center(point):
    _scale = (15, -15)
    _offset = (400, 300)
    return (point[0] * _scale[0] + _offset[0], point[1] * _scale[1] + _offset[1])

def main():
    pygame.init()
    pygame.display.set_caption('Quick Start')
    window_surface = pygame.display.set_mode((800, 600))
    background = pygame.Surface((800, 600))

    road = Road().generate_rand(50)

    is_running = True
    while is_running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                is_running = False
                print("yes")

        road.draw(background, '#ffffff')

        window_surface.blit(background, (0, 0))
        pygame.display.update()

if __name__ == "__main__":
    main()