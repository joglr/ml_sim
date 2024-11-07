import pygame

class BoomAnimation:
    def __init__(self, x, y, duration=30):
     
        self.x = x
        self.y = y
        self.duration = duration
        self.counter = 0
        self.colors = [(255, 0, 0), (255, 127, 0), (255, 255, 0), (255, 255, 255)]
    
    def update(self):
        self.counter += 1
        if self.counter == self.duration:
            return True
        return False
    
    def draw(self, screen):
        if self.counter < self.duration:
            color_index = (self.counter // (self.duration // len(self.colors))) % len(self.colors)
            color = self.colors[color_index]
            size = 50 + 2 * self.counter
            font = pygame.font.SysFont("comicsansms", 172)#pygame.font.Font(self.font, size)
            text_surface = font.render('BOOM', True, color)
            text_rect = text_surface.get_rect(center=(self.x, self.y))
            screen.blit(text_surface, text_rect)

