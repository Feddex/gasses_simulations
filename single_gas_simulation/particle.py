import numpy as np

# defining the particle class
class Particle:
    def __init__(self, mass, radius, position, velocity):
        self.mass = mass
        self.radius = radius
        
        self.pos = np.array(position)
        self.vel = np.array(velocity)

        # storing the positions, velocities and their norm
        self.previous_positions = [np.copy(self.pos)]
        self.previous_velocities = [np.copy(self.vel)]
        self.previous_vel_module = [np.linalg.norm(self.vel)] 
        
        self.color = self.update_color(np.linalg.norm(self.vel))  
        self.previous_color = [self.color]  

    def update_color(self, velocity_magnitude):
        if 0 <= velocity_magnitude < 1:
            return "blue"
        elif 1 <= velocity_magnitude < 2:
            return "green"
        elif 2 <= velocity_magnitude < 3:
            return "yellow"
        elif 3 <= velocity_magnitude < 4:
            return "red"
        elif 4 <= velocity_magnitude < 5:
            return "purple"
        else:
            return "black"  # Default color to depict particles whose velocity is outside the ranges 