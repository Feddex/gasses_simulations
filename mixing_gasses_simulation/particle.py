import numpy as np

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
        
        self.color = 'black'
        self.previous_color = [self.color]
