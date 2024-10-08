import numpy as np
from particle import Particle

# Defining the PhysicsEngine class, which manages the time evolution of the system. 
class PhysicsEngine:
    def __init__(self, time_step, total_steps, particle_array, box_size):
        self.time_step = time_step
        self.total_steps = total_steps
        self.particle_array = particle_array
        self.box_size = box_size
    
    # Evolving a single particle by a single step
    def evolve_steps(self, particle, time_step):
        particle.pos += time_step * particle.vel
        particle.previous_positions.append(np.copy(particle.pos))
        particle.previous_velocities.append(np.copy(particle.vel))
        
        velocity_magnitude = np.linalg.norm(particle.vel)
        particle.previous_vel_module.append(velocity_magnitude)
        
        # Update color based on the new velocity
        particle.color = particle.update_color(velocity_magnitude)
        particle.previous_color.append(particle.color)
    
    # Checking the collision conditions between particles: once the collision conditions are met, the velocities are updated accordingly.
    def collision(self, part1, part2):
        vector_dist = part1.pos - part2.pos
        dist = np.linalg.norm(vector_dist)
        rad1, rad2 = part1.radius, part2.radius
        m1, m2 = part1.mass, part2.mass
        
        if dist < rad1 + rad2:
            normal = vector_dist / dist
            rel_vel = part1.vel - part2.vel
            v_normal = np.dot(rel_vel, normal)
            
            if v_normal < 0:
                part1.vel -= (2 * m2 / (m1 + m2)) * v_normal * normal
                part2.vel += (2 * m1 / (m1 + m2)) * v_normal * normal
    
    # Checking the wall collision conditions: if fullfilled, the velocities are updated accordingly to reflection.
    def bound_check(self, part):
        part_pos = part.pos
        part_radius = part.radius

        if part_pos[0] - part_radius < 0 or part_pos[0] + part_radius > self.box_size:
            part.vel[0] = -part.vel[0]

        if part_pos[1] - part_radius < 0 or part_pos[1] + part_radius > self.box_size:
            part.vel[1] = -part.vel[1]
    
    # Evolving the entire system by one step
    def evolve(self):
        for i in range(len(self.particle_array)):
            self.bound_check(self.particle_array[i])
            for j in range(i + 1, len(self.particle_array)):
                self.collision(self.particle_array[i], self.particle_array[j])
        
        for particle in self.particle_array:
            self.evolve_steps(particle, self.time_step)
    
    # Evolving the entire system until the end of the simulation 
    def full_evolution(self):
        for _ in range(self.total_steps):
            self.evolve()

# Defining the total_Energy function to compute the energy of the system implied in the B-M distribution 
def total_Energy(particle_list, index): 
    return sum([particle_list[i].mass / 2. * particle_list[i].previous_vel_module[index]**2 for i in range(len(particle_list))])


# Checking if two particles overlap
def single_misplacement_check(part1, part2):
    pos1, pos2 = part1.pos, part2.pos
    rad1, rad2 = part1.radius, part2.radius 
    vector_distance = pos1 - pos2
    dist = np.linalg.norm(vector_distance)
    return dist < rad1 + rad2

# Checking if a particle is overlapping with the ones stored inside a particle_vector 
def misplacement_checker(particle, particle_vector):
    for part in particle_vector:
        if single_misplacement_check(particle, part):
            return True
    return False

