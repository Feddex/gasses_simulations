import numpy as np
from particle import Particle
from physics_engine import misplacement_checker

def initial_random_state(num_of_particles, particle_mass, particle_radius, size_box):
    particle_vector = []
    mass = particle_mass
    rad = particle_radius
    
    for _ in range(num_of_particles):
        init_pos = rad + np.random.rand(2) * (size_box - 2 * rad)
        init_vel = np.random.rand(2) * 2.5
        new_particle = Particle(mass, rad, init_pos, init_vel)
        
        if not misplacement_checker(new_particle, particle_vector):
            particle_vector.append(new_particle)
        
    return particle_vector

def test_initial(particle_mass, particle_radius):
    particle_vector = []
    mass = particle_mass
    rad = particle_radius

    particle1 = Particle(mass, rad, [7.0, 50.0], [0.0, 0.001])
    particle_vector.append(particle1)
    particle2 = Particle(10*mass, rad, [7.0, 7.0], [0.0, +4.5])
    particle_vector.append(particle2)

    return particle_vector

def initial_trigger(num_of_particles, particle_mass, particle_radius, size_box):
    particle_vector = []
    mass = particle_mass
    rad = particle_radius
    
    for _ in range(num_of_particles):
        init_pos = np.array([rad + np.random.rand() * (size_box - 2 * rad), rad + np.random.rand() * (0.5 * size_box - 2 * rad)])
        init_vel = np.array([0.0, 0.0])
        new_particle = Particle(mass, rad, init_pos, init_vel)
       
        if not misplacement_checker(new_particle, particle_vector):
            particle_vector.append(new_particle)
    
    trigger = Particle(mass, rad, [size_box / 2, 2 * size_box / 3], [0.0, -40.0])
    particle_vector.append(trigger)
        
    return particle_vector
