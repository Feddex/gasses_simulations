import numpy as np
from particle import Particle
from physics_engine import misplacement_checker

def two_gas_simulation(num_fist_gas, num_second_gas, part_mass1, part_mass2, part_radius1, part_radius2, size_box):
    fisrt_gas_particle_vector, second_gas_particle_vector = [], []

    mass1, mass2 = part_mass1, part_mass2
    rad1 = part_radius1
    rad2 = part_radius2

    for _ in range(num_fist_gas):
        init_pos = np.array([rad1 + np.random.rand() * size_box * 0.75, rad1 + np.random.rand() * size_box - 2 * rad1 ])
        init_vel = np.array([np.random.rand() * 4, 0.0])
        new_particle = Particle(mass1, rad1, init_pos, init_vel)
        new_particle.color = 'blue'

        if not misplacement_checker(new_particle, fisrt_gas_particle_vector):
           fisrt_gas_particle_vector.append(new_particle)

    for _ in range(num_second_gas):
        init_pos = np.array([size_box * 0.75 - rad2 + np.random.rand() * size_box * 0.25, rad2 + np.random.rand() * size_box - 2 * rad2 ])
        init_vel = np.array([-np.random.rand() * 4, 0.0])
        new_particle = Particle(mass2, rad2, init_pos, init_vel)
        new_particle.color = 'red'

        if not misplacement_checker(new_particle, second_gas_particle_vector):
           second_gas_particle_vector.append(new_particle)
    
    return [fisrt_gas_particle_vector, second_gas_particle_vector]
