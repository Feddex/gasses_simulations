import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

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


# DEFINING A SET OF INITIAL STATES:

# Defining the random initial state 
def initial_random_state(num_of_particles, particle_mass, particle_radius, size_box):
    particle_vector = []
    mass = particle_mass
    rad = particle_radius
    
    for _ in range(num_of_particles):
        init_pos = rad + np.random.rand(2) * (0.5 * size_box - 2 * rad)
        init_vel = np.random.rand(2) * 2.5
        new_particle = Particle(mass, rad, init_pos, init_vel)
        
        # cheking for misplacemnts in the intial state
        if not misplacement_checker(new_particle, particle_vector):
           particle_vector.append(new_particle)
        
    return particle_vector

# defining a simple test initial state for debugging 
def test_initial(particle_mass, particle_radius):
    particle_vector = []
    mass = particle_mass
    rad = particle_radius

    particle1 = Particle(mass, rad, [7.0, 50.0], [0.0, 0.001])
    particle_vector.append(particle1)
    particle2 = Particle(10*mass, rad, [7.0, 7.0], [0.0, +4.5])
    particle_vector.append(particle2)

    return particle_vector

# defining the trigger inital state: the molecules of the gas are set still and one single particle moves at high speed at the beginning 
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
    
    # Define the triggering particle with high velocity
    trigger = Particle(mass, rad, [size_box / 2, 2 * size_box / 3], [0.0, 50.0])
    particle_vector.append(trigger)
        
    return particle_vector


# SETTING THE SYSTEM AND EVOLVING IT 
# Main parameters
num_of_particles = 400
time_step = 0.3
total_steps = 50
box_size = 200
particle_radius = 2
particle_mass = 1

# Initialize particles and physics engine
particle_state = initial_trigger(num_of_particles, particle_mass, particle_radius, box_size) 
p_e = PhysicsEngine(time_step, total_steps, particle_state, box_size)
p_e.full_evolution()


# DISPLAYING THE RESULTS 

# Create output directory
output_dir = "frames"
os.makedirs(output_dir, exist_ok=True)
print(f"Directory '{output_dir}' created or already exists.")

# Prepare for plotting
matplotlib.use('Agg')
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 2, 1)
hist = fig.add_subplot(1, 2, 2)

plt.subplots_adjust(bottom=0.2, left=0.15)

ax.axis('equal')
ax.set_xlim([0, box_size])
ax.set_ylim([0, box_size])

# Iterate over all time steps and save each frame
for i in range(total_steps):
    time = i * time_step
    print(f"Processing time step {i}...")

    # Clearing previous plot
    ax.clear()
    ax.set_xlim([0, box_size])
    ax.set_ylim([0, box_size])

    # Update particle positions
    circle = []
    for j in range(len(particle_state)):
        if i < len(particle_state[j].previous_positions):
            pos = particle_state[j].previous_positions[i]
            c = plt.Circle((pos[0], pos[1]), particle_state[j].radius, ec="black", lw=1.5, zorder=20, color=particle_state[j].previous_color[i])
            circle.append(c)
        else:
            print(f"Skipping particle {j} for time step {i} due to insufficient position data.")

    for c in circle:
        ax.add_patch(c)

    # Update histogram
    hist.clear()
    vel_mod = []
    for j in range(len(particle_state)):
        if i < len(particle_state[j].previous_vel_module):
            vel_mod.append(particle_state[j].previous_vel_module[i])
        else:
            print(f"Skipping velocity module of particle {j} for time step {i} due to insufficient data.")

    hist.hist(vel_mod, bins=30, density=True, label="Simulation Data")
    hist.set_xlabel("Speed")
    hist.set_ylabel("Frequency Density")

    # Compute 2D Boltzmann distribution
    E = total_Energy(particle_state, i)
    Average_E = E / len(particle_state)
    k = 1.38064852e-23
    T = 2 * Average_E / (2 * k)
    m = particle_state[0].mass
    v = np.linspace(0, 10, 120)
    fv = m * np.exp(-m * v**2 / (2 * T * k)) / (2 * np.pi * T * k) * 2 * np.pi * v
    hist.plot(v, fv, label="Maxwell–Boltzmann distribution")
    hist.legend(loc="upper right")

    # Save the frame as an image file
    frame_filename = os.path.join(output_dir, f"frame_{i:04d}.png")
    try:
        plt.savefig(frame_filename)
        print(f"Saved frame to '{frame_filename}'.")
    except Exception as e:
        print(f"Failed to save frame {i}: {e}")

print('Simulation and plotting complete.')
