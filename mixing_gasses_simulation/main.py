import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from initial_state import two_gas_simulation
from physics_engine import PhysicsEngine
from physics_engine import total_Energy

def main():
    # SETTING THE SYSTEM AND EVOLVING IT 
    # Main parameters
    num_fist_gas = 800
    num_second_gas = 200
    time_step = 0.3
    total_steps = 800
    box_size = 300
    particle_radius1 = 2
    particle_radius2 = 2
    particle_mass1 = 1
    particle_mass2 = 5

    # Initialize particles and physics engine
    particle_set = two_gas_simulation(num_fist_gas, num_second_gas, particle_mass1, particle_mass2, particle_radius1, particle_radius2, box_size) 
    particle_state = particle_set[0] + particle_set[1]
    p_e = PhysicsEngine(time_step, total_steps, particle_set, box_size)
    p_e.full_evolution()

    # DISPLAYING THE RESULTS 

    # Create output directory
    output_dir = "frames"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Directory '{output_dir}' created or already exists.")

    # Prepare for plotting
    matplotlib.use('Agg')
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(2, 2, 1)
    hist_cumulative = fig.add_subplot(2, 2, 2)
    hist_first_gas = fig.add_subplot(2,2,3)
    hist_second_gas = fig.add_subplot(2,2,4)

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.2, wspace=0.2)

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
                c = plt.Circle((pos[0], pos[1]), particle_state[j].radius, ec="black", lw=1.5, zorder=20, color=particle_state[j].color)
                circle.append(c)
            else:
                print(f"Skipping particle {j} for time step {i} due to insufficient position data.")

        for c in circle:
            ax.add_patch(c)

        # UPDATE HIST_FISRT_GAS
        hist_first_gas.clear()
        vel_mod = []
        for j in range(len(particle_set[0])):
            if i < len(particle_set[0][j].previous_vel_module):
                vel_mod.append(particle_set[0][j].previous_vel_module[i])
            else:
                print(f"Skipping velocity module of particle {j} for time step {i} due to insufficient data.")

        hist_first_gas.hist(vel_mod, bins=30, density=True, label="Simulation Data")
        hist_first_gas.set_xlabel("Speed")
        hist_first_gas.set_ylabel("Frequency Density")

        # Compute 2D Boltzmann distribution for the first gas 
        E_first_gas = total_Energy(particle_set[0], i)
        Average_E_fisrt_gas = E_first_gas / len(particle_set[0])
        k = 1.38064852e-23
        T_1 = 2 * Average_E_fisrt_gas / (2 * k)
        m_1 = particle_set[0][0].mass  # this is the mass of the first gas 
        v = np.linspace(0, 10, 120)
        fv_1 = m_1 * np.exp(-m_1 * v**2 / (2 * T_1 * k)) / (2 * np.pi * T_1 * k) * 2 * np.pi * v
        hist_first_gas.plot(v, fv_1, label="Maxwell–Boltzmann distribution fisrt gas")
        hist_first_gas.legend(loc="upper right")

        # UPDATE HIST_SECOND_GAS
        hist_second_gas.clear()
        vel_mod = []
        for j in range(len(particle_set[1])):
            if i < len(particle_set[1][j].previous_vel_module):
                vel_mod.append(particle_set[1][j].previous_vel_module[i])
            else:
                print(f"Skipping velocity module of particle {j} for time step {i} due to insufficient data.")

        hist_second_gas.hist(vel_mod, bins=30, density=True, label="Simulation Data", color='red')
        hist_second_gas.set_xlabel("Speed")
        hist_second_gas.set_ylabel("Frequency Density")

        # Compute 2D Boltzmann distribution for the second gas 
        E_second_gas = total_Energy(particle_set[1], i)
        Average_E_second_gas = E_second_gas / len(particle_set[1])
        T_2 = 2 * Average_E_second_gas / (2 * k)
        m_2 = particle_set[1][0].mass  # this is the mass of the second gas 
        fv_2 = m_2 * np.exp(-m_2 * v**2 / (2 * T_2 * k)) / (2 * np.pi * T_2 * k) * 2 * np.pi * v
        hist_second_gas.plot(v, fv_2, label="Maxwell–Boltzmann distribution second gas")
        hist_second_gas.legend(loc="upper right")

        # UPDATE HIST_CUMULATIVE
        hist_cumulative.clear()
        vel_mod = []
        for j in range(len(particle_state)):
            if i < len(particle_state[j].previous_vel_module):
                vel_mod.append(particle_state[j].previous_vel_module[i])
            else:
                print(f"Skipping velocity module of particle {j} for time step {i} due to insufficient data.")

        hist_cumulative.hist(vel_mod, bins=30, density=True, label="Simulation Data")
        hist_cumulative.set_xlabel("Speed")
        hist_cumulative.set_ylabel("Frequency Density")

        # Compute 2D Boltzmann distribution for the gas as a whole 
        p1, p2 = len(particle_set[0]) / len(particle_state), len(particle_set[1]) / len(particle_state) 
        fv_tot = fv_1 * p1 + fv_2 * p2  # applying Bayes Theorem 
        hist_cumulative.plot(v, fv_tot, label="Maxwell–Boltzmann distribution second gas")
        hist_cumulative.legend(loc="upper right")

        # Save the frame as an image file
        frame_filename = os.path.join(output_dir, f"frame_{i:04d}.png")
        try:
            plt.savefig(frame_filename)
            print(f"Saved frame to '{frame_filename}'.")
        except Exception as e:
            print(f"Failed to save frame {i}: {e}")

    print('Simulation and plotting complete.')

    # NEW SECTION 
    ####################################################
    ####################################################

    # these histograms collect the velocities of the particle of the gas all over the time steps, 
    # excluded the velocities of the particles during the first 70 steps when the gas has not reached the equilibrium state yet 
    matplotlib.use('Agg')
    canvas = plt.figure(figsize=(30, 10))
    first_gas_global_hist_dist = canvas.add_subplot(1, 3, 1)
    second_gas_global_hist_dist = canvas.add_subplot(1, 3, 2)
    total_gas_global_hist_dist = canvas.add_subplot(1, 3, 3)

    # Processing the first gas global hist dist 
    vel_module_1 = []

    for i in range(len(particle_set[0])):
        for j in range(total_steps - 70):
            vel_module_1.append(particle_set[0][i].previous_vel_module[j])

    hist_1, bin_edges = np.histogram(vel_module_1, bins=30, density=True)

    first_gas_global_hist_dist.hist(vel_module_1, bins=bin_edges, density=True, label="First gas global distribution", color='blue')
    first_gas_global_hist_dist.set_xlabel("Speed")
    first_gas_global_hist_dist.set_ylabel("Frequency density")

    # Calculate the Maxwell-Boltzmann distribution for the first gas
    Tot_energy = 0  # this is the energy of a gas which has len(particle_state) * (total_step - 70) particles, each with a velocity that the particle in particle_state has or had previously had 
    for i in range(total_steps - 70):
        Tot_energy += total_Energy(particle_set[0], i)

    Average_energy = Tot_energy / ((total_steps - 70) * len(particle_set[0]))

    T = 2 * Average_energy / (2 * k)
    m = particle_set[0][0].mass
    fv_1 = m * np.exp(-m * v**2 / (2 * T * k)) / (2 * np.pi * T * k) * 2 * np.pi * v
    first_gas_global_hist_dist.plot(v, fv_1, label="Maxwell–Boltzmann distribution")
    first_gas_global_hist_dist.legend(loc="upper right")

    # Processing the second gas global hist dist 
    vel_module_2 = []

    for i in range(len(particle_set[1])):
        for j in range(total_steps - 70):
            vel_module_2.append(particle_set[1][i].previous_vel_module[j])

    hist_2, _ = np.histogram(vel_module_2, bins=bin_edges, density=True)

    second_gas_global_hist_dist.hist(vel_module_2, bins=bin_edges, density=True, label="Second gas global distribution", color='red')
    second_gas_global_hist_dist.set_xlabel("Speed")
    second_gas_global_hist_dist.set_ylabel("Frequency density")

    # Calculate the Maxwell-Boltzmann distribution for the second gas
    Tot_energy = 0 
    for i in range(total_steps - 70):
        Tot_energy += total_Energy(particle_set[1], i)

    Average_energy = Tot_energy / ((total_steps - 70) * len(particle_set[1]))

    T = 2 * Average_energy / (2 * k)
    m = particle_set[1][0].mass
    fv_2 = m * np.exp(-m * v**2 / (2 * T * k)) / (2 * np.pi * T * k) * 2 * np.pi * v
    second_gas_global_hist_dist.plot(v, fv_2, label="Maxwell–Boltzmann distribution")
    second_gas_global_hist_dist.legend(loc="upper right")

    # Processing the total gas global hist dist
    vel_module_total = []

    for i in range(len(particle_state)):
        for j in range(total_steps - 70):
            vel_module_total.append(particle_state[i].previous_vel_module[j])

    total_gas_global_hist_dist.hist(vel_module_total, bins=bin_edges, density=True, label="Total gas global distribution")
    total_gas_global_hist_dist.set_xlabel("Speed")
    total_gas_global_hist_dist.set_ylabel("Frequency density")

    fv_tot = fv_1 * p1 + fv_2 * p2
    total_gas_global_hist_dist.plot(v, fv_tot, label="Maxwell–Boltzmann distribution")

    # Add a legend to the total gas global histogram
    total_gas_global_hist_dist.legend(loc="upper right")

    # Save the figure
    canvas.savefig('total_gas_global_hist_dist.png')

if __name__ == "__main__":
    main()
