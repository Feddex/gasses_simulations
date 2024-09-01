import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from initial_state import initial_trigger
from initial_state import initial_random_state
from physics_engine import PhysicsEngine, total_Energy

def get_bin_color(velocity):
    """Return the color based on the velocity magnitude."""
    if 0 <= velocity < 1:
        return "blue"
    elif 1 <= velocity < 2:
        return "green"
    elif 2 <= velocity < 3:
        return "yellow"
    elif 3 <= velocity < 4:
        return "red"
    elif 4 <= velocity < 5:
        return "purple"
    else:
        return "black"

def run_simulation():
    num_of_particles = 500
    time_step = 0.3
    total_steps = 600
    box_size = 200
    particle_radius = 2
    particle_mass = 1

    # Initialize particles and physics engine
    #particle_state = initial_trigger(num_of_particles, particle_mass, particle_radius, box_size) 
    particle_state = initial_random_state(num_of_particles, particle_mass, particle_radius, box_size) 
    p_e = PhysicsEngine(time_step, total_steps, particle_state, box_size)
    p_e.full_evolution()

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

        # Compute histogram data
        counts, bin_edges = np.histogram(vel_mod, bins=30, density=True)

        # Plot histogram with colored bins
        for count, edge in zip(counts, bin_edges[:-1]):
            color = get_bin_color(edge)
            hist.bar(edge, count, width=np.diff(bin_edges), color=color, edgecolor='black')

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

if __name__ == "__main__":
    run_simulation()


# this code will create a directory storing each frame of the simulation, to get a full video run the following command:
#  ffmpeg -r 24 -i frames/frame_%04d.png -vcodec libx264 -pix_fmt yuv420p simulation.mp4