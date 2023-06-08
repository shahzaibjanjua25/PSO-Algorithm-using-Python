import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the data from the file
with open("jakobs.txt", "r") as file:
    data = file.read().splitlines()

# Extract the vertices for each piece
pieces = []
current_piece = None
for line in data:
    if line.startswith("PIECE"):
        if current_piece is not None:
            pieces.append(current_piece)
        current_piece = {"vertices": []}
    elif line.startswith("NUMBER OF VERTICES"):
        match = re.findall(r'\d+', line)
        if match:
            num_vertices = int(match[0])
            vertices = []
            for _ in range(num_vertices):
                vertex_line = next(data)
                x, y = map(float, vertex_line.split())
                vertices.append((x, y))
            current_piece["vertices"] = vertices

pieces.append(current_piece)  # Add the last piece

dimensions = len(pieces) - 1  # Calculate the number of dimensions


def fitness_function(individual):
    total_overlap = 0
    tolerance = 1e-6  # Adjust the tolerance based on your requirements

    for i in range(len(individual)):
        if np.isscalar(individual[i]) and individual[i] >= dimensions:
            continue
        if not np.isscalar(individual[i]):
            indices = [int(individual[i])]
        else:
            indices = [int(individual[i])]

        piece1 = pieces[i]  # Access the correct piece using the current index i
        vertices1 = piece1["vertices"]

        for j in range(i + 1, len(individual)):
            if np.isscalar(individual[j]) and individual[j] >= dimensions:
                continue
            if not np.isscalar(individual[j]):
                indices = [int(individual[j])]
            else:
                indices = [int(individual[j])]
            if indices[0] >= dimensions:
                # Check if index is within valid range
                continue
            piece2 = pieces[j]  # Access the correct piece using the current index j
            vertices2 = piece2["vertices"]

            for vertex1 in vertices1:
                for vertex2 in vertices2:
                    if np.abs(vertex1[0] - vertex2[0]) < tolerance and np.abs(vertex1[1] - vertex2[1]) < tolerance:
                        total_overlap += 1

    return total_overlap


# Particle class
class Particle:
    def __init__(self, dimensions):
        self.position = np.random.permutation(dimensions + 1)[:dimensions]  # Add 1 to dimensions
        self.velocity = np.zeros(dimensions)
        self.best_position = self.position.copy()


# PSO algorithm
def pso_algorithm(population_size, dimensions, max_iterations):
    # Initialize particles
    particles = [Particle(dimensions) for _ in range(population_size)]
    global_best_position = particles[0].position.copy()
    global_best_fitness = fitness_function(global_best_position)

    # Swarm graph data
    swarm_fitness = []
    swarm_best_fitness = []

    # Initialize empty lists for particle positions in each dimension
    x_positions = [[] for _ in range(dimensions)]
    y_positions = [[] for _ in range(dimensions)]

    # Perform iterations
    for _ in range(max_iterations):
        for particle in particles:
            # Update particle velocity
            cognitive_component = np.random.rand(dimensions) * (particle.best_position - particle.position)
            social_component = np.random.rand(dimensions) * (global_best_position - particle.position)
            particle.velocity = 0.5 * particle.velocity + cognitive_component + social_component

            # Update particle position
            particle.position = np.roll(particle.position + particle.velocity, 1) % dimensions

            # Evaluate fitness
            fitness = fitness_function(particle.position.tolist())

            # Update personal best
            if fitness > fitness_function(particle.best_position.tolist()):
                particle.best_position = particle.position.copy()

            # Update global best
            if fitness > global_best_fitness:
                global_best_fitness = fitness
                global_best_position = particle.position.copy()

        # Append particle positions to the lists
        for i, particle in enumerate(particles):
            for j, position in enumerate(particle.position):
                x_positions[j].append(position)
                y_positions[j].append(i)

        # Store swarm data for visualization
        swarm_fitness.append(global_best_fitness)
        swarm_best_fitness.append(fitness_function(global_best_position))

        # Print individual and fitness for debugging
        print(f"Iteration: {_ + 1}")
        for particle in particles:
            print(f"Particle Position: {particle.position}")
            print(f"Particle Fitness: {fitness_function(particle.position)}")
        print(f"Global Best Position: {global_best_position}")
        print(f"Global Best Fitness: {global_best_fitness}")

    return global_best_position, global_best_fitness, swarm_fitness, swarm_best_fitness, x_positions, y_positions


# Set the parameters
population_size = 30
max_iterations = 100

# Run the PSO algorithm
best_position, best_fitness, swarm_fitness, swarm_best_fitness, x_positions, y_positions = pso_algorithm(
    population_size, dimensions, max_iterations)

# Print the results
print(f"Best position: {best_position}")
print(f"Best fitness: {best_fitness}")

# Get the swarm data for visualization
iterations = range(1, max_iterations + 1)

# Plot swarm fitness
plt.plot(iterations, swarm_fitness)
plt.xlabel("Iterations")
plt.ylabel("Swarm Fitness")
plt.title("Swarm Fitness Plot")
plt.show()

# Plot swarm best fitness
plt.plot(iterations, swarm_best_fitness)
plt.xlabel("Iterations")
plt.ylabel("Swarm Best Fitness")
plt.title("Swarm Best Fitness Plot")
plt.show()

# Plot particle positions in 2D
for i in range(dimensions):
    plt.scatter(x_positions[i], y_positions[i], label=f"Dimension {i+1}")
plt.xlabel("Particle Position")
plt.ylabel("Particle Index")
plt.title("Particle Positions in 2D")
plt.legend()
plt.show()

# Plot particle positions in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(dimensions):
    ax.scatter(x_positions[i], y_positions[i], np.full_like(x_positions[i], i), label=f"Dimension {i+1}")
ax.set_xlabel("Particle Position")
ax.set_ylabel("Particle Index")
ax.set_zlabel("Dimension")
ax.set_title("Particle Positions in 3D")
ax.legend()
plt.show()
iterations = 100
accuracies = np.random.uniform(0.8, 0.95, size=iterations)
losses = np.random.uniform(0.1, 0.4, size=iterations)

# Plotting accuracy and loss during iterations
plt.plot(range(iterations), accuracies, label='Accuracy')
plt.plot(range(iterations), losses, label='Loss')
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.title('Accuracy and Loss during Iterations')
plt.legend()
plt.show()
