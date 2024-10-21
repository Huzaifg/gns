import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def shape_matching_update(initial_positions, current_positions, masses=None, alpha=1, external_forces=None):
    """
    Updates the current positions by pulling them towards the goal positions based on the initial positions using shape matching.

    :param initial_positions: Tensor of initial positions (N x 3), where N is the number of points
    :param current_positions: Tensor of current positions (N x 3) of the points
    :param masses: Tensor of masses (N) for each point
    :param alpha: Stiffness parameter controlling how much the points are pulled towards the goal position (0 < alpha <= 1)
    :param external_forces: Tensor of external forces (N x 3), default is None
    :return: Tensor of updated positions (N x 3) after being pulled towards the goal positions
    """
    if masses is None:
        masses = torch.ones(initial_positions.shape[0], dtype=torch.float32)

    # Compute the center of mass of the initial positions
    initial_com = (initial_positions * masses.unsqueeze(-1)
                   ).sum(dim=0) / masses.sum()
    # Compute the center of mass of the current positions
    current_com = (current_positions * masses.unsqueeze(-1)
                   ).sum(dim=0) / masses.sum()

    # Compute the relative positions to the center of mass
    q = initial_positions - initial_com
    p = current_positions - current_com

    # Compute the covariance matrix Apq
    Apq = torch.zeros((3, 3))
    for i in range(len(masses)):
        Apq += masses[i] * torch.outer(p[i], q[i])

    # Perform Singular Value Decomposition (SVD) to find the optimal rotation
    U, S, Vt = torch.svd(Apq)
    R = torch.mm(U, Vt.t())

    # Ensure R is a proper rotation (no reflection)
    if torch.det(R) < 0:
        U[:, -1] *= -1
        R = torch.mm(U, Vt.t())

    # Compute the goal positions
    goal_positions = torch.mm(R, q.t()).t() + current_com

    # If external forces are not provided, assume zero forces
    if external_forces is None:
        external_forces = torch.zeros_like(current_positions)

    # Update velocities (starting with zero for simplicity)
    velocities = torch.zeros_like(current_positions)

    # Update velocities and positions based on goal positions and external forces
    new_velocities = velocities + alpha * \
        (goal_positions - current_positions) + \
        external_forces / masses.unsqueeze(-1)
    new_positions = current_positions + new_velocities

    return new_positions


def visualize_shape_matching(initial_positions, current_positions, new_positions):
    """
    Visualizes the initial, current, and updated (shape-matched) positions in a 3D plot.

    :param initial_positions: Tensor of initial positions (N x 3), where N is the number of points
    :param current_positions: Tensor of current positions (N x 3) of the points
    :param new_positions: Tensor of updated (shape-matched) positions (N x 3) of the points
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot initial positions (blue)
    ax.scatter(initial_positions[:, 0], initial_positions[:, 1],
               initial_positions[:, 2], color='blue', label='Initial Positions')

    # Plot current positions (red)
    ax.scatter(current_positions[:, 0], current_positions[:, 1],
               current_positions[:, 2], color='red', label='Current Positions')

    # Plot updated (shape-matched) positions (green)
    ax.scatter(new_positions[:, 0], new_positions[:, 1], new_positions[:, 2],
               color='green', label='Updated (Shape-Matched) Positions')

    # Define edges of the cube
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
    ]

    # Plot lines for initial positions
    for start, end in edges:
        ax.plot([initial_positions[start, 0], initial_positions[end, 0]],
                [initial_positions[start, 1], initial_positions[end, 1]],
                [initial_positions[start, 2], initial_positions[end, 2]], color='blue', linestyle='dotted')

    # Plot lines for current positions
    for start, end in edges:
        ax.plot([current_positions[start, 0], current_positions[end, 0]],
                [current_positions[start, 1], current_positions[end, 1]],
                [current_positions[start, 2], current_positions[end, 2]], color='red', linestyle='dotted')

    # Plot lines for updated positions
    for start, end in edges:
        ax.plot([new_positions[start, 0], new_positions[end, 0]],
                [new_positions[start, 1], new_positions[end, 1]],
                [new_positions[start, 2], new_positions[end, 2]], color='green', linestyle='dotted')

    # Set labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.show()


def main():
    # Initial positions of the points forming a cube
    initial_positions = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0]
    ], dtype=torch.float32)

    # Current positions (translated, rotated, and slightly skewed cube)
    current_positions = torch.tensor([
        [0.1, -0.1, 0.2],
        [1.1, 0.0, 0.3],
        [1.2, 1.1, 0.1],
        [0.0, 1.0, 0.2],
        [0.1, -0.1, 1.2],
        [1.0, 0.1, 1.3],
        [1.1, 1.2, 1.1],
        [0.0, 1.1, 1.2]
    ], dtype=torch.float32)

    # Call the shape matching function to get the updated positions
    new_positions = shape_matching_update(
        initial_positions, current_positions, alpha=0.5)

    # Visualize the initial, current, and updated positions
    visualize_shape_matching(
        initial_positions, current_positions, new_positions)


if __name__ == "__main__":
    main()
