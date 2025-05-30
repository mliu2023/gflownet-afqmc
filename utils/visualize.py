from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrow, Rectangle, Circle
import matplotlib.pyplot as plt
import numpy as np
import imageio
from PIL import Image
import io
from sklearn.metrics import r2_score

x1 = 0.35
x2 = 0.65
pad1 = 0.2
pad2 = 0.2
line_pad = 0.2
particle_separation = 0.3


def _base_lattice(lattice_width, lattice_height):
    return tuple(
        [
            plt.gca().add_patch(
                plt.Circle(
                    (x * particle_separation, y * particle_separation),
                    0.05,
                    fc=(0.9, 0.9, 0.9),
                    zorder=4,
                )
            )
            for y in range(lattice_width)
            for x in range(lattice_height)
        ]
        + [
            plt.gca().add_line(
                Line2D(
                    [-line_pad, (lattice_height - 1) * particle_separation + line_pad],
                    [x * particle_separation, x * particle_separation],
                    color="grey",
                    zorder=1,
                )
            )
            for x in range(lattice_width)
        ]
        + [
            plt.gca().add_line(
                Line2D(
                    [x * particle_separation, x * particle_separation],
                    [-line_pad, (lattice_width - 1) * particle_separation + line_pad],
                    color="grey",
                    zorder=1,
                )
            )
            for x in range(lattice_height)
        ]
        + [
            # Add a square around the image
            plt.gca().add_patch(
                Rectangle(
                    (0 - line_pad, 0 - line_pad),
                    line_pad * 2 + (lattice_height - 1) * particle_separation,
                    line_pad * 2 + (lattice_width - 1) * particle_separation,
                    fill=None,
                    edgecolor="black",
                    zorder=0,
                    lw=2,
                )
            )
        ]
    )


def _draw_arrows(lattice):
    width, height = lattice.shape
    for x in range(width):
        for y in range(height):
            if lattice[x][y] == -1: # spin equals -1
                plt.gca().add_patch(
                    FancyArrow(
                        y * particle_separation,
                        (width - 1 - x) * particle_separation,
                        0,
                        -0.1,
                        width=0.020,
                        head_width=0.04,
                        head_length=0.020,
                        fc="red",
                        ec="red",
                        zorder=3,
                    )
                )
            elif lattice[x][y] == 1: # spin equals 1
                plt.gca().add_patch(
                    FancyArrow(
                        y * particle_separation,
                        (width - 1 - x) * particle_separation,
                        0,
                        0.1,
                        width=0.020,
                        head_width=0.04,
                        head_length=0.020,
                        fc="blue",
                        ec="blue",
                        zorder=3,
                    )
                )
            elif lattice[x][y] == 0: # void
                plt.gca().add_patch(
                    Circle(
                        (y * particle_separation, (width - 1 - x) * particle_separation),
                        .04,
                    )
                )
            else:
                 raise ValueError("Invalid value in lattice: must be equal to -1, 0, or 1.") 

def _draw_lattice(lattice, reward=None):
    width, height = lattice.shape[0], lattice.shape[1]
    _base_lattice(lattice_width=width, lattice_height=height)
    _draw_arrows(lattice)
    if reward is not None:
        plt.title(f"{reward=}")
    plt.axis("scaled")
    plt.axis("off")
    # plt.show()


def visualize_trajectory(
    trajectory, filename="trajectory.gif", reward=None
):
    """
    Create a GIF from a list of states using the `visualize_state` function.
    states (list): A list of states to visualize.
    filename (str): Name of the output GIF file.
    duration (float): Duration of each frame in the GIF in seconds.
    """
    images = []
    last_index = len(trajectory) - 1

    for index, state in enumerate(trajectory):
        # Visualize the state using your existing function
        if index == last_index:
            _draw_lattice(state, reward=reward)
        elif index == 0:
            _draw_lattice(state, reward="starting")
        else:
            _draw_lattice(state, reward="intermediate")

        # Save the plot to a PNG buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        image = Image.open(buffer)

        # Convert the image to an array and add to the list
        images.append(np.array(image))
        plt.close()

    # Save images as a GIF
    imageio.mimsave(filename, images, fps=4)

def visualize_terminal_state(lattice, filename):
    plt.imsave(filename, lattice, cmap='viridis', vmin=-1, vmax=1)

def visualize_terminal_states(lattices, filename, cols):
    num_lattices = len(lattices)
    rows = (num_lattices + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
    axes = np.array(axes).reshape(rows, cols)
    
    for i, ax in enumerate(axes.flat):
        if i < num_lattices:
            ax.imshow(lattices[i], cmap='viridis', vmin=-1, vmax=1)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)

def visualize_distribution(values, filename):
    plt.hist(values, bins=20)
    plt.savefig(filename)
    plt.close()

def visualize_parity_plot(energies, energies_pred, filename):
    plt.scatter(energies, energies_pred)
    plt.plot(energies, np.poly1d(np.polyfit(energies, energies_pred, 1))(energies))
    r_squared = r2_score(energies_pred, energies)
    plt.title(f"r^2 = {r_squared:.3f}")
    plt.savefig(filename)
    plt.close()