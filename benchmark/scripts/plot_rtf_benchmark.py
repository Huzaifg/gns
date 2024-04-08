import pandas as pd
import re
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Load the text file content
# Function to process a single file
def process_benchmark_file(file_path):
    with open(file_path, "r") as file:
        file_content = file.read()

    # Define regex pattern to find the relevant information
    pattern = re.compile(
        r"Number of particles: (?P<particles>\d+).*?"
        r"Time taken for rollout: (?P<time_taken>[\d.]+).*?"
        r"Num of edges: (?P<edges>\d+)",
        re.DOTALL
    )
    
    # Extract connectivity radius
    radius_match = re.search(r"connectivity radius is set to: ([\d.]+)", file_content)
    if radius_match:
        connectivity_radius = float(radius_match.group(1))
    else:
        connectivity_radius = None

    # Extract simulation data
    matches = pattern.findall(file_content)
    df = pd.DataFrame(matches, columns=["particles", "time_taken", "edges"])
    df = df.astype({"particles": int, "time_taken": float, "edges": int})
    df["connectivity_radius"] = connectivity_radius
    
    return df

base_folder = "../output/"
file_paths = list(Path(base_folder).rglob("*.txt"))

# Process each file and combine the results
combined_df = pd.concat([process_benchmark_file(path) for path in file_paths], ignore_index=True)

# Reorganize the DataFrame to have connectivity_radius as a part of the index
combined_df.set_index(["connectivity_radius", combined_df.index], inplace=True)

# Trail print
print(combined_df.loc[0.025])


# Create the "plots" folder if it doesn't exist
if not os.path.exists("plots"):
    os.makedirs("plots")

# Set the common rcParams for all plots
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "sans-serif"
# Loop over index
for connectivity_radius, df in combined_df.groupby(level=0):
    print(f"Connectivity Radius: {connectivity_radius}")
    print(df)
    # Plot 1: Time taken vs num of particles
    plt.figure(figsize=(10, 6))
    sns.regplot(data=combined_df.loc[connectivity_radius], x="particles", y="time_taken", scatter=True, fit_reg=True, ci=None, order=2)
    plt.xlabel("Number of Graph Nodes")
    plt.ylabel("RTF")
    plt.axvline(x=100000, color='grey', linestyle='--')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.ylim(0, plt.ylim()[1] + 50)
    plt.title(f"Connectivity Radius: {connectivity_radius}")
    plt.tight_layout()
    plt.savefig(f"plots/rtf_vs_particles_{connectivity_radius}.png")

    # Plot 2: Time taken vs num of edges
    plt.figure(figsize=(10, 6))
    sns.regplot(data=combined_df.loc[connectivity_radius], x="edges", y="time_taken", scatter=True, fit_reg=True, ci=None, order=2)
    plt.xlabel("Number of Graph Edges")
    plt.ylabel("RTF")
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.ylim(0, plt.ylim()[1] + 50)
    plt.title(f"Connectivity Radius: {connectivity_radius}")
    plt.tight_layout()
    plt.savefig(f"plots/rtf_vs_edges_{connectivity_radius}.png", dpi=300)


# create new figure
plt.figure(figsize=(10, 6))
# Create a 2D surface plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Generate the data for the plot
x = combined_df["particles"]
y = combined_df["edges"]
z = combined_df["time_taken"]

# Plot the surface
ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none')

# Set labels and title
ax.set_xlabel('Number of Graph Nodes')
ax.set_ylabel('Number of Graph Edges')
ax.set_zlabel('RTF')
ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
ax.set_ylim(0, ax.get_ylim()[1] + 50)
ax.set_title(r'$RTF$ vs Number of Particles ($1e^6$) vs Number of Edges ($1e^6$)')

# Show the plot
# plt.show()


# Plot 3: RTF vs num_of_particles for each connectivity radius but on one plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)


for connectivity_radius, df in combined_df.groupby(level=0):
    print(f"Connectivity Radius: {connectivity_radius}")
    print(df)
    # Plot 1: Time taken vs num of particles
    sns.regplot(data=combined_df.loc[connectivity_radius], x="particles", y="time_taken", scatter=True, fit_reg=True, ci=None, order=2, label=f"{connectivity_radius}", ax=ax)



ax.set_xlabel("Number of Graph Nodes")
ax.set_ylabel("RTF")
ax.axvline(x=100000, color='grey', linestyle='--')
ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
ax.set_ylim(0, ax.get_ylim()[1] + 50)
ax.legend(title="Connectivity Radius")
plt.text(0.999, 0.98, "Larger Connectivity Radius have more edges", fontsize='small', verticalalignment='top', horizontalalignment='right', transform=plt.gca().transAxes)
plt.tight_layout()
plt.savefig(f"plots/rtf_vs_particles_all.png", dpi=300)