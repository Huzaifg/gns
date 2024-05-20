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
    if radius_match == None:
        radius_match = re.search(r"Connectivity radius is set to: ([\d.]+)", file_content)
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

# Create the "plots" folder if it doesn't exist
if not os.path.exists("plots"):
    os.makedirs("plots")
# Extract only folder name for base_folder
base_folder = base_folder.split("/")[-2]
print(base_folder)

if not os.path.exists(f"plots/{base_folder}"):
    os.makedirs(f"plots/{base_folder}")

# Set seaborn style with a light grid
sns.set_palette("colorblind")
plt.rcParams["figure.dpi"] = 600
plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "sans-serif"
sns.set(style="whitegrid", rc={"grid.color": "0.9"})  # Setting grid color to a very light shade of gray

# Loop over index
for connectivity_radius, df in combined_df.groupby(level=0):
    print(f"Connectivity Radius: {connectivity_radius}")
    print(df)
    # Plot 1: Time taken vs num of particles
    plt.figure(figsize=(10, 6))
    sns.regplot(data=df, x="particles", y="time_taken", scatter=True, fit_reg=True, ci=None, order=2)
    plt.xlabel("Number of Graph Nodes", fontsize=14)
    plt.ylabel("RTF", fontsize=14)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.ylim(0, plt.ylim()[1] + 50)
    plt.xlim(0)
    plt.title(f"Connectivity Radius: {connectivity_radius}")
    plt.tight_layout()
    plt.savefig(f"plots/{base_folder}/rtf_vs_particles_{connectivity_radius}.png")

    # Plot 2: Time taken vs num of edges
    plt.figure(figsize=(10, 6))
    sns.regplot(data=df, x="edges", y="time_taken", scatter=True, fit_reg=True, ci=None, order=2)
    plt.xlabel("Number of Graph Edges",  fontsize=14)
    plt.ylabel("RTF",  fontsize=14)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.ylim(0, plt.ylim()[1] + 50)
    plt.xlim(0)
    plt.title(f"Connectivity Radius: {connectivity_radius}")
    plt.tight_layout()
    plt.savefig(f"plots/{base_folder}/rtf_vs_edges_{connectivity_radius}.png", dpi=600)


# Plot 3: RTF vs num_of_particles for each connectivity radius but on one plot
plt.figure(figsize=(10, 6))
ax = plt.gca()

for connectivity_radius, df in combined_df.groupby(level=0):
    print(f"Connectivity Radius: {connectivity_radius}")
    print(df)
    # Plot 1: Time taken vs num of particles
    sns.regplot(data=combined_df.loc[connectivity_radius], x="particles", y="time_taken", scatter=True, fit_reg=True, ci=None, order=2, label=f"{connectivity_radius}", ax=ax)

ax.set_xlabel("Number of Graph Nodes",  fontsize=14)
ax.set_ylabel("RTF", fontsize=14)
ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
ax.set_ylim(0, ax.get_ylim()[1] + 50)
ax.set_xlim(0)
ax.legend(title="Connectivity Radius")
plt.text(0.999, 0.98, "Larger Connectivity Radius have more edges", fontsize='small', verticalalignment='top', horizontalalignment='right', transform=plt.gca().transAxes)
plt.tight_layout()
plt.savefig(f"plots/{base_folder}/rtf_vs_particles_all.png", dpi=600)
