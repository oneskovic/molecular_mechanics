from xyz2graph import MolGraph, to_plotly_figure
from plotly.offline import offline

# Create the MolGraph object
mg = MolGraph()

# Read the data from the .xyz file
mg.read_xyz("dna.xyz")

# Create the Plotly figure object
fig = to_plotly_figure(mg)

# Plot the figure
offline.plot(fig)
