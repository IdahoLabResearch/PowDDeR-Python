from adaptive_capacity_resilience import *
from PowDDeR_plot_functions import *

# Note:
# Time of the adaptive capacity calculation is set in adaptive_capacity_resilience (line 225)

# Create assets with default values
battery = Battery()
solar = Solar()

# Print the battery and solar
print(battery)
print(solar)

# Create asset without default values
dam = Dam('Hydro Gen 3MW', P_output=1000, Q_output=5, P_nameplate_pos_neg=[3000, 0], Q_nameplate_pos_neg=[3000, -3000], P_real_time_max=2500)

print(dam)

# Create a bus and add the assets to get aggregation
bus = Bus()
bus.add_asset_to_bus(battery)
bus.add_asset_to_bus(solar)

# Plot the bus manifold
plot_ac(bus)

# Plot the dam manifold
plot_ac(dam)
