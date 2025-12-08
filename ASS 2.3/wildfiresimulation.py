'''
Advanced Spatial Analysis - Wildfire simulation and prevention – Assignment 2.3
Author: Ruth Femicit Dala

PLAN
Aim and expected outcomes
- To build a simple dynamic wildfire model on a 2D grid that simulates how fire starts, consumes wood, spreads heat, and ignites neighbouring cells over time.
- Expected outcomes: 
    -Time series of grid states for wood, temperature, and burning, 
    -visualisation of wildfire spread and temperature patterns,  
    -simple scenario to test how a firebreak reduces spread.

Input data
(Internal, generated in code – no external files)
1. Wood grid
    File format: in-memory numpy array (optionally saved as .npy if exported)
    Raster/vector/xxx: Raster (regular grid)
    Variable: wood_amount (kg of wood per cell)
    Units (if applicable): kilograms (kg)
    Dimensions (if applicable): [nrows, ncols]
    Resolution (if applicable): 
        - Space: 1 cell = 1 arbitrary spatial unit (e.g. 1 ha or 1 arbitrary grid unit)
        - Time: updated each model timestep Δt (unitless / abstract)
    Projection: Abstract grid (no real-world projection)

2. Temperature grid
    File format: in-memory numpy array (optionally .npy if exported)
    Raster/vector/xxx: Raster (regular grid)
    Variable: temperature (°C)
    Units (if applicable): degrees Celsius (°C)
    Dimensions (if applicable): [nrows, ncols]
    Resolution (if applicable):
        - Space: same as wood grid
        - Time: updated each model timestep Δt
    Projection: Abstract grid
    Other: Initial constant background temperature (e.g. 20 °C)

3. Burning state grid
    File format: in-memory numpy array (optionally .npy if exported)
    Raster/vector/xxx: Raster (regular grid)
    Variable: burning (boolean, True = burning, False = not burning)
    Units (if applicable): dimensionless
    Dimensions (if applicable): [nrows, ncols]
    Resolution (if applicable):
        - Space: same as wood grid
        - Time: updated each model timestep Δt
    Projection: Abstract grid
    Other: Initially all False (no burning); one ignition cell is set to True.

Output data
- Time series of grid states (wood_amount, temperature, burning) in memory.
- Terminal visualisation per timestep using emojis:
    - 🌲 = forest with wood (wood_amount > burned_threshold)
    - 🔥 = burning forest
    - ⬛ = burned / no forest (wood_amount ≤ burned_threshold)
    - Right-hand grid = temperature visualised with color scale.
- Optionally:
    - Saved numpy arrays (.npy) for final or intermediate states.
    - Screenshots / text logs of patterns over time.

Processing steps
1. Model setup
    - Choose grid size (nrows, ncols) as variables so they are easy to change.
    - Create wood_amount grid with random values between 20,000 and 30,000 kg.
    - Create temperature grid with a constant initial temperature (e.g. 20 °C).
    - Create burning grid with all False.
    - Implement display_grid() and use it to visualise the initial state.

2. Ignition
    - Choose an ignition cell (e.g. centre of the grid).
    - Set burning[ign_row, ign_col] = True.
    - Visualise the new state to confirm ignition.

3. Local fire dynamics without spread
    - Implement a function update_ignited_cell(...) that:
        - Burns 20% of the remaining wood in the ignited cell per timestep.
        - Increases temperature based on burned wood (0.6 °C per kg burned).
        - Extinguishes the fire if wood < 100 kg (set burning to False).
    - Run a loop over multiple timesteps but only update this one cell.
    - Visualise at each timestep to see wood decrease and temperature change.

4. Add heat loss to atmosphere
    - Extend the update rule so that in each timestep:
        - 20% of the heat in the cell is lost to the atmosphere 
          (temperature *= 0.8 or equivalent).
    - Re-run the single-cell simulation and visualise again.
    - Check whether the temperature pattern now looks more realistic.

5. Implement heat spread to neighboring cells (single burning cell)
    - Implement get_neighbors(row, col, nrows, ncols) (4-connected neighbours).
    - Implement a function spread_heat_single_cell(...) that:
        - Takes a copy of the temperature grid from the previous timestep.
        - Spreads 50% of the heat from the ignited cell to its neighbours.
        - Subtracts the transferred heat from the original cell.
    - Visualise in time to see how heat “diffuses” away from the source.

6. Apply heat spread for the whole grid
    - Generalise to spread_heat(current_heat) across all cells:
        - Use a copy of the old temperature grid for calculations.
        - Update a new_heat grid for the next timestep (in-place vs out-of-place logic).
    - Visualise the spread from the initial hot cell across the full grid.

7. Link temperature to fire spread (ignition threshold)
    - Implement a function ignite_from_temperature(...) that:
        - Checks all cells where temperature > ignition_threshold (e.g. 500 °C).
        - For those cells with available wood (wood_amount > burned_threshold), sets burning = True.
    - In each timestep:
        - Update burning due to ignition.
        - Update wood and temperature in burning cells.
        - Spread heat across the grid.
        - Apply atmospheric heat loss.
    - Visualise each timestep to see the fire spread.

8. Implement a firebreak
    - Define a strip (e.g. a row or column) with:
        - wood_amount = 0 or very low.
        - burning always forced to False.
    - Add this to the initial conditions.
    - Run the model and analyse whether the fire stops at the firebreak or crosses it.

9. (Optional steps)
    - BONUS: implement wind by biasing heat transfer and ignition probability in one direction.
    - BONUS: implement a firefighting team function that can extinguish burning cells (set burning = False and maybe reduce temperature).
'''

# ---------------------------------------------
# Imports
# ---------------------------------------------

import numpy as np
import time

# ---------------------------------------------
# Parameters for grids 
# ---------------------------------------------

nrows, ncols = 10, 10

INITIAL_WOOD_MIN = 20000.0
INITIAL_WOOD_MAX = 30000.0

INITIAL_TEMPERATURE = 20.0  # °C
BURNED_THRESHOLD = 50.0     # kg -> below this counts as burned / ⬛

WOOD_BURN_FRACTION = 0.2    # burn 20% of remaining wood per timestep
TEMP_INCREASE_PER_KG = 0.6  # °C per kg burned
HEAT_LOSS_FRACTION = 0.2    # 20% of heat lost to atmosphere
HEAT_SPREAD_FRACTION = 0.5  # 50% of heat spreads to neighbours

IGNITION_TEMP = 500.0       # °C
EXTINGUISH_WOOD = 100.0     # kg

N_TIMESTEPS = 50
SLEEP_TIME = 0.5            # seconds between frames

# ---------------------------------------------
# Wind settings (extra – currently turned off)
# ---------------------------------------------

WIND_DIRECTION = "south"     # options: "none", "north", "south", "east", "west"
WIND_FACTOR = 3.0           # how much extra heat goes downwind (>= 1.0)

# ---------------------------------------------
# Firefighting settings (extra)
# ---------------------------------------------

FIREFIGHTING_ACTIVE = True          # set to True to enable firefighting
FIREFIGHTER_CELLS_PER_STEP = 3       # how many burning cells can be extinguished each timestep
COOLING_FACTOR = 0.3                 # how much the temperature is reduced in extinguished cells

# ---------------------------------------------
# Function for Visualisation
# ---------------------------------------------

def display_grid(wood_amount, burning, temperature, burned_threshold=50):
    assert wood_amount.shape == burning.shape == temperature.shape

    color_scale = ["🟦", "🟩", "🟨", "🟧", "🟥", "🟪"]

    rows, cols = wood_amount.shape

    for i in range(rows):
        row_str = ""
        for j in range(cols):
            if burning[i, j]:
                row_str += "🔥"
            elif wood_amount[i, j] > burned_threshold:
                row_str += "🌲"
            else:
                row_str += "⬛"
        row_str += "  "
        for j in range(cols):
            index = int(temperature[i, j]) // 300
            index = min(index, len(color_scale) - 1)
            row_str += color_scale[index]
        print(row_str)
    print("\n")

#------------------------------------------------
# 2. HELPER FUNCTIONS
# ------------------------------------------------

def get_neighbors(row: int, col: int, nrows: int, ncols: int) -> list[tuple[int, int]]:
    """Return list of (row, col) indices of 4-connected neighbours."""
    neighbors = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ni, nj = row + dx, col + dy
        if 0 <= ni < nrows and 0 <= nj < ncols:
            neighbors.append((ni, nj))
    return neighbors


def spread_heat(current_heat: np.ndarray) -> np.ndarray:
    """Spread heat along the grid (all cells)"""
    nrows, ncols = current_heat.shape

    old_heat = current_heat.copy()
    new_heat = current_heat

    for row in range(nrows):
        for col in range(ncols):
            neighbors = get_neighbors(row, col, nrows, ncols)
            if len(neighbors) == 0:
                continue

            # Total heat that leaves this cell
            heat_transfer_total = HEAT_SPREAD_FRACTION * old_heat[row, col]

            # Compute directional weights for neighbours (for wind)
            weights = []
            for nr, nc in neighbors:
                w = 1.0  # base weight

                if WIND_DIRECTION == "south" and nr > row:
                    w *= WIND_FACTOR
                elif WIND_DIRECTION == "north" and nr < row:
                    w *= WIND_FACTOR
                elif WIND_DIRECTION == "east" and nc > col:
                    w *= WIND_FACTOR
                elif WIND_DIRECTION == "west" and nc < col:
                    w *= WIND_FACTOR
                # if WIND_DIRECTION is "none", no scaling is applied

                weights.append(w)

            weight_sum = sum(weights)

            # Distribute heat according to weights
            for (nr, nc), w in zip(neighbors, weights):
                new_heat[nr, nc] += heat_transfer_total * (w / weight_sum)

            # Remove transferred heat from the source cell
            new_heat[row, col] -= heat_transfer_total

    return new_heat


def update_burning_cell(wood_grid, temp_grid, burning_grid, row, col):
    """Update wood, temperature, and burning state for one burning cell."""
    # amount of wood before burning
    wood_before = wood_grid[row, col]

    if wood_before <= 0:
        # no fuel, should not be burning
        burning_grid[row, col] = False
        return

    # burn 20% of remaining wood
    burned_wood = WOOD_BURN_FRACTION * wood_before
    wood_after = wood_before - burned_wood

    # increase temperature based on burned wood
    temp_before = temp_grid[row, col]
    temp_after = temp_before + burned_wood * TEMP_INCREASE_PER_KG

    # heat loss to atmosphere
    temp_after = temp_after * (1 - HEAT_LOSS_FRACTION)

    # update grids
    wood_grid[row, col] = wood_after
    temp_grid[row, col] = temp_after

    # extinguish if too little wood
    if wood_after < EXTINGUISH_WOOD:
        burning_grid[row, col] = False


def ignite_from_temperature(wood_grid, temp_grid, burning_grid):
    """Ignite cells where temperature exceeds ignition threshold and there is fuel."""
    nrows, ncols = wood_grid.shape
    for row in range(nrows):
        for col in range(ncols):
            if (temp_grid[row, col] > IGNITION_TEMP and
                wood_grid[row, col] > BURNED_THRESHOLD and
                not burning_grid[row, col]):
                burning_grid[row, col] = True


def apply_firebreak(wood_grid, burning_grid):
    """Create a horizontal firebreak in the middle of the grid (no fuel, no burning)."""
    row_break = nrows // 2
    wood_grid[row_break, :] = 0.0
    burning_grid[row_break, :] = False


def firefight_hottest_cells(wood_grid, temp_grid, burning_grid,
                            max_cells: int = FIREFIGHTER_CELLS_PER_STEP):
    """Firefighting team: extinguish up to max_cells hottest burning cells."""
    burning_indices = np.argwhere(burning_grid)
    if burning_indices.size == 0:
        return

    # Temperatures of all burning cells
    temps = np.array([temp_grid[r, c] for r, c in burning_indices])

    # Indices of burning cells sorted by temperature (hot to cold)
    order = np.argsort(temps)[::-1]

    # Extinguish up to max_cells hottest cells
    for idx in order[:max_cells]:
        r, c = burning_indices[idx]
        burning_grid[r, c] = False
        temp_grid[r, c] *= COOLING_FACTOR   # cool these cells strongly


# ------------------------------------------------
# 3. INITIALISE GRIDS
# ------------------------------------------------

wood = np.random.uniform(
    INITIAL_WOOD_MIN, INITIAL_WOOD_MAX, size=(nrows, ncols)
)
temperature = np.full((nrows, ncols), INITIAL_TEMPERATURE, dtype=float)
burning = np.zeros((nrows, ncols), dtype=bool)

# apply firebreak- left as a comment, uncomment below to activate
apply_firebreak(wood, burning)

# choose ignition cell above the firebreak
ign_row = (nrows // 2) - 2
ign_col = ncols // 2
burning[ign_row, ign_col] = True

print("Initial state:")
display_grid(wood, burning, temperature)


# ------------------------------------------------
# 4. MAIN TIME LOOP
# ------------------------------------------------

for t in range(N_TIMESTEPS):
    print(f"Timestep {t + 1}/{N_TIMESTEPS}")

    # 1) Update burning cells (local fire dynamics)
    burning_indices = np.argwhere(burning)
    for r, c in burning_indices:
        update_burning_cell(wood, temperature, burning, r, c)

    # 2) Firefighting team (bonus, optional)
    if FIREFIGHTING_ACTIVE:
        firefight_hottest_cells(wood, temperature, burning)

    # 3) Spread heat across the grid (with wind bias if WIND_DIRECTION != "none")
    temperature = spread_heat(temperature)

    # 4) Ignite new cells based on high temperature
    ignite_from_temperature(wood, temperature, burning)

    # 5) Enforce firebreak 
    apply_firebreak(wood, burning)

    # 6) Visualise
    display_grid(wood, burning, temperature)
    time.sleep(SLEEP_TIME)
