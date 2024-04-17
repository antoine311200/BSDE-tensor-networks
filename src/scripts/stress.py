from bsde_solver import xp
import time

def stress_test_tile(shape, reps, seed=45):
    # Generate a random array
    xp.random.seed(seed)
    arr = xp.random.rand(*shape)

    # Measure the time taken for xp.tile
    start_time = time.time()
    tiled_arr = xp.tile(arr, (reps, 1))
    tile_time = time.time() - start_time

    return tile_time, tiled_arr

def stress_test_broadcast_to(shape, reps, seed=45):
    # Generate a random array
    xp.random.seed(seed)
    arr = xp.random.rand(*shape)

    # Measure the time taken for xp.broadcast_to
    start_time = time.time()
    broadcasted_arr = xp.broadcast_to(arr, (reps,) + shape)
    broadcast_time = time.time() - start_time

    return broadcast_time, broadcasted_arr

# Define the shape of the array
shape = (10000, )

# Define the number of repetitions
reps = 100000

# Run stress tests
tile_time, arr_tile = stress_test_tile(shape, reps)
broadcast_time, arr_broadcast = stress_test_broadcast_to(shape, reps)

# Print the results
print(f"Time taken for xp.tile: {tile_time:.5f} seconds")
print(f"Time taken for xp.broadcast_to: {broadcast_time:.5f} seconds")

# Compare the results
print(f"Are the arrays equal? {xp.array_equal(arr_tile, arr_broadcast)}")