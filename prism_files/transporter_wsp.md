Taxi/transporter on a 5x5 grid (constants MAX_JOBS=2, MAX_FUEL=10).

Coordinates: x = 0..4 from west to east, y = 0..4 from south to north.
NORTH: y+1, SOUTH: y-1, EAST: x+1, WEST: x-1. Driving into the border keeps the position.
Start: x=2, y=2, fuel=10, passenger=0, jobs_done=0, done=0.

Jobs are always the same: PICK_UP is enabled only at (x=0, y=4) when passenger=0, and
DROP only at (x=0, y=0) when passenger=1 -- for every job. (The guards use the constants
passenger_location_* / passenger_destination_*.) The state variables passenger_loc_x,
passenger_loc_y, passenger_dest_x and passenger_dest_y are re-drawn uniformly at random
from 16 corner combinations after every DROP, but they only appear in the rewards
section: they never influence which actions are enabled or how x, y, fuel, passenger,
jobs_done and done evolve.

Fuel: outside the gas station every action costs 1 fuel (moves, PICK_UP, DROP).
Gas station at (x=1, y=2): there only NORTH, EAST, SOUTH and WEST are enabled; they move
normally and set fuel to 10, so leaving the station costs nothing. Arriving at the station
with fuel 0 is fine. PICK_UP and DROP are never enabled on the station.

Out of fuel: fuel=0 outside the gas station is exactly the label "empty". There every
action sets done=1 and nothing else changes. This also happens if the final DROP uses the
last unit of fuel (jobs_done=2, fuel=0 is still labelled "empty").
Episode end: jobs_done=2 (success; no action is enabled afterwards unless fuel=0) or done=1.

Distances in moves (shortest paths on the grid): start (2,2)->(0,4): 4, (0,4)->(0,0): 4,
start->gas: 1, (0,4)->gas: 3, (0,0)->gas: 3, gas->(0,4): 3, gas->(0,0): 3.
