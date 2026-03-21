"""Storm remote API — usage examples.

This file shows how to use COOL-MC's remote Storm/stormpy interface.
All model checking runs inside the Docker container where Storm 1.7.0 is
installed.  Your local machine needs nothing beyond the coolmc pip package.

Run with:
    python storm_wrapper_example.py
"""

import coolmc

mc = coolmc.CoolMC()

# ─────────────────────────────────────────────────────────────────────────────
# 1. Build a model — inspect states, transitions, and labels
# ─────────────────────────────────────────────────────────────────────────────
# Works with built-in environments (prism_files/) and user-uploaded ones.

info = mc.storm.build_model(
    "transporter.prism",
    constants="MAX_JOBS=2,MAX_FUEL=10",
)

print(f"Model type:       {info.model_type}")   # e.g. ModelType.MDP
print(f"States:           {info.nr_states}")
print(f"Transitions:      {info.nr_transitions}")
print(f"Labels:           {info.labels}")        # atomic propositions
print(f"Reward models:    {info.reward_models}")
print(f"Initial states:   {info.initial_states}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Model checking — compute the optimal reachability probability
# ─────────────────────────────────────────────────────────────────────────────
# Supports any PCTL/LTL/reward formula that Storm accepts.

res = mc.storm.check(
    "transporter.prism",
    "Pmax=? [ F jobs_done=2 ]",
    constants="MAX_JOBS=2,MAX_FUEL=10",
)

print(f"\nCheck result (initial state): {res.result}")
print(f"Initial state index:          {res.initial_state}")
print(f"Model info bundled:           {res.model}")

# Use float() if you need a plain Python number:
value = float(res)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Model checking — results for every state
# ─────────────────────────────────────────────────────────────────────────────
# all_states=True returns a list indexed by state ID.

res_all = mc.storm.check(
    "transporter.prism",
    "Pmax=? [ F jobs_done=2 ]",
    constants="MAX_JOBS=2,MAX_FUEL=10",
    all_states=True,
)

print(f"\nResults for first 5 states: {res_all.all_states[:5]}")

# Example: find all states that can reach the goal with probability > 0.5
reachable = [
    (state_id, prob)
    for state_id, prob in enumerate(res_all.all_states)
    if prob > 0.5
]
print(f"States with prob > 0.5: {reachable[:5]}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Optimal scheduler — what action to take in each state
# ─────────────────────────────────────────────────────────────────────────────
# Extracts the optimal strategy computed by Storm during model checking.

sched = mc.storm.get_scheduler(
    "transporter.prism",
    "Pmax=? [ F jobs_done=2 ]",
    constants="MAX_JOBS=2,MAX_FUEL=10",
)

print(f"\nScheduler result (initial state): {sched.result}")
print(f"Action in state 0: {sched.get_action(0)}")  # int action index
print(f"Action in state 1: {sched.get_action(1)}")

# Iterate over all scheduled choices:
for state_id, action in list(sched.choices.items())[:5]:
    print(f"  state {state_id} → action {action}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Simulation — run a single execution trace through the model
# ─────────────────────────────────────────────────────────────────────────────
# The simulator steps through the model, picking the first available action.
# Set seed for reproducible traces.

sim = mc.storm.simulate(
    "transporter.prism",
    nr_steps=15,
    constants="MAX_JOBS=2,MAX_FUEL=10",
    seed=42,
)

print(f"\nSimulation: {sim}")
print(f"Final state: {sim.final_state}  done: {sim.is_done}")
for step in sim.path:
    print(f"  step {step['step']:>2}: action={step['action']}  state={step['state']}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Transition matrix — inspect the sparse matrix directly
# ─────────────────────────────────────────────────────────────────────────────
# Each entry: {"from": state_id, "action": action_id, "to": state_id, "probability": float}

trans = mc.storm.get_transitions(
    "transporter.prism",
    constants="MAX_JOBS=2,MAX_FUEL=10",
)

print(f"\nTransition matrix — {trans['nr_states']} states, "
      f"{trans['nr_transitions']} transitions")

# Example: find all successors of state 0
from_zero = [t for t in trans["transitions"] if t["from"] == 0]
print(f"Transitions out of state 0:")
for t in from_zero:
    print(f"  action {t['action']} → state {t['to']} with prob {t['probability']}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. User-uploaded PRISM files — same API, just pass the absolute path
# ─────────────────────────────────────────────────────────────────────────────
# First upload your file (only needed once per container lifecycle):

mc.upload_prism("dummy.prism", dest_name="dummy.prism")

# Then use it with any Storm method:
user_res = mc.storm.check(
    "/workspaces/coolmc/prism_files_user/dummy.prism",
    'Pmin=? [ F "collide" ]',
    constants="xMax=3,yMax=3,slickness=0.1",
)
print(f"\nUser file check result: {user_res.result}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Parametric model checking — result is a rational function
# ─────────────────────────────────────────────────────────────────────────────
# Use with PRISM models that contain uninstantiated probability parameters.
# Storm returns the exact rational function over those parameters.

# (Commented out — requires a parametric PRISM file, e.g. one where a
#  transition probability is a free variable like 'p' instead of a number.)

# pres = mc.storm.parametric_check(
#     "my_parametric_model.prism",
#     "P=? [ F target ]",
# )
# print(f"\nParametric result: {pres.result}")   # e.g. "p/(1 - p + p**2)"
# print(f"Free parameters:   {pres.parameters}") # e.g. ["p"]

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\nDone. All Storm queries ran inside Docker — no local Storm needed.")
