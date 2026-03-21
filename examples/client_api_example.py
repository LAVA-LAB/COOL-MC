from pathlib import Path

import coolmc

mc = coolmc.CoolMC()

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1: Train a DQN agent on the Transporter environment
# ─────────────────────────────────────────────────────────────────────────────
print("=== Experiment 1: Training DQN on Transporter ===")

train_job = mc.cmd(
    task="safe_training",
    project_name="transporter_dqn",
    prism_file_path="transporter.prism",
    constant_definitions="MAX_JOBS=2,MAX_FUEL=10",
    prop="Pmax=? [ F jobs_done=2 ]",
    algorithm="dqn_agent",
    num_episodes=100,
    eval_interval=10,
    max_steps=100,
    layers=2,
    neurons=64,
    lr=0.0001,
    epsilon=1.0,
    epsilon_dec=0.9999,
    epsilon_min=0.1,
    gamma=0.99,
    batch_size=32,
)
print(f"Submitted: {train_job}")

train_job = mc.wait(train_job.job_id)
print(f"Done:      {train_job}")
print(f"Run ID:    {train_job.mlflow_run_id}")
print(mc.get_logs(train_job.job_id))

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1: Verify the trained policy
# ─────────────────────────────────────────────────────────────────────────────
print("=== Experiment 1: Verifying Transporter policy ===")

verify_job = mc.cmd(
    task="rl_model_checking",
    project_name="transporter_dqn",
    parent_run_id=train_job.mlflow_run_id,
    prism_file_path="transporter.prism",
    constant_definitions="MAX_JOBS=2,MAX_FUEL=10",
    prop="P=? [ F jobs_done=2 ]",
)
print(f"Submitted: {verify_job}")

verify_job = mc.wait(verify_job.job_id)
print(f"Done:      {verify_job}")
print(mc.get_logs(verify_job.job_id))

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2: Train a Behavioral Cloning agent on the Frozen Lake environment
# ─────────────────────────────────────────────────────────────────────────────
print("=== Experiment 2: Training DQN on Frozen Lake ===")

bc_job = mc.cmd(
    task="safe_training",
    project_name="frozen_lake_dqn",
    prism_file_path="frozen_lake.prism",
    constant_definitions="start_position=0,control=0.333",
    prop="Pmax=? [ F AT_FRISBEE=true ]",
    algorithm="dqn_agent",
    reward_flag=1,
    num_episodes=100,
    eval_interval=10,
    max_steps=100,
    layers=2,
    neurons=64,
    lr=0.0001,
    epsilon=0.5,
    epsilon_dec=0.9999,
    epsilon_min=0.01,
    gamma=0.99,
    batch_size=32,
)
print(f"Submitted: {bc_job}")

bc_job = mc.wait(bc_job.job_id)
print(f"Done:      {bc_job}")
print(f"Run ID:    {bc_job.mlflow_run_id}")
print(mc.get_logs(bc_job.job_id))

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2: Verify the BC policy
# ─────────────────────────────────────────────────────────────────────────────
print("=== Experiment 2: Verifying Frozen Lake BC policy ===")

bc_verify_job = mc.cmd(
    task="rl_model_checking",
    project_name="frozen_lake_dqn",
    parent_run_id=bc_job.mlflow_run_id,
    prism_file_path="frozen_lake.prism",
    constant_definitions="start_position=0,control=0.333",
    prop="P=? [ F AT_FRISBEE=true ]",
)
print(f"Submitted: {bc_verify_job}")

bc_verify_job = mc.wait(bc_verify_job.job_id)
print(f"Done:      {bc_verify_job}")
print(mc.get_logs(bc_verify_job.job_id))

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 3: Upload avoid.prism as dummy.prism, train and verify
# ─────────────────────────────────────────────────────────────────────────────
print("=== Experiment 3: Uploading dummy.prism (copy of avoid.prism) ===")

mc.upload_prism("dummy.prism", dest_name="dummy.prism")
print("Uploaded dummy.prism")

print("=== Experiment 3: Training DQN on dummy.prism ===")

dummy_job = mc.cmd(
    task="safe_training",
    project_name="dummy_dqn",
    prism_file_path="/workspaces/coolmc/prism_files_user/dummy.prism",
    constant_definitions="xMax=3,yMax=3,slickness=0.1",
    prop='Pmin=? [ F "collide" ]',
    algorithm="dqn_agent",
    reward_flag=1,
    num_episodes=100,
    eval_interval=10,
    max_steps=100,
    layers=2,
    neurons=64,
    lr=0.0001,
    epsilon=1.0,
    epsilon_dec=0.9999,
    epsilon_min=0.1,
    gamma=0.99,
    batch_size=32,
)
print(f"Submitted: {dummy_job}")

dummy_job = mc.wait(dummy_job.job_id)
print(f"Done:      {dummy_job}")
print(f"Run ID:    {dummy_job.mlflow_run_id}")
print(mc.get_logs(dummy_job.job_id))

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 3: Verify the dummy policy
# ─────────────────────────────────────────────────────────────────────────────
print("=== Experiment 3: Verifying dummy.prism policy ===")

dummy_verify_job = mc.cmd(
    task="rl_model_checking",
    project_name="dummy_dqn",
    parent_run_id=dummy_job.mlflow_run_id,
    prism_file_path="/workspaces/coolmc/prism_files_user/dummy.prism",
    constant_definitions="xMax=3,yMax=3,slickness=0.1",
    prop='P=? [ F "collide" ]',
)
print(f"Submitted: {dummy_verify_job}")

dummy_verify_job = mc.wait(dummy_verify_job.job_id)
print(f"Done:      {dummy_verify_job}")
print(mc.get_logs(dummy_verify_job.job_id))

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 4: Upload compressed_dummy.prism + subfolder, train and verify
# ─────────────────────────────────────────────────────────────────────────────
print("=== Experiment 4: Uploading compressed_dummy bundle ===")

# Upload the .prism file together with its mandatory companion subfolder.
cd_paths = mc.upload_prism_bundle("compressed_dummy.prism", "compressed_dummy/")
cd_prism = "/workspaces/coolmc/" + cd_paths[0]
print(f"Uploaded: {cd_paths}")

print("=== Experiment 4: Training DQN on compressed_dummy.prism ===")

cd_job = mc.cmd(
    task="safe_training",
    project_name="compressed_dummy_dqn",
    prism_file_path=cd_prism,
    prop='Pmax=? [ F "goal" ]',
    algorithm="dqn_agent",
    reward_flag=1,
    num_episodes=100,
    eval_interval=10,
    max_steps=100,
    layers=2,
    neurons=64,
    lr=0.0001,
    epsilon=1.0,
    epsilon_dec=0.9999,
    epsilon_min=0.1,
    gamma=0.99,
    batch_size=32,
)
print(f"Submitted: {cd_job}")

cd_job = mc.wait(cd_job.job_id)
print(f"Done:      {cd_job}")
print(f"Run ID:    {cd_job.mlflow_run_id}")
print(mc.get_logs(cd_job.job_id))

# ─────────────────────────────────────────────────────────────────────────────
# Experiment 4: Verify the compressed_dummy policy
# ─────────────────────────────────────────────────────────────────────────────
print("=== Experiment 4: Verifying compressed_dummy.prism policy ===")

cd_verify_job = mc.cmd(
    task="rl_model_checking",
    project_name="compressed_dummy_dqn",
    parent_run_id=cd_job.mlflow_run_id,
    prism_file_path=cd_prism,
    prop='P=? [ F "goal" ]',
)
print(f"Submitted: {cd_verify_job}")

cd_verify_job = mc.wait(cd_verify_job.job_id)
print(f"Done:      {cd_verify_job}")
print(mc.get_logs(cd_verify_job.job_id))

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("=== Summary ===")
for job in mc.list_jobs():
    print(job)

# ─────────────────────────────────────────────────────────────────────────────
# Test: server_info
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Test: server_info ===")
info = mc.server_info()
for k, v in info.items():
    print(f"  {k}: {v}")

# ─────────────────────────────────────────────────────────────────────────────
# Test: get_result — extract final property result from completed jobs
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Test: get_result ===")
for job in [train_job, verify_job, bc_job, bc_verify_job, dummy_job, dummy_verify_job, cd_job, cd_verify_job]:
    result = mc.get_result(job.job_id)
    print(f"  {job} → result: {result}")

# ─────────────────────────────────────────────────────────────────────────────
# Test: MLflow integration
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Test: list_experiments ===")
for exp in mc.list_experiments():
    print(f"  {exp}")

print("\n=== Test: list_runs ===")
runs = mc.list_runs(max_results=5)
for run in runs:
    print(f"  run_id={run['run_id'][:8]}  status={run['status']}  metrics={run['metrics']}")

print("\n=== Test: get_run ===")
run = mc.get_run(train_job.mlflow_run_id)
print(f"  params:  {run['params']}")
print(f"  metrics: {run['metrics']}")
print(f"  tags:    {run['tags']}")

print("\n=== Test: get_metric_history ===")
# Use the first available metric key from the training run
if run["metrics"]:
    metric_key = next(iter(run["metrics"]))
    history = mc.get_metric_history(train_job.mlflow_run_id, metric_key)
    print(f"  metric '{metric_key}' — {len(history)} data points")
    for entry in history[:5]:  # show first 5 steps
        print(f"    step={entry['step']}  value={entry['value']}")

# ─────────────────────────────────────────────────────────────────────────────
# Test: cancel_all — submit two jobs, then immediately cancel them
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Test: cancel_all ===")
j1 = mc.cmd(task="safe_training", project_name="cancel_test_1", prism_file_path="transporter.prism",
            constant_definitions="MAX_JOBS=2,MAX_FUEL=10", prop="Pmax=? [ F jobs_done=2 ]",
            algorithm="dqn_agent", num_episodes=10)
j2 = mc.cmd(task="safe_training", project_name="cancel_test_2", prism_file_path="transporter.prism",
            constant_definitions="MAX_JOBS=2,MAX_FUEL=10", prop="Pmax=? [ F jobs_done=2 ]",
            algorithm="dqn_agent", num_episodes=10)
print(f"  Submitted: {j1}, {j2}")
cancelled = mc.cancel_all()
print(f"  Cancelled {len(cancelled)} job(s): {cancelled}")

# ─────────────────────────────────────────────────────────────────────────────
# Test: clear_history — remove all completed jobs
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Test: clear_history ===")
before = len(mc.list_jobs())
removed = mc.clear_history()
after = len(mc.list_jobs())
print(f"  Jobs before: {before}  removed: {removed}  jobs after: {after}")

# ─────────────────────────────────────────────────────────────────────────────
# Test: stop and restart the container
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Test: stop + restart ===")
print("  Stopping container...")
mc.stop()
print("  Restarting container...")
mc.restart()
print("  Container is healthy again.")
info = mc.server_info()
print(f"  Server back up — commit: {info['cool_mc_commit']}")

# ─────────────────────────────────────────────────────────────────────────────
# Storm remote API tests
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== Storm: build_model ===")
model_info = mc.storm.build_model(
    "transporter.prism", constants="MAX_JOBS=2,MAX_FUEL=10"
)
print(f"  {model_info}")
print(f"  labels:         {model_info.labels}")
print(f"  reward_models:  {model_info.reward_models}")
print(f"  initial_states: {model_info.initial_states}")

print("\n=== Storm: check ===")
res = mc.storm.check(
    "transporter.prism",
    "Pmax=? [ F jobs_done=2 ]",
    constants="MAX_JOBS=2,MAX_FUEL=10",
)
print(f"  {res}")
print(f"  result = {res.result}")

print("\n=== Storm: check (all_states=True) ===")
res_all = mc.storm.check(
    "transporter.prism",
    "Pmax=? [ F jobs_done=2 ]",
    constants="MAX_JOBS=2,MAX_FUEL=10",
    all_states=True,
)
print(f"  Results for first 5 states: {res_all.all_states[:5]}")

print("\n=== Storm: get_scheduler ===")
sched = mc.storm.get_scheduler(
    "transporter.prism",
    "Pmax=? [ F jobs_done=2 ]",
    constants="MAX_JOBS=2,MAX_FUEL=10",
)
print(f"  {sched}")
print(f"  Action in state 0: {sched.get_action(0)}")
print(f"  Action in state 1: {sched.get_action(1)}")

print("\n=== Storm: simulate ===")
sim = mc.storm.simulate(
    "transporter.prism",
    nr_steps=10,
    constants="MAX_JOBS=2,MAX_FUEL=10",
    seed=42,
)
print(f"  {sim}")
for step in sim.path:
    print(f"    step {step['step']}: state={step['state']}  "
          f"labels={step['labels']}  action={step['action']}")

print("\n=== Storm: get_transitions ===")
trans = mc.storm.get_transitions(
    "transporter.prism", constants="MAX_JOBS=2,MAX_FUEL=10"
)
print(f"  model_type={trans['model_type']}  "
      f"nr_states={trans['nr_states']}  "
      f"nr_transitions={trans['nr_transitions']}")
print(f"  First 5 transitions:")
for t in trans["transitions"][:5]:
    print(f"    {t}")

print("\n=== Storm: build_model on user-uploaded file ===")
cd_info = mc.storm.build_model(
    "/workspaces/coolmc/prism_files_user/compressed_dummy.prism"
)
print(f"  {cd_info}")

print("\n=== Storm: check on user-uploaded file ===")
cd_res = mc.storm.check(
    "/workspaces/coolmc/prism_files_user/compressed_dummy.prism",
    'Pmax=? [ F "goal" ]',
)
print(f"  result = {cd_res.result}")
