"""Weight-space programming (WSP) agent.

An LLM writes the weights of a small ReLU MLP policy *directly* -- no
gradient descent anywhere. The pipeline hooks are used as follows:

  1. ``load_env``          -> initial creation. The LLM receives the PRISM
                              model, constants, property, feature/action
                              layout and (optionally) an extra info file and
                              writes all weights from scratch.
  2. ``behavioral_cloning`` -> optional dataset-driven repair. Each "epoch"
                              is one LLM round: measure accuracy, send the
                              misclassified states back, apply the edit. The
                              best weights (by accuracy) are kept.
  3. ``model_checking_learn`` -> verification-driven repair. Storm checks the
                              induced DTMC; the LLM gets the result, the
                              policy's decisions on the reachable states and
                              one simulated episode, and proposes an edit. The
                              edit is applied *lazily* (in ``episodic_learn``)
                              so that a ``project.save()`` right after model
                              checking always stores the weights Storm just
                              verified. No repair is requested after the last
                              evaluation (it could never be verified).

Every answer must also contain "checks": states with the action the LLM
intends there. The agent evaluates the written network on them; mismatches
(arithmetic errors, transposed matrices, ...) are sent back for a retry
(``self_check=retry``, default) or only recorded (``self_check=report``).

Configured via a ';'-separated string in ``--algorithm``:

    weight_space_programming_agent;model=deepseek-v4-pro:cloud;\
    host=http://localhost:11434;info_file=../prism_files/transporter_wsp.md;\
    temperature=0.2;timeout=900;max_feedback_states=200;retries=2;\
    self_check=retry;rollout_steps=60;debug=1;export_dir=/abs/path

``export_dir`` (optional) receives JSON files that are rewritten after every
event, so they are complete even if the run stops early: wsp_step1.json
(network after creation), wsp_step2.json (network after the dataset repair),
wsp_best.json (best network verified by Storm), wsp_mc.json (all model-checking
results), wsp_calls.json (one record per LLM request with its outcome) and
wsp_history.json.

Architecture comes from ``--layers`` / ``--neurons``. The network is a
``BCNetwork``, so saved checkpoints (``bc_network.chkpt``) are also loadable
by ``bc_nn_agent``.

LLM access goes through the ``ollama`` Python package. Without ``host`` the
agent uses $OLLAMA_HOST or the local daemon (http://localhost:11434), which
serves ``*:cloud`` models once it is signed in. Ollama Cloud directly:
export OLLAMA_API_KEY and use host=https://ollama.com (the package sends the
key itself; ``:cloud`` suffixes are stripped for the direct API).

NOTE: the network sees the state *after* cool-mc preprocessors. If you use
e.g. a normalizer, describe the transformed input in the ``info_file``.
"""
import json
import math
import os
import re
import shutil
import time
from typing import Dict, List, Optional, Tuple

import mlflow
import numpy as np
import torch

from common.agents.agent import Agent
from common.agents.bc_nn_agent import BCNetwork
from common.utilities.helper import parse_prop_type

try:
    from ollama import Client, ResponseError
except ImportError:  # only required when the agent is actually used; fails loudly in _get_client
    Client, ResponseError = None, None


SYSTEM_PROMPT = """You are an expert in neural networks, Markov decision processes and the PRISM modelling language.
You practice *weight-space programming*: you write the weights of a neural network policy directly, by reasoning about what each neuron should compute. There is no training loop and no gradient descent.

Network semantics (PyTorch convention):
  x      = state vector (feature order given below)
  h_1    = relu(W_0 x + b_0)
  h_k+1  = relu(W_k h_k + b_k)
  logits = W_L h_L + b_L           (last layer has NO relu)
  action = argmax(logits)          (ties -> lowest index)
Each W_k has shape [out_features][in_features]: one row per output neuron, one column per input.

Useful constructions for integer-valued inputs:
  step(z) = relu(z+1) - relu(z) is exactly 1 if z >= 0 and 0 if z <= -1 (z integer).
  bump at k: relu(v-k+1) - 2*relu(v-k) + relu(v-k-1) is 1 if v == k and 0 for every other integer v.
  AND of 0/1 indicators a_1..a_n: relu(a_1 + ... + a_n - (n-1)).
Since indicators built in one layer are linear combinations of that layer's ReLU outputs, the NEXT layer can combine them.
Make the intended logit win by a clear margin (>= 1.0) and only choose actions that are enabled in the respective state.

Answer with ONE JSON object and nothing else:
{"notes": "<short explanation of what the neurons compute and the strategy>",
 "layers": [{"W": [[...], ...], "b": [...]}, ...],
 "checks": [{"state": {"<feature name>": <value>, ...}, "action": "<action label>"}, ...]}
"checks" must list 5-20 representative states (covering every branch of your strategy) with the action YOUR network should choose there. They are evaluated on the network you wrote."""


def _parse_config(config_str: str) -> dict:
    cfg = {}
    for part in config_str.split(";")[1:]:
        key, _, value = part.partition("=")
        key, value = key.strip(), value.strip()
        if key:
            cfg[key] = value
    return cfg


class WeightSpaceProgrammingAgent(Agent):

    def __init__(self, config_str: str, state_dimension: int, number_of_neurons: List[int],
                 number_of_actions: int, action_names: List[str], prop: str = "",
                 constant_definitions: str = "", training_threshold: Optional[float] = None,
                 deploy: bool = False, reward_flag: bool = False, wrong_action_penalty: float = 0,
                 max_steps: Optional[int] = None, num_episodes: Optional[int] = None,
                 eval_interval: Optional[int] = None):
        super().__init__()
        cfg = _parse_config(config_str)
        self.model = cfg.pop("model", "deepseek-v4-pro:cloud")
        self.host = cfg.pop("host", os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.info_file = cfg.pop("info_file", "")
        self.temperature = float(cfg.pop("temperature", 0.2))
        self.timeout = float(cfg.pop("timeout", 900))
        self.seed = int(cfg["seed"]) if "seed" in cfg else None
        cfg.pop("seed", None)
        self.think = cfg.pop("think", None)  # "true"/"false"/"low"/"medium"/"high"
        self.max_feedback_states = int(cfg.pop("max_feedback_states", 200))
        self.retries = int(cfg.pop("retries", 2))
        self.self_check = cfg.pop("self_check", "retry")
        if self.self_check not in ("retry", "report", "off"):
            raise ValueError("self_check must be retry, report or off")
        self.rollout_steps = int(cfg.pop("rollout_steps", 60))
        self.debug = int(cfg.pop("debug", 1))
        self.export_dir = cfg.pop("export_dir", "")
        for k in cfg:
            print(f"Warning: unknown WeightSpaceProgrammingAgent parameter '{k}'")

        self.state_dimension = state_dimension
        self.number_of_neurons = list(number_of_neurons)
        self.number_of_actions = number_of_actions
        self.action_names = list(action_names)
        self.prop = prop or ""
        self.prop_type = parse_prop_type(self.prop) if self.prop else "reward"
        self.constant_definitions = constant_definitions or ""
        self.training_threshold = training_threshold
        self.deploy = deploy
        self.reward_flag = bool(reward_flag)
        self.wrong_action_penalty = wrong_action_penalty
        self.max_steps = max_steps
        # Number of model-checking evaluations training.py will run (episodes 0, k, 2k, ... < num_episodes).
        self.total_evaluations = (math.ceil(num_episodes / eval_interval)
                                  if num_episodes and eval_interval else None)

        self.network = BCNetwork(state_dimension, self.number_of_neurons, number_of_actions, lr=1e-3)
        self.shapes = self._layer_shapes()
        n_params = sum(o * i + o for o, i in self.shapes)
        if n_params > 5000:
            print(f"Warning: WSP network has {n_params} parameters. Weight-space programming "
                  f"is only reported to work on small networks.")

        self.extra_info = ""
        if self.info_file:
            with open(self.info_file, "r") as f:
                self.extra_info = f.read()

        self.env = None
        self.enabled: Optional[Dict[tuple, List[str]]] = None  # state vector -> enabled action labels
        self.feature_names: List[str] = []
        self.prism_source = ""
        self.notes = ""                 # LLM's own explanation of the current weights
        self.history: List[dict] = []   # log of rounds (shown to the LLM, saved as artifact)
        self.loaded_from_checkpoint = False
        self.pending_weights = None     # edit proposed by model_checking_learn, applied lazily
        self.visited = {}               # state tuple -> action, recorded during model checking
        self.best_mc_result = None
        self.best_mc_weights = None     # (layers, notes) of the best network Storm has verified
        self.evaluations = 0
        self.last_check = ""            # self-check summary of the last accepted answer
        self.calls: List[dict] = []     # one record per LLM request (exported for the experiments)
        self.mc_results: List[float] = []
        self._client = None

    # ------------------------------------------------------------------ #
    # LLM plumbing
    # ------------------------------------------------------------------ #
    def _get_client(self):
        if self._client is None:
            if Client is None:
                raise ImportError("WeightSpaceProgrammingAgent needs the 'ollama' package (pip install ollama).")
            # The ollama package adds 'Authorization: Bearer $OLLAMA_API_KEY' itself.
            self._client = Client(host=self.host, timeout=self.timeout)
        return self._client

    def _chat(self, user_prompt: str) -> str:
        options = {"temperature": self.temperature}
        if self.seed is not None:
            options["seed"] = self.seed
        kwargs = {}
        if self.think is not None:
            kwargs["think"] = {"true": True, "false": False}.get(self.think.lower(), self.think)
        model = self.model
        if "ollama.com" in self.host and model.endswith(":cloud"):
            # Direct cloud API uses plain names (the ':cloud' suffix is for the local daemon).
            model = model[: -len(":cloud")]
        response = self._get_client().chat(
            model=model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": user_prompt}],
            format="json",
            options=options,
            **kwargs,
        )
        return response.message.content or ""

    def _fatal_error_message(self, exc: Exception) -> Optional[str]:
        """Return a helpful message for setup errors that retrying cannot fix, else None."""
        where = f"(model={self.model}, host={self.host})"
        if isinstance(exc, ImportError):
            return f"[WSP] {exc}"
        if isinstance(exc, ConnectionError):
            return (f"[WSP] Cannot reach Ollama {where}: {exc}\n"
                    "Set host=... in the algorithm string or export OLLAMA_HOST.")
        status = getattr(exc, "status_code", None)
        if ResponseError is not None and isinstance(exc, ResponseError) and status in (401, 403):
            key = "set" if os.environ.get("OLLAMA_API_KEY") else "NOT set"
            return (f"[WSP] Ollama refused the request {where}: {exc} (OLLAMA_API_KEY is {key}).\n"
                    "For host=https://ollama.com export a valid OLLAMA_API_KEY "
                    "(https://ollama.com/settings/keys); for a local daemon, sign it in.")
        if ResponseError is not None and isinstance(exc, ResponseError) and status == 404:
            return (f"[WSP] Model not found {where}: {exc}\n"
                    "Local daemon: use the ':cloud' name (e.g. deepseek-v4-pro:cloud). "
                    "Direct API (https://ollama.com): use the plain name.")
        return None

    def _request_weights(self, user_prompt: str, purpose: str):
        """Ask the LLM for weights; retry with the validation / self-check error on failure.
        Returns (list_of_(W, b)_arrays, notes) or (None, None). Sets self.last_check."""
        prompt = user_prompt
        fallback = None  # last structurally valid answer that failed only the self-check
        for attempt in range(self.retries + 1):
            if self.debug:
                print(f"[WSP] {purpose}: querying {self.model} (attempt {attempt + 1}, "
                      f"prompt {len(prompt)} chars)", flush=True)
            started = time.time()
            try:
                text = self._chat(prompt)
            except Exception as exc:
                self._log_call(purpose, attempt, "request_failed", started, f"{type(exc).__name__}: {exc}")
                fatal = self._fatal_error_message(exc)
                if fatal is not None:
                    # Setup problems (package, connection, auth, model name) never fix
                    # themselves -- stop instead of silently running with random weights.
                    raise RuntimeError(fatal) from exc
                print(f"[WSP] LLM request failed (will retry): {type(exc).__name__}: {exc}", flush=True)
                continue
            try:
                layers, notes, checks = self._parse_weights(text)
            except ValueError as exc:
                self._log_call(purpose, attempt, self._error_kind(exc), started, str(exc))
                print(f"[WSP] Invalid answer: {exc}", flush=True)
                if self.debug > 1:
                    print(text[:3000])
                # Keep the last valid-but-wrong answer in view; add why this answer was unusable.
                prompt = (user_prompt + self._rejected_section(fallback)
                          + "\n\n## Your last answer was rejected (invalid)\n"
                          f"{exc}\nReturn a corrected, complete JSON object with exactly the required shapes.")
                continue

            n_checks, problems = self._run_self_check(layers, checks)
            summary = (f"self-check {n_checks - len(problems)}/{n_checks} states as intended"
                       if n_checks else "no self-check states given")
            if self.debug:
                print(f"[WSP] {purpose}: valid weights, {summary}. Notes: {notes[:400]}", flush=True)
                for p in problems[:10]:
                    print(f"[WSP]   self-check mismatch: {p}", flush=True)
            if problems and self.self_check == "retry" and attempt < self.retries:
                self._log_call(purpose, attempt, "self_check_failed", started, summary)
                fallback = (layers, notes, summary, problems)
                prompt = user_prompt + self._rejected_section(fallback)
                continue
            self._log_call(purpose, attempt, "accepted_despite_failure" if problems else "accepted",
                           started, summary)
            self.last_check = summary
            return layers, notes
        if fallback is not None:
            print("[WSP] Accepting the last answer despite self-check mismatches (no retries left).", flush=True)
            # Not an LLM request: the earlier answer (already logged as self_check_failed) is used.
            self.calls.append({"purpose": purpose, "attempt": None, "outcome": "fallback_accepted",
                               "seconds": 0.0, "detail": fallback[2]})
            self._export("wsp_calls.json", self.calls)
            self.last_check = fallback[2]
            return fallback[0], fallback[1]
        return None, None

    # ------------------------------------------------------------------ #
    # Experiment logging (only active with export_dir)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _error_kind(exc: Exception) -> str:
        """Classify a rejected answer: unreadable JSON vs. readable JSON with a wrong network."""
        msg = str(exc)
        if msg.startswith(("no JSON object", "answer is not valid JSON", "JSON object needs")):
            return "invalid_json"
        return "wrong_shape"   # wrong layer count / shapes, unreadable W or b, NaN/inf

    def _log_call(self, purpose: str, attempt: int, outcome: str, started: float, detail: str = ""):
        self.calls.append({"purpose": purpose, "attempt": attempt, "outcome": outcome,
                           "seconds": round(time.time() - started, 2), "detail": detail[:500]})
        self._export("wsp_calls.json", self.calls)

    def _export(self, name: str, obj):
        if not self.export_dir:
            return
        os.makedirs(self.export_dir, exist_ok=True)
        path = os.path.join(self.export_dir, name)
        with open(path + ".tmp", "w") as f:
            json.dump(obj, f, indent=1)
        os.replace(path + ".tmp", path)   # atomic: a crash never leaves a half-written file

    def _export_network(self, name: str, extra: Optional[dict] = None):
        if self.export_dir:
            self._export(name, {"notes": self.notes, **(extra or {}),
                                **json.loads(self._weights_json(decimals=None))})

    def _rejected_section(self, fallback) -> str:
        """The last structurally valid answer that failed the self-check, so the LLM can fix it."""
        if fallback is None:
            return ""
        layers, notes, _, problems = fallback
        return ("\n\n## Your previous answer was rejected by the self-check\n"
                "These are the weights you wrote:\n" + self._layers_json(layers)
                + (f"\nYour notes on them: {notes}" if notes else "")
                + "\nThis network does NOT compute the actions you intended in these states:\n"
                + "\n".join(f"- {p}" for p in problems[:30])
                + "\nDo not start over: find the neurons and weights responsible and correct them "
                  "(check the [out][in] orientation and your arithmetic). Return the complete JSON "
                  "object again, including checks.")

    def _parse_weights(self, text: str):
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match is None:
                raise ValueError("no JSON object found in the answer")
            try:
                obj = json.loads(match.group(0))
            except json.JSONDecodeError as exc:
                raise ValueError(f"answer is not valid JSON: {exc}")
        if not isinstance(obj, dict) or not isinstance(obj.get("layers"), list):
            raise ValueError("JSON object needs a 'layers' list")
        raw = obj["layers"]
        if len(raw) != len(self.shapes):
            raise ValueError(f"expected {len(self.shapes)} layers, got {len(raw)}")
        layers = []
        for k, ((out_f, in_f), layer) in enumerate(zip(self.shapes, raw)):
            try:
                W = np.asarray(layer["W"], dtype=np.float32)
                b = np.asarray(layer["b"], dtype=np.float32)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"layer {k}: cannot read W/b ({exc})")
            if W.shape != (out_f, in_f):
                raise ValueError(f"layer {k}: W must have shape [{out_f}][{in_f}], got {list(W.shape)}")
            if b.shape != (out_f,):
                raise ValueError(f"layer {k}: b must have length {out_f}, got shape {list(b.shape)}")
            if not (np.isfinite(W).all() and np.isfinite(b).all()):
                raise ValueError(f"layer {k}: contains NaN/inf")
            layers.append((W, b))
        checks = obj.get("checks", [])
        return layers, str(obj.get("notes", "")), checks if isinstance(checks, list) else []

    # ------------------------------------------------------------------ #
    # Self-check: does the written network compute what the LLM intended?
    # ------------------------------------------------------------------ #
    @staticmethod
    def _forward_np(layers, x):
        h = np.asarray(x, dtype=np.float32)
        for k, (W, b) in enumerate(layers):
            h = W @ h + b
            if k < len(layers) - 1:
                h = np.maximum(h, 0.0)
        return h

    def _run_self_check(self, layers, checks) -> Tuple[int, List[str]]:
        if self.self_check == "off" or not checks:
            return 0, []
        problems = []
        checks = checks[:30]
        for i, c in enumerate(checks):
            try:
                st, intended = c["state"], c["action"]
                if isinstance(st, dict):
                    missing = [n for n in self.feature_names if n not in st]
                    if missing:
                        problems.append(f"check {i}: state lacks features {missing}")
                        continue
                    x = [float(st[n]) for n in self.feature_names]
                else:
                    x = [float(v) for v in st]
                    if len(x) != self.state_dimension:
                        problems.append(f"check {i}: state needs {self.state_dimension} values")
                        continue
            except (KeyError, TypeError, ValueError) as exc:
                problems.append(f"check {i}: unreadable ({exc})")
                continue
            if intended not in self.action_names:
                problems.append(f"check {i}: unknown action '{intended}'")
                continue
            logits = self._forward_np(layers, x)
            got = self.action_names[int(np.argmax(logits))]
            if got != intended:
                lg = ", ".join(f"{a}={v:.3g}" for a, v in zip(self.action_names, logits))
                problems.append(f"{self._state_str(x)}: you intended {intended}, the network outputs {got} "
                                f"(logits {lg})")
        return len(checks), problems

    # ------------------------------------------------------------------ #
    # Network <-> arrays
    # ------------------------------------------------------------------ #
    def _linear_layers(self):
        return list(self.network.layers)

    def _layer_shapes(self):
        return [(l.out_features, l.in_features) for l in self._linear_layers()]

    def get_weights(self):
        return [(l.weight.detach().cpu().numpy().copy(), l.bias.detach().cpu().numpy().copy())
                for l in self._linear_layers()]

    def set_weights(self, layers):
        with torch.no_grad():
            for lin, (W, b) in zip(self._linear_layers(), layers):
                lin.weight.copy_(torch.as_tensor(np.asarray(W), dtype=lin.weight.dtype))
                lin.bias.copy_(torch.as_tensor(np.asarray(b), dtype=lin.bias.dtype))

    @staticmethod
    def _layers_json(layers, decimals: Optional[int] = 6) -> str:
        def conv(a):
            a = np.asarray(a, dtype=np.float64)
            return (np.round(a, decimals) if decimals is not None else a).tolist()
        return json.dumps({"layers": [{"W": conv(W), "b": conv(b)} for W, b in layers]})

    def _weights_json(self, decimals: Optional[int] = 6) -> str:
        return self._layers_json(self.get_weights(), decimals)

    def _best_section(self, title: str, best) -> str:
        """An earlier, better version of the network (weights, notes) to return to or learn from."""
        if best is None:
            return ""
        layers, notes = best
        return (f"\n\n## {title}\n" + self._layers_json(layers)
                + (f"\nYour notes on it: {notes}" if notes else "")
                + "\nThe current network is worse than this one. You may continue from the current network, "
                  "go back to this one, or combine ideas from both.")

    # ------------------------------------------------------------------ #
    # Environment knowledge: enabled actions per state
    # ------------------------------------------------------------------ #
    def _build_enabled_table(self, env) -> Optional[Dict[tuple, List[str]]]:
        """Map every reachable MDP state (as the network's input vector) to its enabled action
        labels, built with Storm exactly like cool-mc's datasets build their states."""
        try:
            import stormpy
            prog = stormpy.parse_prism_program(env.storm_bridge.path)
            prog = stormpy.preprocess_symbolic_input(prog, [], self.constant_definitions)[0].as_prism_program()
            options = stormpy.BuilderOptions(True, True)
            options.set_build_state_valuations()
            options.set_build_choice_labels()
            model = stormpy.build_sparse_model_with_options(prog, options)
            keys = list(json.loads(env.storm_bridge.state_json_example).keys())
            sm = env.storm_bridge.state_mapper
            table: Dict[tuple, set] = {}
            for sid in range(model.nr_states):
                val = json.loads(str(model.state_valuations.get_json(sid)))
                arr = np.array([int(val[k]) for k in keys if k in val], dtype=np.int32)
                vec = tuple(float(v) for v in sm.map(arr).tolist())
                labels = set()
                for a in range(model.get_nr_available_actions(sid)):
                    labels |= set(model.choice_labeling.get_labels_of_choice(model.get_choice_index(sid, a)))
                table.setdefault(vec, set()).update(l for l in labels if l in self.action_names)
            return {k: sorted(v) for k, v in table.items()}
        except Exception as exc:
            print(f"[WSP] Warning: could not build the enabled-action table ({type(exc).__name__}: {exc}); "
                  "feedback will not show fallback actions.", flush=True)
            return None

    def _key(self, state) -> tuple:
        return tuple(float(v) for v in np.asarray(state).reshape(-1).tolist())

    def _enabled_at(self, state) -> Optional[List[str]]:
        if self.enabled is None:
            return None
        return self.enabled.get(self._key(state))

    def _action_str(self, state, action_idx: int) -> str:
        """Chosen action, annotated with what cool-mc really executes if it is disabled."""
        name = self.action_names[action_idx]
        enabled = self._enabled_at(state)
        if enabled is None or name in enabled:
            return name
        if not enabled:
            return f"{name} (terminal state, nothing is executed)"
        return f"{name} (NOT enabled here -> cool-mc executes {enabled[0]})"

    def _rollout_trace(self) -> str:
        """Simulate one episode with the current network (deterministic argmax)."""
        if self.env is None or self.rollout_steps <= 0:
            return ""
        lines = []
        try:
            state = self.env.reset()
            for t in range(self.rollout_steps):
                s = [state] if isinstance(state, (int, np.integer)) else state
                a = self.select_action(np.asarray(s), deploy=False)
                desc = self._action_str(s, a)
                state, reward, done, truncated, _ = self.env.step(a)
                lines.append(f"{t}: {self._state_str(s)} -> {desc} | reward {self._num(reward)}")
                if done:
                    ns = [state] if isinstance(state, (int, np.integer)) else state
                    lines.append(f"episode ended after {t + 1} steps"
                                 + (" (step limit)" if truncated else "") + f"; final state: {self._state_str(ns)}")
                    break
            else:
                lines.append(f"(trace cut after {self.rollout_steps} steps)")
        except Exception as exc:
            lines.append(f"(simulation failed: {type(exc).__name__}: {exc})")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Prompt building
    # ------------------------------------------------------------------ #
    def _state_str(self, state) -> str:
        s = np.asarray(state).reshape(-1)
        return ", ".join(f"{n}={self._num(v)}" for n, v in zip(self.feature_names, s))

    @staticmethod
    def _num(v):
        v = float(v)
        return int(v) if v.is_integer() else round(v, 4)

    def _context(self) -> str:
        parts = ["## Environment (PRISM MDP)", "```prism", self.prism_source.strip(), "```"]
        if self.constant_definitions:
            parts.append(f"Constant definitions: `{self.constant_definitions}`")
        parts.append("\n## Objective")
        if self.prop:
            goal = {"min_prop": "MINIMIZE", "max_prop": "MAXIMIZE"}.get(self.prop_type, "optimize")
            parts.append(f"The policy is verified with the PCTL property `{self.prop}` on the Markov chain induced by "
                         f"the network from the initial state. Goal: {goal} its value"
                         + (f" (target: {self.training_threshold})." if self.training_threshold is not None else "."))
        if self.reward_flag:
            reward_text = ("The PRISM `rewards` section defines REWARDS (gains): the simulator returns them as they are, "
                           "so higher is better.")
        else:
            reward_text = ("The PRISM `rewards` section defines COSTS: cool-mc negates them in the simulator, "
                           "so lower PRISM reward values are better.")
        if self.prop:
            parts.append(f"(Episodes are also logged with rewards. {reward_text})")
        else:
            parts.append("There is no property: the policy is judged by its expected cumulative reward per episode"
                         + (f" (episodes end after at most {self.max_steps} steps)" if self.max_steps else "")
                         + f". {reward_text}")
        parts.append("\n## Disabled actions")
        parts.append("If argmax(logits) picks an action whose guard is false in the current state, cool-mc executes "
                     "the alphabetically FIRST enabled action instead (enabled actions sorted by name)"
                     + (f", and during simulation that step is punished with reward -{self.wrong_action_penalty}"
                        if self.wrong_action_penalty else "")
                     + ". So make sure the intended action is enabled in every state where it is chosen.")
        parts.append("\n## Network interface")
        parts.append(f"Input x has {self.state_dimension} features (index: name):")
        parts += [f"  x[{i}]: {n}" for i, n in enumerate(self.feature_names)]
        parts.append(f"Output has {self.number_of_actions} logits (index: PRISM action label):")
        parts += [f"  out[{i}]: {a}" for i, a in enumerate(self.action_names)]
        parts.append("Inputs are the raw integer/float values of the PRISM state variables (booleans as 0/1).")
        parts.append("Required layer shapes (W as [out][in], b as [out]): "
                     + ", ".join(f"layer {k}: W[{o}][{i}], b[{o}]" for k, (o, i) in enumerate(self.shapes)))
        if self.extra_info:
            parts += ["\n## Additional information from the user", self.extra_info.strip()]
        return "\n".join(parts)

    def _history_str(self) -> str:
        if not self.history:
            return ""
        lines = ["\n## History of previous rounds"]
        for h in self.history[-10:]:
            lines.append(f"- {h['phase']} round {h['round']}: {h['summary']}")
        return "\n".join(lines)

    def _current_str(self) -> str:
        return ("\n## Current weights\n" + self._weights_json()
                + (f"\nYour notes on these weights: {self.notes}" if self.notes else ""))

    # ------------------------------------------------------------------ #
    # 1) Initial creation
    # ------------------------------------------------------------------ #
    def load_env(self, env):
        self.env = env
        sm = env.storm_bridge.state_mapper
        try:
            self.feature_names = sorted(sm.mapper.keys(), key=lambda n: sm.mapper[n])
        except Exception:
            self.feature_names = list(sm.get_feature_names())
        with open(env.storm_bridge.path, "r") as f:
            self.prism_source = f.read()
        if not self.constant_definitions:
            self.constant_definitions = getattr(env.storm_bridge, "constant_definitions", "") or ""
        if self.enabled is None:
            self.enabled = self._build_enabled_table(env)

        if self.loaded_from_checkpoint or self.deploy:
            print("[WSP] Using loaded weights; skipping initial creation.")
            return
        prompt = (self._context()
                  + "\n\n## Task\nWrite the weights of a policy network that achieves the objective above. "
                    "Study the PRISM guards, updates and labels first, decide the best action for the "
                    "relevant regions of the state space, then encode that strategy in the weights.")
        layers, notes = self._request_weights(prompt, "initial creation")
        if layers is None:
            raise RuntimeError(f"[WSP] Initial creation failed: {self.model} returned no valid weights "
                               f"after {self.retries + 1} attempts (see messages above). "
                               "Not continuing with a random policy.")
        self.set_weights(layers)
        self.notes = notes
        self.history.append({"phase": "create", "round": 0,
                             "summary": f"initial weights written ({self.last_check})"})
        self._export_network("wsp_step1.json", {"self_check": self.last_check})
        self._export("wsp_history.json", self.history)

    # ------------------------------------------------------------------ #
    # 2) Dataset-driven repair (behavioral cloning hook)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _group_labels(X, y):
        """Group rows by state: datasets such as all_optimal_dataset contain one
        row per optimal action, so a state is correct if the network picks ANY
        of its labels. Returns (unique_states, list_of_label_sets)."""
        groups = {}
        for x, label in zip(np.asarray(X, dtype=np.float32), np.asarray(y).reshape(-1).astype(int)):
            groups.setdefault(tuple(x.tolist()), set()).add(int(label))
        states = np.asarray(list(groups.keys()), dtype=np.float32)
        return states, list(groups.values())

    def _accuracy(self, X, y):
        """Returns (accuracy in %, unique states, label sets, predictions, correct mask)."""
        if X is None or y is None or len(X) == 0:
            return None, None, None, None, None
        states, labels = self._group_labels(X, y)
        with torch.no_grad():
            pred = torch.argmax(self.network.forward(states), dim=1).cpu().numpy()
        correct = np.array([p in ls for p, ls in zip(pred, labels)])
        return float(correct.mean() * 100.0), states, labels, pred, correct

    def behavioral_cloning(self, env, data: dict, epochs: int = 100, accuracy_threshold: float = 100.0):
        X_train, y_train = data.get("X_train"), data.get("y_train")
        X_test, y_test = data.get("X_test"), data.get("y_test")
        if X_train is None or y_train is None or len(X_train) == 0:
            print("[WSP] Dataset is empty; skipping dataset repair.", flush=True)
            return None, None, None, None, None
        best_weights, best_notes = self.get_weights(), self.notes
        best_acc, best_round = -1.0, 0
        rng = np.random.default_rng(0)
        for rnd in range(epochs + 1):
            acc, states, labels, pred, correct = self._accuracy(X_train, y_train)
            test_acc = self._accuracy(X_test, y_test)[0]
            print(f"[WSP] dataset round {rnd}: accuracy {acc:.2f}%"
                  + (f", test accuracy {test_acc:.2f}%" if test_acc is not None else ""), flush=True)
            self.history.append({"phase": "dataset", "round": rnd, "summary": f"accuracy {acc:.2f}%"
                                 + (f" (weights: {self.last_check})" if rnd > 0 else "")})
            if acc > best_acc:
                best_acc, best_round = acc, rnd
                best_weights, best_notes = self.get_weights(), self.notes
            if acc >= accuracy_threshold or rnd == epochs:
                break

            wrong_all = np.flatnonzero(~correct)
            wrong = wrong_all
            if len(wrong) > self.max_feedback_states:
                wrong = np.sort(rng.choice(wrong, self.max_feedback_states, replace=False))
            rows = [f"- {self._state_str(states[i])} -> expected "
                    + " or ".join(self.action_names[a] for a in sorted(labels[i]))
                    + f", network chose {self._action_str(states[i], int(pred[i]))}" for i in wrong]
            best = self._best_section(f"Most accurate network so far (accuracy {best_acc:.2f}%)",
                                      (best_weights, best_notes) if acc < best_acc else None)
            prompt = (self._context() + self._history_str() + self._current_str() + best
                      + f"\n\n## Dataset feedback\nThe dataset labels {len(states)} distinct states with their "
                        f"correct action(s). The network is correct on {acc:.2f}% of them. "
                        f"{len(wrong_all)} states are wrong"
                      + (f" (random sample of {len(wrong)} shown)" if len(wrong) < len(wrong_all) else "")
                      + ":\n" + "\n".join(rows)
                      + "\n\n## Task\nEdit the weights so that these states get the expected action "
                        "without breaking the states that are already correct. Return the complete new weights.")
            layers, notes = self._request_weights(prompt, f"dataset repair {rnd + 1}")
            if layers is None:
                break
            self.set_weights(layers)
            self.notes = notes

        self.set_weights(best_weights)
        self.notes = best_notes
        self._export_network("wsp_step2.json", {"accuracy": best_acc, "round": best_round})
        self._export("wsp_history.json", self.history)
        test_acc = self._accuracy(X_test, y_test)[0]
        return best_round, best_acc, test_acc, None, None

    # ------------------------------------------------------------------ #
    # 3) Verification-driven repair (model checking hook)
    # ------------------------------------------------------------------ #
    def select_action(self, state: np.ndarray, deploy: bool = False) -> int:
        with torch.no_grad():
            action = int(torch.argmax(self.network.forward(np.asarray(state, dtype=np.float32))).item())
        if deploy:
            # The model checker queries every reachable state with deploy=True.
            self.visited[self._key(state)] = action
        return action

    def get_raw_outputs(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.network.forward(np.asarray(state, dtype=np.float32)).cpu().numpy()

    def _is_better(self, result) -> bool:
        if self.best_mc_result is None:
            return True
        if self.prop_type == "min_prop":
            return result < self.best_mc_result
        return result > self.best_mc_result

    def _satisfied(self, result) -> bool:
        if self.training_threshold is None:
            return False
        if self.prop_type == "min_prop":
            return result <= self.training_threshold
        if self.prop_type == "max_prop":
            return result >= self.training_threshold
        return False

    def model_checking_learn(self, model_checking_result, model_checking_info, model_checker=None):
        visited, self.visited = self.visited, {}
        if self.deploy or model_checking_result is None:
            return
        self.evaluations += 1
        rnd = self.evaluations - 1
        result = float(model_checking_result)
        improved = self._is_better(result)
        if improved:
            # The network currently holds exactly the weights Storm just verified (repairs are applied lazily).
            self.best_mc_result = result
            self.best_mc_weights = (self.get_weights(), self.notes)
        worse_than_best = (result > self.best_mc_result if self.prop_type == "min_prop"
                           else result < self.best_mc_result)
        self.history.append({"phase": "model checking", "round": rnd, "summary": f"{self.prop} = {result:.6g}"})
        self.mc_results.append(result)
        if improved:
            self._export_network("wsp_best.json", {"result": result, "round": rnd})
        self._export("wsp_mc.json", {"prop": self.prop, "results": self.mc_results})
        self._export("wsp_history.json", self.history)
        print(f"[WSP] model checking round {rnd}: {self.prop} = {result:.6g}", flush=True)
        if self._satisfied(result):
            print(f"[WSP] Property satisfied ({result:.6g}); no repair.", flush=True)
            return
        if self.total_evaluations is not None and self.evaluations >= self.total_evaluations:
            print("[WSP] Last evaluation of this run; no repair requested (it could not be verified).", flush=True)
            return

        items = list(visited.items())
        terminal = [s for s, _ in items if self._enabled_at(s) == []]
        items = [(s, a) for s, a in items if self._enabled_at(s) != []]
        fallback = [(s, a) for s, a in items
                    if self._enabled_at(s) is not None and self.action_names[a] not in self._enabled_at(s)]
        fallback_keys = {s for s, _ in fallback}
        normal = [(s, a) for s, a in items if s not in fallback_keys]
        rng = np.random.default_rng(rnd)
        shown = fallback[: self.max_feedback_states]
        room = self.max_feedback_states - len(shown)
        if room > 0 and normal:
            idx = rng.choice(len(normal), min(room, len(normal)), replace=False)
            shown += [normal[i] for i in sorted(idx)]
        rows = [f"- {self._state_str(s)} -> {self._action_str(s, a)}" for s, a in shown]
        trace = self._rollout_trace()
        best = self._best_section(f"Best network verified so far ({self.prop} = {self.best_mc_result:.6g})",
                                  self.best_mc_weights if worse_than_best else None)
        prompt = (self._context() + self._history_str() + self._current_str() + best
                  + f"\n\n## Model checking feedback\nStorm built the DTMC induced by the current network "
                    f"({len(visited)} reachable states, {len(terminal)} of them terminal) and computed "
                    f"`{self.prop}` = {result:.6g}"
                  + (f" (best so far {self.best_mc_result:.6g})" if self.best_mc_result is not None else "")
                  + (f", target {self.training_threshold}" if self.training_threshold is not None else "")
                  + f".\nIn {len(fallback)} reachable states the network picks a disabled action (listed first)."
                  + "\nDecisions of the network in the reachable non-terminal states"
                  + (f" ({len(shown)} of {len(items)} shown)" if len(shown) < len(items) else "") + ":\n"
                  + "\n".join(rows)
                  + (("\n\n## One simulated episode with the current network\n" + trace) if trace else "")
                  + "\n\n## Task\nFind the states where the chosen action hurts the objective (use the PRISM "
                    "transitions), edit the weights to fix them, and return the complete new weights.")
        layers, notes = self._request_weights(prompt, f"model checking repair {rnd + 1}")
        if layers is not None:
            # Applied lazily in episodic_learn, so a save() right after this call
            # still stores the weights that were just verified.
            self.pending_weights = (layers, notes, self.last_check)

    def episodic_learn(self):
        if self.pending_weights is not None:
            layers, notes, check = self.pending_weights
            self.set_weights(layers)
            self.notes = notes
            self.pending_weights = None
            self.history.append({"phase": "repair applied", "round": self.evaluations, "summary": check})
            self._export("wsp_history.json", self.history)
            print("[WSP] Applied repaired weights; they will be verified at the next evaluation.", flush=True)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, artifact_path="model"):
        os.makedirs("tmp_model", exist_ok=True)
        self.network.save_checkpoint("tmp_model/bc_network.chkpt")
        with open("tmp_model/wsp_weights.json", "w") as f:  # full precision = exactly the checkpoint
            json.dump({"notes": self.notes, **json.loads(self._weights_json(decimals=None))}, f, indent=1)
        with open("tmp_model/wsp_history.json", "w") as f:
            json.dump(self.history, f, indent=1)
        mlflow.log_artifacts("tmp_model", artifact_path=artifact_path)
        shutil.rmtree("tmp_model")

    def load(self, model_root_folder_path: str):
        if not model_root_folder_path:
            return
        checkpoint_path = os.path.join(model_root_folder_path, "bc_network.chkpt")
        if os.path.exists(checkpoint_path):
            self.network.load_checkpoint(checkpoint_path)
            self.loaded_from_checkpoint = True
            notes_path = os.path.join(model_root_folder_path, "wsp_weights.json")
            if os.path.exists(notes_path):
                with open(notes_path) as f:
                    self.notes = json.load(f).get("notes", "")
            print(f"[WSP] Loaded weights from {checkpoint_path}")

    def get_hyperparameters(self):
        return {"model": self.model, "number_of_neurons": self.number_of_neurons,
                "temperature": self.temperature, "info_file": self.info_file,
                "self_check": self.self_check}
