"""Ollama-backed black-box LLM policy.

The agent is configured via a semicolon-separated string so it can be
passed through a single CLI flag, e.g.

    ollama;model=gemma3:1b;temperature=0.7;\
    prompt_file=prism_files/frozen_lake_prompt.txt;\
    host=http://host.docker.internal:11434;seed=0;timeout=30

The prompt file is a plain text template. Placeholders use Python's
``str.format`` syntax and may reference:

  * any state feature by name (e.g. ``{pos}`` for frozen_lake),
  * ``{actions}`` -> comma-separated list of valid action names,
  * ``{action_list}`` -> same list, one per line.

The agent runs inside a docker container, so ``host`` should point at
the host machine's Ollama server. On Docker Desktop (macOS / Windows),
``http://host.docker.internal:11434`` works out of the box. On Linux,
add ``--add-host=host.docker.internal:host-gateway`` when starting the
container.
"""
import json
import os
import re
from typing import List

import numpy as np
import requests

try:
    import jinja2  # noqa: F401
    _JINJA = jinja2.Environment(undefined=jinja2.StrictUndefined,
                                keep_trailing_newline=True)
except ImportError:  # jinja2 is optional; only needed for {% %} templates
    jinja2 = None
    _JINJA = None

from common.agents.black_box_agent import BlackBoxAgent


def _parse_constants(constant_definitions: str) -> dict:
    """Parse a PRISM constant_definitions string ("a=1,b=2.5") into a dict.
    Values are left as strings; feature-level coercion handles numerics."""
    out = {}
    for part in constant_definitions.split(","):
        k, _, v = part.partition("=")
        k, v = k.strip(), v.strip()
        if not k:
            continue
        try:
            if "." in v:
                out[k] = float(v)
            else:
                out[k] = int(v)
        except ValueError:
            out[k] = v
    return out


class OllamaAgent(BlackBoxAgent):

    def __init__(self, config_str: str, action_names: List[str],
                 feature_names: List[str] = None):
        super().__init__()
        self.model: str = "gemma3:1b"
        self.temperature: float = 0.7
        self.prompt_file: str = ""
        self.host: str = "http://host.docker.internal:11434"
        # By default we do NOT seed the Ollama sampler: the stochasticity
        # of repeated queries is what the policy-sampling methodology is
        # designed to measure. Set ``seed`` in the config string to force
        # a deterministic sampler.
        self.seed: int = None
        self.timeout: int = 30
        # How long Ollama keeps the model loaded after each request.
        # "0" unloads immediately (best for multi-model comparisons where
        # holding both models resident would exceed VRAM). Use "5m" or
        # similar to amortise load time across many queries of the same
        # model in one run.
        self.keep_alive: str = "30s"
        # Print the prompt, raw response, and parsed action for the first
        # ``debug_first_n`` queries. Useful for verifying that the LLM is
        # being prompted correctly and that the parser picks the right
        # action. Set to 0 to disable.
        self.debug_first_n: int = 3
        self._debug_seen: int = 0
        self._parse_config(config_str)

        self.action_names = list(action_names)
        self.feature_names = list(feature_names) if feature_names else []
        self.constants: dict = {}

        if not self.prompt_file:
            raise ValueError(
                "OllamaAgent requires 'prompt_file' in its config string"
            )
        with open(self.prompt_file, "r") as f:
            self.prompt_template = f.read()

        self._action_regex = [
            (name, re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE))
            for name in self.action_names
        ]

        print(f"OllamaAgent(model={self.model}, temperature={self.temperature}, "
              f"host={self.host}, prompt_file={self.prompt_file})")

    def _parse_config(self, config_str: str) -> None:
        for part in config_str.split(";")[1:]:
            key, _, value = part.partition("=")
            key, value = key.strip(), value.strip()
            if key == "":
                continue
            if key == "model":
                self.model = value
            elif key == "temperature":
                self.temperature = float(value)
            elif key == "prompt_file":
                self.prompt_file = value
            elif key == "host":
                self.host = value.rstrip("/")
            elif key == "seed":
                self.seed = int(value)
            elif key == "timeout":
                self.timeout = int(value)
            elif key == "keep_alive":
                self.keep_alive = value
            elif key == "debug_first_n":
                self.debug_first_n = int(value)
            else:
                print(f"Warning: unknown OllamaAgent parameter '{key}'")

    def _render_prompt(self, state: np.ndarray,
                       available_actions=None) -> str:
        avail = (list(available_actions) if available_actions is not None
                 else list(self.action_names))
        fields = {
            # Global list (all actions the MDP has anywhere).
            "actions": ", ".join(self.action_names),
            "action_list": "\n".join(f"- {a}" for a in self.action_names),
            # Per-state list (Act(s) if the updater passed it in; falls
            # back to the global list for compatibility).
            "available_actions": ", ".join(avail),
            "available_action_list": "\n".join(f"- {a}" for a in avail),
        }
        # PRISM constant_definitions -> individual placeholders (e.g. {control}).
        # State-feature names win on collision.
        for k, v in self.constants.items():
            fields.setdefault(k, v)
        for i, name in enumerate(self.feature_names):
            value = state[i] if i < len(state) else ""
            if isinstance(value, (np.integer,)):
                value = int(value)
            elif isinstance(value, (np.floating,)):
                value = float(value)
            fields[name] = value
        # Use Jinja2 when the template contains control directives or
        # `{{ ... }}` expressions (so prompts can branch on state with
        # {% if ... %}/{% elif ... %}/{% endif %}). Otherwise fall back
        # to str.format, which keeps the simple `{var}` syntax working
        # for existing prompt files.
        if "{%" in self.prompt_template or "{{" in self.prompt_template:
            if _JINJA is None:
                raise RuntimeError(
                    "Prompt template uses Jinja2 directives but jinja2 "
                    "is not installed."
                )
            try:
                return _JINJA.from_string(self.prompt_template).render(**fields)
            except jinja2.UndefinedError as e:
                raise KeyError(
                    f"Prompt template references unknown placeholder: {e}. "
                    f"Available: {sorted(fields)}"
                )
        try:
            return self.prompt_template.format(**fields)
        except KeyError as e:
            raise KeyError(
                f"Prompt template references unknown placeholder {e}; "
                f"available: {sorted(fields)}"
            )

    def _query_ollama(self, prompt: str) -> str:
        url = f"{self.host}/api/generate"
        options = {"temperature": self.temperature}
        if self.seed is not None:
            options["seed"] = int(self.seed)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": options,
        }
        try:
            r = requests.post(url, json=payload, timeout=self.timeout)
            r.raise_for_status()
            return r.json().get("response", "")
        except requests.exceptions.Timeout as exc:
            print(f"[OllamaAgent] TIMEOUT after {self.timeout}s "
                  f"(model={self.model}, host={self.host}): {exc}. "
                  "Returning empty response -> parser will fall back to "
                  "the alphabetical-first action.", flush=True)
            return ""
        except requests.exceptions.RequestException as exc:
            print(f"[OllamaAgent] REQUEST ERROR "
                  f"(model={self.model}, host={self.host}): "
                  f"{type(exc).__name__}: {exc}. "
                  "Returning empty response -> parser will fall back to "
                  "the alphabetical-first action.", flush=True)
            return ""
        except ValueError as exc:  # JSON decode error
            print(f"[OllamaAgent] BAD RESPONSE "
                  f"(model={self.model}, host={self.host}): {exc}. "
                  "Returning empty response.", flush=True)
            return ""

    def _parse_action(self, text: str) -> int:
        """Return the LAST action name that appears in the model's output.
        Reasoning models typically enumerate candidates during analysis
        and state their final choice last, so the tail match is the
        decision. Returns -1 when no action is found; the policy_sampling
        updater treats out-of-range indices as unavailable and falls back
        to cool_mc's alphabetical first-available-action default."""
        best_idx = None
        best_pos = -1
        for i, (_, pat) in enumerate(self._action_regex):
            last = None
            for m in pat.finditer(text):
                last = m
            if last is not None and last.start() > best_pos:
                best_pos = last.start()
                best_idx = i
        if best_idx is None:
            return -1
        return best_idx

    def load_env(self, env) -> None:
        """Extract feature names and PRISM constant definitions from env."""
        if not self.feature_names:
            example = json.loads(env.storm_bridge.state_json_example)
            self.feature_names = list(example.keys())
        self.constants = _parse_constants(
            getattr(env.storm_bridge, "constant_definitions", "") or ""
        )

    def sample_action(self, state: np.ndarray,
                      available_actions=None) -> int:
        prompt = self._render_prompt(state, available_actions)
        response = self._query_ollama(prompt)
        action_idx = self._parse_action(response)
        if self._debug_seen < self.debug_first_n:
            self._debug_seen += 1
            action_name = (self.action_names[action_idx]
                           if 0 <= action_idx < len(self.action_names)
                           else "<UNPARSED>")
            print("-" * 64)
            print(f"[OllamaAgent debug query {self._debug_seen}"
                  f"/{self.debug_first_n}]  model={self.model}")
            print("--- prompt ---")
            print(prompt)
            print("--- response ---")
            print(response)
            print(f"--- parsed: idx={action_idx} name={action_name} ---")
            print("-" * 64, flush=True)
        return action_idx
