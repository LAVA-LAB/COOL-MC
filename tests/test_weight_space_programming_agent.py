"""Tests for the weight-space programming agent and its environment descriptions.

No LLM is contacted (the chat call is replaced by a fake) and nothing is trained.
Run from the repository root:  python -m unittest tests.test_weight_space_programming_agent
"""
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from collections import deque
from itertools import product

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import stormpy  # noqa: E402

from common.agents.weight_space_programming_agent import WeightSpaceProgrammingAgent  # noqa: E402
from common.safe_gym.safe_gym import SafeGym  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRANSPORTER = os.path.join(ROOT, "prism_files", "transporter.prism")
FROZEN_LAKE = os.path.join(ROOT, "prism_files", "frozen_lake.prism")
T_CONST = "MAX_JOBS=2,MAX_FUEL=10"
F_CONST = "control=0.33,start_position=0"
JOBS = "Pmax=? [ F jobs_done=2 ]"
EMPTY = 'Pmin=? [ F "empty" ]'


def build_mdp(path, constants):
    prog = stormpy.parse_prism_program(path)
    prog = stormpy.preprocess_symbolic_input(prog, [], constants)[0].as_prism_program()
    options = stormpy.BuilderOptions(True, True)
    options.set_build_state_valuations()
    options.set_build_choice_labels()
    model = stormpy.build_sparse_model_with_options(prog, options)
    states = {}
    for sid in range(model.nr_states):
        val = json.loads(str(model.state_valuations.get_json(sid)))
        choices = {}
        for a in range(model.get_nr_available_actions(sid)):
            ci = model.get_choice_index(sid, a)
            labels = list(model.choice_labeling.get_labels_of_choice(ci))
            succ = [(e.column, e.value()) for e in model.transition_matrix.get_row(ci)]
            choices[labels[0] if labels else None] = succ
        states[sid] = (val, choices, set(model.labeling.get_labels_of_state(sid)))
    return model, prog, states


def check_property(prog, model_path, constants, prop):
    props = stormpy.parse_properties_for_prism_program(prop, prog)
    model = stormpy.build_model(prog, props)
    return stormpy.model_checking(model, props[0]).at(model.initial_states[0])


def make_agent(env, prop, layers=(32, 32), cfg="", **kw):
    agent = WeightSpaceProgrammingAgent("weight_space_programming_agent;model=fake;debug=0" + cfg,
                                        env.observation_space.shape[0], list(layers),
                                        len(env.action_mapper.actions), env.action_mapper.actions,
                                        prop=prop, **kw)
    agent.loaded_from_checkpoint = True     # load_env must not contact an LLM
    agent.load_env(env)
    agent.loaded_from_checkpoint = False
    return agent


def transporter_reference_layers(feature_names, action_names, shapes):
    """Hand-written (no training) 10 -> 32 -> 32 -> 6 ReLU network for the transporter.

    Strategy: target is (0,4) without and (0,0) with passenger. Keep going to the target while
    fuel >= dist(pos, target) + 1 + dist(target, gas)=3, otherwise drive to the gas station (1,2).
    At the target PICK_UP / DROP, on the gas station move towards the target (refuels).
    """
    f = {n: i for i, n in enumerate(feature_names)}
    act = {n: i for i, n in enumerate(action_names)}
    (h1, d_in), (h2, _), (n_out, _) = shapes
    W0, b0 = np.zeros((h1, d_in)), np.zeros(h1)
    units = {}

    def unit1(name, coeffs, bias):
        k = len(units)
        for feat, c in coeffs.items():
            W0[k, f[feat]] = c
        b0[k] = bias
        units[name] = k

    unit1("p", {"passenger": 1}, 0)
    unit1("y-3", {"y": 1}, -3)
    unit1("y-4", {"y": 1}, -4)
    unit1("1-y", {"y": -1}, 1)
    unit1("-y", {"y": -1}, 0)
    unit1("y-2", {"y": 1}, -2)
    unit1("2-y", {"y": -1}, 2)
    unit1("x", {"x": 1}, 0)
    unit1("x-1", {"x": 1}, -1)
    unit1("x-2", {"x": 1}, -2)
    # g0 = fuel - x + y - 8 - 20p   (enough fuel to finish towards (0,4) and then reach the gas station)
    unit1("g0+1", {"fuel": 1, "x": -1, "y": 1, "passenger": -20}, -7)
    unit1("g0", {"fuel": 1, "x": -1, "y": 1, "passenger": -20}, -8)
    # g1 = fuel - x - y - 24 + 20p  (same towards (0,0))
    unit1("g1+1", {"fuel": 1, "x": -1, "y": -1, "passenger": 20}, -23)
    unit1("g1", {"fuel": 1, "x": -1, "y": -1, "passenger": 20}, -24)

    def ind(terms, const=0.0):  # indicator = const + sum(c * h[unit])
        v = np.zeros(h1)
        for name, c in terms.items():
            v[units[name]] += c
        return v, const

    def neg(i):
        return -i[0], 1.0 - i[1]

    def plus(*inds):
        return sum(i[0] for i in inds), sum(i[1] for i in inds)

    P = ind({"p": 1})
    NP = neg(P)
    Y4 = ind({"y-3": 1, "y-4": -1})
    Y0 = ind({"1-y": 1, "-y": -1})
    Y3 = ind({"y-2": 1, "y-3": -1})
    Y1 = ind({"2-y": 1, "1-y": -1})
    Y2 = neg(plus(Y3, Y1))
    X1 = ind({"x": 1, "x-1": -1})
    X0 = neg(X1)
    X2 = ind({"x-1": 1, "x-2": -1})
    XE1 = plus(X1, (-X2[0], -X2[1]))
    C = ind({"g0+1": 1, "g0": -1, "g1+1": 1, "g1": -1})
    NC = neg(C)

    W1, b1 = np.zeros((h2, h1)), np.zeros(h2)
    W2, b2 = np.zeros((n_out, h2)), np.zeros(n_out)
    rules = [((NP, X0, Y4), "PICK_UP", 100), ((P, X0, Y0), "DROP", 100),
             ((XE1, Y2, NP), "NORTH", 50), ((XE1, Y2, P), "SOUTH", 50),
             ((C, NP, neg(Y4)), "NORTH", 10), ((C, NP, Y4), "WEST", 10),
             ((C, P, neg(Y0)), "SOUTH", 10), ((C, P, Y0), "WEST", 10),
             ((NC, Y1), "NORTH", 5), ((NC, Y3), "SOUTH", 5),
             ((NC, Y2, X0), "EAST", 5), ((NC, Y2, X2), "WEST", 5)]
    for k, (inds, action, weight) in enumerate(rules):
        vec, const = plus(*inds)
        W1[k], b1[k] = vec, const - (len(inds) - 1)      # AND of 0/1 indicators
        W2[act[action], k] = weight
    return [(W0, b0), (W1, b1), (W2, b2)]


class FakeLLM:
    """Replaces WeightSpaceProgrammingAgent._chat: returns the queued answers in order."""

    def __init__(self, answers):
        self.answers, self.prompts = list(answers), []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.answers.pop(0)


def answer(layers, checks=(), notes="test"):
    return json.dumps({"notes": notes,
                       "layers": [{"W": np.asarray(W).tolist(), "b": np.asarray(b).tolist()} for W, b in layers],
                       "checks": list(checks)})


class TestTransporterWiring(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.env = SafeGym(TRANSPORTER, T_CONST, 100, 1000, False, 0, "")
        cls.agent = make_agent(cls.env, JOBS, training_threshold=0.99, wrong_action_penalty=1000, max_steps=100)
        cls.ref = transporter_reference_layers(cls.agent.feature_names, cls.agent.action_names, cls.agent.shapes)

    def test_feature_names_match_state_vector(self):
        state = dict(zip(self.agent.feature_names, self.env.reset().tolist()))
        self.assertEqual(state, {"done": 0, "fuel": 10, "jobs_done": 0, "passenger": 0, "passenger_dest_x": 0,
                                 "passenger_dest_y": 0, "passenger_loc_x": 0, "passenger_loc_y": 4, "x": 2, "y": 2})

    def test_output_index_i_executes_action_i(self):
        expected = {"EAST": (3, 2), "NORTH": (2, 3), "SOUTH": (2, 1), "WEST": (1, 2),
                    "DROP": (3, 2), "PICK_UP": (3, 2)}  # DROP/PICK_UP disabled at start -> fallback EAST
        names = self.agent.feature_names
        for i, name in enumerate(self.agent.action_names):
            layers = [(np.zeros((o, n)), np.zeros(o)) for o, n in self.agent.shapes]
            layers[-1][1][i] = 1.0
            self.agent.set_weights(layers)
            s = self.env.reset()
            self.assertEqual(self.agent.select_action(s), i)
            nxt, reward, _, _, _ = self.env.step(i)
            d = dict(zip(names, nxt.tolist()))
            self.assertEqual((d["x"], d["y"]), expected[name], name)
            self.assertEqual(reward, -1000 if name in ("DROP", "PICK_UP") else -25.0, name)
            if name in ("DROP", "PICK_UP"):
                self.assertIn("NOT enabled here -> cool-mc executes EAST", self.agent._action_str(s, i))

    def test_enabled_table_matches_simulator(self):
        rng = np.random.default_rng(0)
        compared = 0
        for _ in range(150):
            s = self.env.reset()
            for _ in range(60):
                sim = sorted(self.env.storm_bridge.simulator.available_actions())
                self.assertEqual(self.agent._enabled_at(s), sim)
                compared += 1
                s, _, done, _, _ = self.env.step(int(rng.integers(6)))
                if done:
                    break
        self.assertGreater(compared, 500)

    def test_reference_network_is_optimal_for_both_properties(self):
        self.agent.set_weights(self.ref)
        mc = self.env.storm_bridge.model_checker
        jobs, _ = mc.induced_markov_chain(self.agent, None, self.env, T_CONST, JOBS)
        empty, _ = mc.induced_markov_chain(self.agent, None, self.env, T_CONST, EMPTY)
        self.assertAlmostEqual(jobs, 1.0, places=9)
        self.assertAlmostEqual(empty, 0.0, places=9)
        trace = self.agent._rollout_trace()
        self.assertIn("episode ended", trace)
        self.assertIn("jobs_done=2", trace.splitlines()[-1])
        self.assertNotIn("NOT enabled", trace)

    def test_self_check_catches_transposed_square_layer(self):
        bad = [self.ref[0], (self.ref[1][0].T, self.ref[1][1]), self.ref[2]]   # shapes still valid!
        checks = []
        for x, y, fuel, p, action in [(2, 2, 10, 0, "NORTH"), (0, 4, 6, 0, "PICK_UP"), (1, 2, 2, 1, "SOUTH"),
                                      (0, 0, 8, 1, "DROP"), (0, 4, 5, 1, "SOUTH")]:
            st = {n: 0 for n in self.agent.feature_names}
            st.update(x=x, y=y, fuel=fuel, passenger=p)
            checks.append({"state": st, "action": action})
        n, problems = self.agent._run_self_check(self.ref, checks)
        self.assertEqual((n, problems), (5, []))
        self.assertGreater(len(self.agent._run_self_check(bad, checks)[1]), 0)
        fake = FakeLLM([answer(bad, checks), answer(self.ref, checks)])
        self.agent._chat = fake
        layers, _ = self.agent._request_weights("PROMPT", "test")
        self.assertEqual(len(fake.prompts), 2)
        self.assertIn("rejected by the self-check", fake.prompts[1])
        np.testing.assert_array_equal(layers[1][0], self.ref[1][0])
        self.assertEqual(self.agent.last_check, "self-check 5/5 states as intended")

    def test_malformed_answers_are_retried(self):
        good = answer(self.ref)
        for broken in ['{"layers": 5}', "no json", answer(self.ref[:2]),
                       answer([self.ref[0], self.ref[1], (self.ref[2][0][:, :5], self.ref[2][1])]),
                       answer([(np.full_like(self.ref[0][0], np.nan), self.ref[0][1])] + self.ref[1:])]:
            fake = FakeLLM([broken, good])
            self.agent._chat = fake
            layers, _ = self.agent._request_weights("PROMPT", "test")
            self.assertIsNotNone(layers, broken[:40])
            self.assertEqual(len(fake.prompts), 2)
            self.assertIn("was rejected", fake.prompts[1])

    def test_repairs_are_applied_lazily_and_not_after_last_evaluation(self):
        agent = make_agent(self.env, JOBS, training_threshold=0.99, wrong_action_penalty=1000,
                           max_steps=100, num_episodes=2, eval_interval=1)
        zero = [(np.zeros((o, n)), np.zeros(o)) for o, n in agent.shapes]
        agent.set_weights(zero)
        fake = FakeLLM([answer(self.ref)])
        agent._chat = fake
        mc = self.env.storm_bridge.model_checker
        res, info = mc.induced_markov_chain(agent, None, self.env, T_CONST, JOBS)
        agent.model_checking_learn(res, info, mc)
        self.assertEqual(len(fake.prompts), 1)
        self.assertIn("One simulated episode", fake.prompts[0])
        self.assertIn("NOT enabled here", fake.prompts[0])      # zero network -> DROP everywhere
        np.testing.assert_array_equal(agent.get_weights()[0][0], zero[0][0])   # not yet applied
        agent.episodic_learn()
        np.testing.assert_array_equal(agent.get_weights()[0][0], np.float32(self.ref[0][0]))
        res, info = mc.induced_markov_chain(agent, None, self.env, T_CONST, JOBS)
        self.assertAlmostEqual(res, 1.0)
        agent.training_threshold = 2.0       # pretend unsatisfied: still no call in the last evaluation
        agent.model_checking_learn(res, info, mc)
        self.assertEqual(len(fake.prompts), 1)

    def _checks(self):
        checks = []
        for x, y, fuel, p, action in [(2, 2, 10, 0, "NORTH"), (0, 4, 6, 0, "PICK_UP"), (0, 0, 8, 1, "DROP")]:
            st = {n: 0 for n in self.agent.feature_names}
            st.update(x=x, y=y, fuel=fuel, passenger=p)
            checks.append({"state": st, "action": action})
        return checks

    def test_experiment_log_classifies_every_llm_request(self):
        bad = [self.ref[0], (self.ref[1][0].T, self.ref[1][1]), self.ref[2]]      # fails the self-check
        w33 = np.vstack([self.ref[0][0], self.ref[0][0][:1]])                     # 33 rows instead of 32
        good, wrong = answer(self.ref, self._checks()), answer(bad, self._checks())

        def server_error_then_good(prompts=[]):
            def chat(prompt):
                prompts.append(prompt)
                if len(prompts) == 1:
                    raise TimeoutError("server error")
                return good
            return chat

        with tempfile.TemporaryDirectory() as tmp:
            agent = make_agent(self.env, JOBS, cfg=f";export_dir={tmp}")
            for purpose, chat in [
                    ("a", FakeLLM(["no json", '{"layers": 5}', answer([(w33, self.ref[0][1])] + self.ref[1:])])),
                    ("b", FakeLLM([wrong, good])),
                    ("c", FakeLLM([wrong, wrong, wrong])),
                    ("d", FakeLLM([wrong, "no json", "no json"])),
                    ("e", server_error_then_good())]:
                agent._chat = chat
                layers, _ = agent._request_weights("PROMPT", purpose)
                self.assertEqual(layers is None, purpose == "a", purpose)
            with open(os.path.join(tmp, "wsp_calls.json")) as f:
                outcomes = [(c["purpose"], c["outcome"]) for c in json.load(f)]
        self.assertEqual(outcomes, [
            ("a", "invalid_json"), ("a", "invalid_json"), ("a", "wrong_shape"),
            ("b", "self_check_failed"), ("b", "accepted"),
            ("c", "self_check_failed"), ("c", "self_check_failed"), ("c", "accepted_despite_failure"),
            ("d", "self_check_failed"), ("d", "invalid_json"), ("d", "invalid_json"), ("d", "fallback_accepted"),
            ("e", "request_failed"), ("e", "accepted")])

    def test_exports_hold_the_networks_of_every_step(self):
        mc = self.env.storm_bridge.model_checker
        with tempfile.TemporaryDirectory() as tmp:
            # Step 1: creation through the real load_env path (fake LLM).
            agent = WeightSpaceProgrammingAgent(
                f"weight_space_programming_agent;model=fake;debug=0;export_dir={tmp}",
                self.env.observation_space.shape[0], [32, 32], len(self.env.action_mapper.actions),
                self.env.action_mapper.actions, prop=JOBS, training_threshold=0.99, num_episodes=2, eval_interval=1)
            agent._chat = FakeLLM([answer(self.ref, self._checks())])
            agent.load_env(self.env)
            step1 = json.load(open(os.path.join(tmp, "wsp_step1.json")))
            self.assertIn("3/3", step1["self_check"])
            np.testing.assert_allclose(step1["layers"][1]["W"], self.ref[1][0], atol=1e-6)
            # Step 2: a dataset the network already solves needs no LLM call.
            state = self.env.reset()
            X = np.asarray([state], dtype=np.float32)
            y = np.asarray([agent.select_action(state)])
            agent.behavioral_cloning(self.env, {"X_train": X, "y_train": y}, epochs=5, accuracy_threshold=100.0)
            step2 = json.load(open(os.path.join(tmp, "wsp_step2.json")))
            self.assertEqual((step2["accuracy"], step2["round"]), (100.0, 0))
            # Step 3: a worse network is verified first, then the repair; the export keeps the best.
            agent.set_weights([(np.zeros((o, n)), np.zeros(o)) for o, n in agent.shapes])
            res0, info0 = mc.induced_markov_chain(agent, None, self.env, T_CONST, JOBS)
            agent._chat = FakeLLM([answer(self.ref)])
            agent.model_checking_learn(res0, info0, mc)
            agent.episodic_learn()
            res1, info1 = mc.induced_markov_chain(agent, None, self.env, T_CONST, JOBS)
            agent.model_checking_learn(res1, info1, mc)
            self.assertEqual(json.load(open(os.path.join(tmp, "wsp_mc.json")))["results"], [res0, res1])
            best = json.load(open(os.path.join(tmp, "wsp_best.json")))
        self.assertLess(res0, res1)
        self.assertEqual((best["result"], best["round"]), (res1, 1))
        check = make_agent(self.env, JOBS)
        check.set_weights([(np.asarray(l["W"]), np.asarray(l["b"])) for l in best["layers"]])
        self.assertEqual(mc.induced_markov_chain(check, None, self.env, T_CONST, JOBS)[0], res1)

    def test_evaluation_script_recomputes_logged_results(self):
        spec = importlib.util.spec_from_file_location("wsp_evaluate", os.path.join(ROOT, "examples", "wsp_evaluate.py"))
        wsp_evaluate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wsp_evaluate)
        layers = json.loads(self.agent._layers_json(self.ref, decimals=None))
        with tempfile.TemporaryDirectory() as tmp:
            run = {"rq": "rq1", "order": 1, "env": "transporter", "label": "Transporter", "prism": TRANSPORTER,
                   "constants": T_CONST, "prop": JOBS, "layers": 2, "neurons": 32, "reward_flag": 0,
                   "description": "transporter_wsp.md", "model": "fake", "seed": 1, "optimum": 1.0,
                   "states": 8560, "threshold": 0.9999, "dir": tmp, "id": "fake", "status": "done"}
            for name in ("wsp_step1.json", "wsp_step2.json", "wsp_best.json"):
                with open(os.path.join(tmp, name), "w") as f:
                    json.dump(layers, f)
            with open(os.path.join(tmp, "wsp_mc.json"), "w") as f:
                json.dump({"results": [1.0]}, f)
            with open(os.path.join(tmp, "wsp_calls.json"), "w") as f:
                json.dump([{"outcome": "accepted"}, {"outcome": "fallback_accepted"}], f)
            warnings = []
            wsp_evaluate.evaluate_run(run, 1e-4, warnings, with_accuracy=True)
        self.assertEqual(warnings, [])
        self.assertEqual(run["requests"], 1)
        for stage in ("step1", "step2", "best"):
            self.assertAlmostEqual(run[stage]["q"], 1.0)
            self.assertTrue(run[stage]["opt"])
            self.assertTrue(0 < run[stage]["acc"] <= 100)
        three = wsp_evaluate.check_network(run, "best", "MAX_JOBS=3,MAX_FUEL=10", "Pmax=? [ F jobs_done=3 ]")
        self.assertTrue(0 <= three <= 1)

    def test_retries_show_the_rejected_weights_until_fixed(self):
        bad = [self.ref[0], (self.ref[1][0].T, self.ref[1][1]), self.ref[2]]
        bad_json = self.agent._layers_json([(np.float32(W), np.float32(b)) for W, b in bad])
        w33 = np.vstack([self.ref[0][0], self.ref[0][0][:1]])           # 33 rows instead of 32
        broken = answer([(w33, self.ref[0][1])] + self.ref[1:])
        fake = FakeLLM([answer(bad, self._checks(), notes="NOTES-OF-BAD"), broken,
                        answer(self.ref, self._checks())])
        self.agent._chat = fake
        layers, _ = self.agent._request_weights("PROMPT", "test")
        self.assertEqual(len(fake.prompts), 3)
        for p in fake.prompts[1:]:   # rejected weights stay visible, also after the invalid 2nd answer
            self.assertIn("These are the weights you wrote:\n" + bad_json, p)
            self.assertIn("NOTES-OF-BAD", p)
            self.assertIn("Do not start over", p)
        self.assertIn("rejected (invalid)", fake.prompts[2])
        np.testing.assert_array_equal(layers[1][0], np.float32(self.ref[1][0]))

    def test_model_checking_prompt_shows_best_verified_network_when_worse(self):
        agent = make_agent(self.env, JOBS, wrong_action_penalty=1000, max_steps=100)
        zero = [(np.zeros((o, n)), np.zeros(o)) for o, n in agent.shapes]
        agent.set_weights(self.ref)
        agent.notes = "REFERENCE-NOTES"
        fake = FakeLLM([answer(zero, notes="zero"), answer(self.ref)])
        agent._chat = fake
        mc = self.env.storm_bridge.model_checker
        res, info = mc.induced_markov_chain(agent, None, self.env, T_CONST, JOBS)
        agent.model_checking_learn(res, info, mc)                  # 1.0 = best so far
        self.assertNotIn("Best network verified so far", fake.prompts[0])
        agent.episodic_learn()                                     # zero network applied
        res, info = mc.induced_markov_chain(agent, None, self.env, T_CONST, JOBS)
        self.assertEqual(res, 0.0)
        agent.model_checking_learn(res, info, mc)
        ref_json = agent._layers_json([(np.float32(W), np.float32(b)) for W, b in self.ref])
        self.assertIn("## Best network verified so far (Pmax=? [ F jobs_done=2 ] = 1)\n" + ref_json, fake.prompts[1])
        self.assertIn("REFERENCE-NOTES", fake.prompts[1])

    def test_dataset_prompt_shows_most_accurate_network_when_worse(self):
        agent = make_agent(self.env, JOBS)
        zero = [(np.zeros((o, n)), np.zeros(o)) for o, n in agent.shapes]
        agent.set_weights(self.ref)
        X = np.array([[0, 10, 0, 0, 0, 0, 0, 4, 2, 2], [0, 6, 0, 0, 0, 0, 0, 4, 0, 4]], dtype=np.float32)
        y = np.array([agent.action_names.index("NORTH"), agent.action_names.index("PICK_UP")])
        fake = FakeLLM([answer(zero), answer(zero)])
        agent._chat = fake
        result = agent.behavioral_cloning(self.env, {"X_train": X, "y_train": y}, epochs=2, accuracy_threshold=101)
        self.assertEqual(result[:2], (0, 100.0))
        self.assertNotIn("Most accurate network so far", fake.prompts[0])
        self.assertIn("## Most accurate network so far (accuracy 100.00%)", fake.prompts[1])
        np.testing.assert_array_equal(agent.get_weights()[0][0], np.float32(self.ref[0][0]))  # best kept

    def test_saved_weights_json_is_exact(self):
        self.agent.set_weights(self.ref)
        saved = json.loads(self.agent._weights_json(decimals=None))["layers"]
        for (W, b), s in zip(self.agent.get_weights(), saved):
            np.testing.assert_array_equal(np.float32(s["W"]), W)
            np.testing.assert_array_equal(np.float32(s["b"]), b)

    def test_dataset_round_uses_label_sets(self):
        self.agent.set_weights(self.ref)
        X = np.array([[0, 10, 0, 0, 0, 0, 0, 4, 2, 2]] * 2, dtype=np.float32)
        acc = self.agent._accuracy(X, np.array([self.agent.action_names.index("WEST"),
                                                self.agent.action_names.index("NORTH")]))[0]
        self.assertEqual(acc, 100.0)  # reference chooses NORTH, which is one of the two labels


class TestTransporterDescription(unittest.TestCase):
    """Every claim of prism_files/transporter_wsp.md, checked on the Storm MDP."""

    @classmethod
    def setUpClass(cls):
        cls.model, cls.prog, cls.states = build_mdp(TRANSPORTER, T_CONST)

    def test_start_state(self):
        val = self.states[self.model.initial_states[0]][0]
        self.assertEqual({k: val[k] for k in ("x", "y", "fuel", "passenger", "jobs_done", "done")},
                         {"x": 2, "y": 2, "fuel": 10, "passenger": False, "jobs_done": 0, "done": False})

    def test_dynamics(self):
        moves = {"NORTH": (0, 1), "SOUTH": (0, -1), "EAST": (1, 0), "WEST": (-1, 0)}
        corners = [(0, 4), (0, 0), (4, 4), (3, 0)]
        core = ("x", "y", "fuel", "passenger", "jobs_done", "done")
        by_core = {}
        for sid, (v, choices, labels) in self.states.items():
            gas = (v["x"], v["y"]) == (1, 2)
            self.assertEqual("empty" in labels, v["fuel"] == 0 and not gas, v)
            key = tuple(v[k] for k in core)
            enabled = frozenset(c for c in choices if c is not None)
            self.assertEqual(by_core.setdefault(key, enabled), enabled, "loc/dest must not affect guards")

            def succ(c):
                return [(self.states[t][0], pr) for t, pr in choices[c]]

            if gas:
                self.assertEqual(enabled, set(moves))
                for c, (dx, dy) in moves.items():
                    (n, pr), = succ(c)
                    self.assertEqual((n["x"], n["y"], n["fuel"]), (v["x"] + dx, v["y"] + dy, 10))
            elif v["fuel"] == 0:
                self.assertEqual(enabled, set(moves) | {"PICK_UP", "DROP"})
                for c in enabled:
                    (n, pr), = succ(c)
                    self.assertEqual(n, dict(v, done=True))
            elif v["done"] or v["jobs_done"] == 2:
                self.assertEqual(enabled, frozenset(), v)
            else:
                expect = set(moves)
                if (v["x"], v["y"]) == (0, 4) and not v["passenger"]:
                    expect.add("PICK_UP")
                if (v["x"], v["y"]) == (0, 0) and v["passenger"]:
                    expect.add("DROP")
                self.assertEqual(enabled, expect, v)
                for c, (dx, dy) in moves.items():
                    (n, pr), = succ(c)
                    x, y = v["x"] + dx, v["y"] + dy
                    if not (0 <= x <= 4 and 0 <= y <= 4):
                        x, y = v["x"], v["y"]
                    self.assertEqual(n, dict(v, x=x, y=y, fuel=v["fuel"] - 1))
                if "PICK_UP" in expect:
                    (n, pr), = succ("PICK_UP")
                    self.assertEqual(n, dict(v, passenger=True, fuel=v["fuel"] - 1))
                if "DROP" in expect:
                    outs = succ("DROP")
                    self.assertEqual(len(outs), 16)
                    combos = set()
                    for n, pr in outs:
                        self.assertAlmostEqual(pr, 1 / 16)
                        combos.add(((n["passenger_loc_x"], n["passenger_loc_y"]),
                                    (n["passenger_dest_x"], n["passenger_dest_y"])))
                        rest = {k: n[k] for k in core}
                        self.assertEqual(rest, dict({k: v[k] for k in core}, passenger=False,
                                                    jobs_done=v["jobs_done"] + 1, fuel=v["fuel"] - 1))
                    self.assertEqual(combos, set(product(corners, corners)))

    def test_one_job_from_start_empties_the_tank(self):
        env = SafeGym(TRANSPORTER, T_CONST, 100, 1000, False, 0, "")
        idx = {a: i for i, a in enumerate(env.action_mapper.actions)}
        names = [n for n, _ in sorted(env.storm_bridge.state_mapper.mapper.items(), key=lambda kv: kv[1])]
        env.reset()
        for a in ["NORTH", "NORTH", "WEST", "WEST", "PICK_UP", "SOUTH", "SOUTH", "SOUTH", "SOUTH", "DROP"]:
            s, r, done, _, _ = env.step(idx[a])
            self.assertNotEqual(r, -1000, a)
        d = dict(zip(names, s.tolist()))
        self.assertEqual((d["x"], d["y"], d["fuel"], d["jobs_done"], d["passenger"]), (0, 0, 0, 1, 0))

    def test_two_jobs_need_two_refuels(self):
        # 0-1 BFS: cost 1 for every action taken on the gas station (= leaving it with a full tank).
        init = self.model.initial_states[0]
        dist = {init: 0}
        queue = deque([init])
        while queue:
            sid = queue.popleft()
            v, choices, _ = self.states[sid]
            w = 1 if (v["x"], v["y"]) == (1, 2) else 0
            for succ in choices.values():
                for t, _ in succ:
                    if dist[sid] + w < dist.get(t, 1 << 30):
                        dist[t] = dist[sid] + w
                        (queue.appendleft if w == 0 else queue.append)(t)
        best = min(d for sid, d in dist.items() if self.states[sid][0]["jobs_done"] == 2)
        self.assertEqual(best, 2)

    def test_best_achievable_values(self):
        self.assertAlmostEqual(check_property(self.prog, TRANSPORTER, T_CONST, JOBS), 1.0)
        self.assertAlmostEqual(check_property(self.prog, TRANSPORTER, T_CONST, EMPTY), 0.0)


class TestFrozenLakeDescription(unittest.TestCase):
    """Every claim of prism_files/frozen_lake_wsp.md, checked on the Storm MDP."""

    def test_dynamics(self):
        model, _, states = build_mdp(FROZEN_LAKE, F_CONST)
        control = 0.33
        self.assertEqual(states[model.initial_states[0]][0]["pos"], 0)
        delta = {"a3_up": (-1, 0), "a1_down": (1, 0), "a0_left": (0, -1), "a2_right": (0, 1)}
        perp = {"a3_up": ("a0_left", "a2_right"), "a1_down": ("a0_left", "a2_right"),
                "a0_left": ("a3_up", "a1_down"), "a2_right": ("a3_up", "a1_down")}

        def move(pos, a):
            r, c = divmod(pos, 4)
            r2, c2 = r + delta[a][0], c + delta[a][1]
            return r2 * 4 + c2 if 0 <= r2 < 4 and 0 <= c2 < 4 else pos

        seen = set()
        for sid, (v, choices, _) in states.items():
            pos = v["pos"]
            if pos == 16:
                continue
            seen.add(pos)
            self.assertEqual(set(choices), set(delta))
            for a, succ in choices.items():
                got = {}
                for t, pr in succ:
                    got[states[t][0]["pos"]] = got.get(states[t][0]["pos"], 0) + pr
                if pos in (5, 7, 11, 12, 15):
                    expected = {16: 1.0}
                else:
                    expected = {}
                    for target, pr in [(move(pos, a), control), (move(pos, perp[a][0]), (1 - control) / 2),
                                       (move(pos, perp[a][1]), (1 - control) / 2)]:
                        expected[target] = expected.get(target, 0) + pr
                self.assertEqual(set(got), set(expected), (pos, a))
                for k in expected:
                    self.assertAlmostEqual(got[k], expected[k], places=9, msg=(pos, a, k))
        self.assertGreaterEqual(len(seen), 10)


if __name__ == "__main__":
    unittest.main()
