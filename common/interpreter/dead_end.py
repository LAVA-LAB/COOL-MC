import csv
import numpy as np
import stormpy
from common.interpreter.interpreter import Interpreter


class DeadEndInterpreter(Interpreter):
    """
    Dead-End State Interpreter for MDP analysis.

    Identifies states in the MDP from which target goal states (e.g., "survival")
    can no longer be reached, regardless of which actions are taken. This analysis
    is performed on the full MDP without considering any trained policy.

    Based on "Medical Dead-ends and Learning to Identify High-risk States and Treatments"
    (Fatemi et al., 2022).

    Config format: "dead_end;property_formula[;output_file.csv]"

    Examples:
        "dead_end;Pmin=? [F \"survival\"]"
        "dead_end;Pmax=? [F \"goal\"];dead_end_results.csv"
    """

    def __init__(self, config):
        super().__init__(config)
        parts = config.split(";")
        if len(parts) < 2:
            raise ValueError(
                "Dead-end interpreter requires property formula. "
                "Format: dead_end;property_formula[;output_file.csv]")
        self.property_formula = parts[1].strip()
        self.output_file = parts[2].strip() if len(parts) > 2 and parts[2].strip() else ""

    def _extract_goal_label(self, property_formula):
        """Extract the goal label from a property formula.

        E.g., 'Pmin=? [F "survival"]' -> "survival"
              'Pmax=? [F "death"]' -> "death"
        """
        import re
        # Match patterns like [F "label"], [ F "label"], [F 'label'], etc.
        # Allow optional whitespace around brackets and after F
        match = re.search(r'\[\s*F\s+["\']([^"\']+)["\']\s*\]', property_formula)
        if match:
            return match.group(1)
        raise ValueError(
            f"Could not extract goal label from property: {property_formula}\n"
            f"Expected format like: Pmin=? [F \"survival\"] or Pmin=? [ F \"survival\" ]")

    def _build_full_mdp(self, env):
        """Build the full MDP from the PRISM file without any policy."""
        prism_path = env.storm_bridge.path
        constant_defs = env.storm_bridge.constant_definitions

        print(f"Building MDP from: {prism_path}")
        if constant_defs:
            print(f"Constants: {constant_defs}")

        # Parse PRISM program
        program = stormpy.parse_prism_program(prism_path)

        # Apply constant definitions if provided
        if constant_defs:
            program = stormpy.preprocess_symbolic_input(
                program, [], constant_defs)[0].as_prism_program()

        # Parse property to ensure it's valid
        properties = stormpy.parse_properties(self.property_formula, program)

        # Build the full MDP
        model = stormpy.build_model(program, properties)

        return model, properties[0]

    def _find_goal_states(self, model, goal_label):
        """Find all states labeled with the goal label."""
        if not model.labeling.contains_label(goal_label):
            available_labels = sorted(model.labeling.get_labels())
            raise ValueError(
                f"Label '{goal_label}' not found in model.\n"
                f"Available labels: {available_labels}")

        goal_states = set()
        for state_id in range(model.nr_states):
            if model.labeling.has_state_label(goal_label, state_id):
                goal_states.add(state_id)

        return goal_states

    def _compute_reachability(self, model, goal_states):
        """Compute which states can reach the goal states.

        Returns:
            reachable_states: set of state IDs that can reach goal states
            dead_end_states: set of state IDs that cannot reach goal states
        """
        # Use backward reachability: start from goal states and find all predecessors
        reachable = set(goal_states)
        frontier = list(goal_states)

        # Build predecessor map: state -> set of predecessor states
        predecessors = {i: set() for i in range(model.nr_states)}

        for state_id in range(model.nr_states):
            state = model.states[state_id]
            for action in state.actions:
                for transition in action.transitions:
                    target_id = transition.column
                    predecessors[target_id].add(state_id)

        # Backward search from goal states
        while frontier:
            current = frontier.pop()
            for pred in predecessors[current]:
                if pred not in reachable:
                    reachable.add(pred)
                    frontier.append(pred)

        # Dead-end states are those not reachable
        all_states = set(range(model.nr_states))
        dead_end_states = all_states - reachable

        return reachable, dead_end_states

    def _get_state_features(self, env, model, state_id):
        """Extract feature values for a given state ID in the model."""
        # Get state valuation from the model's state valuations
        state_valuations = model.state_valuations
        state_valuation = state_valuations.get_state(state_id)

        # Convert to dictionary
        state_dict = {}
        for var in state_valuation:
            state_dict[var.name] = var.value

        # Parse into numpy array using the environment's state mapper
        import json
        state_array = env.storm_bridge.parse_state(json.dumps(state_dict))

        return state_array

    def _get_feature_names(self, env):
        """Get feature names, preferring compressed names."""
        state_mapper = env.storm_bridge.state_mapper
        if state_mapper.has_compressed_state_representation():
            names = state_mapper.get_compressed_feature_names()
            if names is not None:
                return names
        return state_mapper.get_feature_names()

    def _analyze_dead_end_actions(self, model, dead_end_states, env):
        """Analyze which actions most frequently lead to dead-end states."""
        action_to_dead_end_count = {}
        action_labels = env.action_mapper.actions if hasattr(env, 'action_mapper') else None

        # Count transitions to dead-end states per action
        for state_id in range(model.nr_states):
            if state_id in dead_end_states:
                continue  # Skip dead-end states themselves

            state = model.states[state_id]
            for action_idx, action in enumerate(state.actions):
                for transition in action.transitions:
                    target_id = transition.column
                    if target_id in dead_end_states:
                        # This action leads to a dead-end
                        if action_labels and action_idx < len(action_labels):
                            action_name = action_labels[action_idx]
                        else:
                            action_name = f"action_{action_idx}"

                        if action_name not in action_to_dead_end_count:
                            action_to_dead_end_count[action_name] = 0
                        action_to_dead_end_count[action_name] += 1

        return action_to_dead_end_count

    def interpret(self, env, rl_agent, model_checking_info):
        """Main interpretation method."""

        print(f"\n{'='*80}")
        print("Dead-End State Interpreter (MDP Structure Analysis)")
        print(f"{'='*80}")
        print(f"Property: {self.property_formula}")

        # Extract goal label from property
        goal_label = self._extract_goal_label(self.property_formula)
        print(f"Goal label: '{goal_label}'")

        # Build the full MDP
        print(f"\n{'='*80}")
        print("Building Full MDP (without policy)")
        print(f"{'='*80}")
        model, property_obj = self._build_full_mdp(env)

        print(f"MDP Statistics:")
        print(f"  States:      {model.nr_states}")
        print(f"  Transitions: {model.nr_transitions}")
        print(f"  Choices:     {model.nr_choices}")
        print(f"  Model type:  {model.model_type}")
        print(f"  Labels:      {sorted(model.labeling.get_labels())}")

        # Find goal states
        print(f"\n{'='*80}")
        print(f"Finding Goal States (labeled '{goal_label}')")
        print(f"{'='*80}")
        goal_states = self._find_goal_states(model, goal_label)
        print(f"Goal states found: {len(goal_states)}")
        print(f"Percentage:        {len(goal_states)/model.nr_states*100:.2f}%")

        # Compute reachability
        print(f"\n{'='*80}")
        print(f"Computing Reachability Analysis")
        print(f"{'='*80}")
        reachable_states, dead_end_states = self._compute_reachability(model, goal_states)

        # Statistics
        print(f"\n{'='*80}")
        print(f"DEAD-END STATE STATISTICS")
        print(f"{'='*80}")
        print(f"Total states:           {model.nr_states}")
        print(f"Goal states:            {len(goal_states):>8d} ({len(goal_states)/model.nr_states*100:>6.2f}%)")
        print(f"Reachable states:       {len(reachable_states):>8d} ({len(reachable_states)/model.nr_states*100:>6.2f}%)")
        print(f"Dead-end states:        {len(dead_end_states):>8d} ({len(dead_end_states)/model.nr_states*100:>6.2f}%)")

        if len(dead_end_states) == 0:
            print(f"\n✓ No dead-end states found! All states can reach '{goal_label}'.")
            return

        print(f"\n⚠ Warning: {len(dead_end_states)} states cannot reach '{goal_label}' regardless of actions taken!")

        # Analyze actions leading to dead-ends
        print(f"\n{'='*80}")
        print(f"Actions Leading to Dead-End States")
        print(f"{'='*80}")
        action_dead_end_counts = self._analyze_dead_end_actions(model, dead_end_states, env)

        if action_dead_end_counts:
            # Sort by count descending
            sorted_actions = sorted(action_dead_end_counts.items(),
                                  key=lambda x: x[1], reverse=True)
            print(f"{'Action':<20s}  {'Transitions to Dead-End':>25s}")
            print("-" * 48)
            for action_name, count in sorted_actions[:10]:  # Top 10
                print(f"{action_name:<20s}  {count:>25d}")
        else:
            print("No actions lead from reachable to dead-end states.")

        # Save detailed results to CSV
        if self.output_file or True:  # Always save
            csv_path = self.output_file if self.output_file else 'dead_end_results.csv'
            self._save_csv(env, model, dead_end_states, goal_states,
                          reachable_states, csv_path)

    def _save_csv(self, env, model, dead_end_states, goal_states,
                  reachable_states, csv_path):
        """Save dead-end state information to CSV."""
        feature_names = self._get_feature_names(env)

        print(f"\n{'='*80}")
        print(f"Saving Results to CSV")
        print(f"{'='*80}")

        fieldnames = ['state_id', 'state_type'] + feature_names

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            # Write all states with their classification
            for state_id in range(model.nr_states):
                if state_id in goal_states:
                    state_type = 'goal'
                elif state_id in dead_end_states:
                    state_type = 'dead_end'
                else:
                    state_type = 'reachable'

                # Get state features
                try:
                    state_features = self._get_state_features(env, model, state_id)
                    row = {'state_id': state_id, 'state_type': state_type}
                    for i, fname in enumerate(feature_names):
                        if i < len(state_features):
                            row[fname] = float(state_features[i])
                    writer.writerow(row)
                except Exception as e:
                    # If we can't extract features for some reason, skip
                    print(f"Warning: Could not extract features for state {state_id}: {e}")
                    continue

        print(f"Results saved to: {csv_path}")
        print(f"Columns: state_id, state_type, {', '.join(feature_names)}")
