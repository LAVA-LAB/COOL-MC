class PrismEncoder:

    def __init__(self, interconnections):
        self.interconnections = interconnections

    def get_feature_min_dictionary(self, interconnections):
        feature_mins = {}
        for interconnection in interconnections:
            for idx, feature in enumerate(interconnection[4]):
                if feature not in feature_mins:
                    feature_mins[feature] = interconnection[0][idx]
                else:
                    if interconnection[0][idx] < feature_mins[feature]:
                        feature_mins[feature] = interconnection[0][idx]
        for next_states in interconnections:
            for next_state in next_states[3]:
                for idx, feature in enumerate(interconnection[4]):
                    if feature not in feature_mins:
                        feature_mins[feature] = next_state[0][idx]
                    else:
                        if next_state[0][idx] < feature_mins[feature]:
                            feature_mins[feature] = next_state[idx]
        # New feature
        for interconnection in interconnections:
            n_feature = interconnection[-2]
            if "n_feature" not in feature_mins:
                feature_mins["n_feature"] = n_feature
            else:
                if n_feature < feature_mins["n_feature"]:
                    feature_mins["n_feature"] = n_feature
        #print(feature_mins)
        return feature_mins

    def get_feature_max_dictionary(self, interconnections):
        feature_maxs = {}
        for interconnection in interconnections:
            for idx, feature in enumerate(interconnection[4]):
                if feature not in feature_maxs:
                    feature_maxs[feature] = interconnection[0][idx]
                else:
                    if interconnection[0][idx] > feature_maxs[feature]:
                        feature_maxs[feature] = interconnection[0][idx]
        for next_states in interconnections:
            for next_state in next_states[3]:
                for idx, feature in enumerate(interconnection[4]):
                    if feature not in feature_maxs:
                        feature_maxs[feature] = next_state[0][idx]
                    else:
                        if next_state[0][idx] > feature_maxs[feature]:
                            feature_maxs[feature] = next_state[idx]
        # New feature
        for interconnection in interconnections:
            n_feature = interconnection[-2]
            if "n_feature" not in feature_maxs:
                feature_maxs["n_feature"] = n_feature
            else:
                if n_feature > feature_maxs["n_feature"]:
                    feature_maxs["n_feature"] = n_feature
        #print(feature_maxs)
        return feature_maxs

    def get_init_values(self, interconnections):
        feature_inits = {}
        for interconnection in interconnections:
            for idx, feature in enumerate(interconnection[4]):
                if feature not in feature_inits:
                    feature_inits[feature] = interconnection[0][idx]
            
            #print(feature_inits)
            # New feature init value
            feature_inits["n_feature"] = interconnections[0][-2]
            break
        return feature_inits
            
    def get_guard(self, state, feature_names):
        guard = ""
        for idx, feature_name in enumerate(feature_names):
            guard += f"{feature_name} = {state[idx]}"
            if idx < len(feature_names)-1:
                guard += " & "
         
        return guard

    def get_update(self, next_state, feature_names):
        update = ""
        for idx, feature_name in enumerate(feature_names):
            update += f"({feature_name}' = {next_state[idx]})"
            if idx < len(feature_names)-1:
                update += " & "
        return update

    def fix_min_max(self, feature_mins, feature_maxs):
        for feature_name in feature_mins.keys():
            if feature_mins[feature_name] == feature_maxs[feature_name]:
                feature_maxs[feature_name] += 1
                feature_mins[feature_name] -= 1
        return feature_mins, feature_maxs

    
    def encode(self):
        prism_str = "dtmc"
        prism_str += "\n\nmodule extended_model\n"
        # Initialization of the features
        feature_mins = self.get_feature_min_dictionary(self.interconnections)
        feature_maxs = self.get_feature_max_dictionary(self.interconnections)
        feature_mins, feature_maxs = self.fix_min_max(feature_mins, feature_maxs)
        feature_inits = self.get_init_values(self.interconnections)
        for feature_name in feature_mins.keys():
            prism_str += f"\n\t  {feature_name} : [{feature_mins[feature_name]}..{feature_maxs[feature_name]}] init {feature_inits[feature_name]};"
        prism_str += "\n\n"
        # Commands
        for interconnection in self.interconnections:
            # np_state, action_idx, action_name, all_next_states, env.storm_bridge.state_mapper.get_feature_names()
            state = interconnection[0] #+ [interconnection[-2]]
            action_idx = interconnection[1]
            action_name = interconnection[2]
            all_next_states = interconnection[3]
            feature_names = interconnection[4]
            #feature_names.append("n_feature")
            # [north] NOT_NORTH_BORDER & IS_NOT_DONE & NOT_COLLISION-> (1-slickness) : (y'=y+1) + slickness : true;
            prism_str += f"\t[] {self.get_guard(state, feature_names)} ->"
            for idx, next_state in enumerate(all_next_states):
                prism_str += f" {next_state[1]} : {self.get_update(next_state[0], feature_names)} & (n_feature'={interconnection[-1][idx]}) +"
            prism_str = prism_str[:-1]
            prism_str += ";\n"
        prism_str += "endmodule"
        self.prism_code = prism_str

    def to_file(self, filepath):
        with open(filepath, "w") as file:
            file.write(self.prism_code)