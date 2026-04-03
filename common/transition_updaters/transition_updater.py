class TransitionUpdater:
    """Base class for transition updaters.

    Transition updaters run after the induced DTMC/MDP is built and replace it
    with a new model — typically an interval model (SparseIntervalDtmc /
    SparseIntervalMdp).

    The PCTL property is specified via the CLI --prop flag. When the resulting
    model is an interval model, the model checker automatically checks both
    Pmax=? and Pmin=? bounds.

    Config parsing convention:
        parts[0]  = updater name
        parts[1:] = key=value parameters
    """

    def __init__(self, config_str: str):
        self.config_str = config_str
        self.parse_config(config_str)

    def parse_config(self, config_str: str) -> None:
        """Override in subclass to parse parameters."""
        pass

    @property
    def requires_full_mdp(self) -> bool:
        """If True, the model_checker will build the full MDP with Storm's native
        builder (fast) instead of the incremental policy-induction path (slow).
        The updater then receives the full MDP with all actions and is responsible
        for producing an interval model itself (e.g. via policy sampling).
        """
        return False

    def update_model(self, model, env, agent):
        """Transform the induced model, returning a (possibly new) stormpy model.

        The default is a no-op. Subclasses should return a SparseIntervalDtmc
        or SparseIntervalMdp as appropriate.

        Args:
            model: Induced SparseDtmc or SparseMdp built by Storm.
            env:   SafeGym environment.
            agent: RL agent.

        Returns:
            A stormpy model valid for stormpy.model_checking().
        """
        return model

    def get_updater_name(self) -> str:
        return "base"
