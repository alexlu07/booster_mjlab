"""Read-only critic and AMP diagnostics for checkpoint playback."""

from contextlib import contextmanager

import torch


VALUE = "Critic V(s)"
SCORE = "Discriminator score"
RAW_STYLE = "Style reward (raw)"
WEIGHTED_STYLE = "Style reward (weighted per step)"


class DiagnosticPolicy:
    """Keep inference-only AMP history separate from training/replay state.

    Samples describe the observations before each action. The policy still
    receives exactly its original actor observations; privileged critic and AMP
    observations are used only for the diagnostics.
    """

    def __init__(self, policy, runner, *, has_critic, has_discriminator):
        self.policy = policy
        self.runner = runner
        self.env = runner.env.unwrapped
        self.alg = runner.alg
        self.has_critic = has_critic
        amp = getattr(self.alg, "amp", None)
        self.discriminator = amp.discriminator if amp is not None and has_discriminator else None
        if self.discriminator is not None:
            self.discriminator.eval()
        self.history_length = runner.cfg.get("num_amp_obs_steps", 1)
        self._history = None
        self.latest = {}
        self.episode_start = None
        self.is_sac = hasattr(self.alg.critic, "qnets")
        self.term_names = []
        if has_critic:
            self.alg.critic.eval()
            self.term_names += (
                [f"Critic Q{i + 1}(s, a)" for i in range(len(self.alg.critic.qnets))]
                if self.is_sac else [VALUE]
            )
        if self.discriminator is not None:
            self.term_names += [SCORE, RAW_STYLE, WEIGHTED_STYLE]

    @torch.no_grad()
    def __call__(self, obs):
        actions = self.policy(obs)
        model_obs = obs.to(self.runner.device)
        self.episode_start = self.env.episode_length_buf == 0
        values = []
        if self.has_critic:
            if self.is_sac:
                groups = self.alg._critic_obs_group_names
                critic_obs = torch.cat([model_obs[key] for key in groups], dim=-1)
                normalized = self.alg.critic_obs_normalizer(critic_obs, update=False)
                logits = self.alg.critic(normalized, actions.to(self.runner.device))
                values.extend(self.alg.critic.get_value(logits.softmax(dim=-1)).unbind(0))
            else:
                self.alg.critic.reset(self.episode_start.to(self.runner.device))
                values.append(self.alg.critic(model_obs).reshape(-1))

        if self.discriminator is not None:
            frame = torch.cat([model_obs[key] for key in self.runner.amp_group_names], dim=-1)
            if self._history is None:
                self._history = frame[:, None].expand(-1, self.history_length, -1).clone()
            else:
                self._history = torch.cat((self._history[:, 1:], frame[:, None]), dim=1)
                reset = self.episode_start.to(frame.device)
                self._history[reset] = frame[reset, None]
            history = self._history.flatten(1)
            score = self.discriminator(history).reshape(-1)
            raw_style = self.discriminator.predict_reward(history).reshape(-1)
            weighted_style = (
                raw_style * self.runner._compute_style_weight()
                * self.runner._compute_style_reward_scale()
            )
            values.extend((score, raw_style, weighted_style))
        if values:
            # One device transfer for all curves, not one sync per scalar.
            data = torch.stack(values, dim=-1).detach().cpu()
            self.latest = dict(zip(self.term_names, data.unbind(-1), strict=True))
        return actions

    def reset(self, *args, **kwargs):
        self._history = None
        self.latest = {}
        reset = getattr(self.policy, "reset", None)
        if reset is not None:
            reset(*args, **kwargs)


def diagnostics_runner_class(runner_class):
    """Enable saved critic weights and diagnostic inference only inside play."""

    class RunnerWithDiagnostics(runner_class):
        def load(self, path, load_cfg=None, strict=True, map_location=None):
            # Inspect model availability rather than plotting randomly initialized
            # networks when playing an actor-only or non-AMP checkpoint.
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            has_critic = "critic_state_dict" in checkpoint or "model_state_dict" in checkpoint
            has_discriminator = "discriminator_state_dict" in checkpoint
            del checkpoint
            if load_cfg is not None:
                load_cfg = {**load_cfg, "critic": has_critic}
            result = super().load(path, load_cfg, strict, map_location)
            self._diagnostic_models = has_critic, has_discriminator
            return result

        def get_inference_policy(self, device=None):
            policy = super().get_inference_policy(device)
            has_critic, has_discriminator = self._diagnostic_models
            return DiagnosticPolicy(
                policy, self, has_critic=has_critic, has_discriminator=has_discriminator
            )

    return RunnerWithDiagnostics


@contextmanager
def diagnostic_reward_plots(get_policy):
    """Extend the existing Rewards selector, leaving reward bars unchanged.

    Scope the factory replacement to viewer setup; the resulting plotter uses
    mjlab's checkbox/filter/history/plot behavior and lifecycle unchanged.
    """
    from mjlab.viewer.viser import overlays

    original_plotter = overlays.ViserTermPlotter

    class DiagnosticTermPlotter(original_plotter):
        def __init__(self, server, term_names, name="Reward", **kwargs):
            self._include_diagnostics = name == "Reward"
            self._last_diagnostic_policy = None
            policy = get_policy()
            if self._include_diagnostics and isinstance(policy, DiagnosticPolicy):
                term_names = [*term_names, *policy.term_names]
            super().__init__(server, term_names, name=name, **kwargs)

        def update(self, terms):
            policy = get_policy()
            if self._include_diagnostics and isinstance(policy, DiagnosticPolicy):
                if policy is not self._last_diagnostic_policy:
                    self.clear_histories()
                    self._last_diagnostic_policy = policy
                terms = [*terms, *[
                    (name, [float(values[self._env_idx])])
                    for name, values in policy.latest.items()
                ]]
            super().update(terms)

    overlays.ViserTermPlotter = DiagnosticTermPlotter
    try:
        yield
    finally:
        overlays.ViserTermPlotter = original_plotter
