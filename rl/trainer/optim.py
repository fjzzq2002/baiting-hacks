"""The optimizer. Routed runs use SplitMomentAdamW: m from the routed gradient, v from the pre-routing one."""

from split_moment import SplitMomentAdamW, clip_pre_routing_grads_


class OptimizerMixin:
    def create_optimizer(self):
        if self.optimizer is None and self.gradient_routing_enabled:
            args = self.args
            lr = args.learning_rate
            groups = [{"params": [p for p in params if p.requires_grad], "lr": lr,
                       "weight_decay": args.weight_decay, "graft_role": role}
                      for role, params in (("retain", self._retain_params), ("forget", self._forget_params))]
            self.optimizer = SplitMomentAdamW(groups, lr=lr, betas=(args.adam_beta1, args.adam_beta2),
                                              eps=args.adam_epsilon, weight_decay=args.weight_decay)
            # Survives the accelerator's wrapping, for `set_window`.
            self._split_moment_optimizer = self.optimizer
            self._clip_both_moments()
        return super().create_optimizer()

    def _clip_both_moments(self):
        """Apply each gradient clip to v's pre-routing gradient too, with the same coefficient."""
        clip = self.accelerator.clip_grad_norm_

        def clip_with_v(parameters, max_norm, *args, **kwargs):
            total_norm = clip(parameters, max_norm, *args, **kwargs)
            clip_pre_routing_grads_(self.optimizer.param_groups, max_norm, total_norm)
            return total_norm

        self.accelerator.clip_grad_norm_ = clip_with_v
