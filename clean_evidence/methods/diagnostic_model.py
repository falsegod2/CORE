"""Read-only, fixed-replay diagnostics for the Evidence world model.

Only this optional assay computes ablated objectives. Ordinary updates continue
to use EvidenceWorldModel._auxiliary_losses, with their original weights and
random-start sampling. These probes are not held-out environment evaluations.
"""
from __future__ import annotations

import contextlib
import math

import torch
from torch.nn import functional as F


@contextlib.contextmanager
def _parameter_grad_flags(module, enabled):
    flags = [(parameter, parameter.requires_grad) for parameter in module.parameters()]
    try:
        if enabled:
            for parameter, _ in flags:
                parameter.requires_grad_(True)
        yield
    finally:
        for parameter, flag in flags:
            parameter.requires_grad_(flag)


def _ratio(stats, name, values, mask, weight=1.0):
    """Store sufficient statistics; empty populations keep denominator zero."""
    values, mask = values.detach(), mask.detach().bool()
    numerator = (values * mask).sum().double() * float(weight)
    denominator = mask.sum().double() * float(weight)
    stats[name + '_numerator'] = stats.get(name + '_numerator', 0.0) + numerator
    stats[name + '_denominator'] = stats.get(name + '_denominator', 0.0) + denominator


def _fixed_starts(model, data, limit):
    """Choose evenly spaced valid starts without consuming the training RNG."""
    valid = data['loss_mask'].bool() & data['sequence_valid'].bool()
    valid &= ~data['is_terminal'].bool() & ~data.get('is_last', torch.zeros_like(valid)).bool()
    rows = []
    for batch in range(valid.shape[0]):
        times = valid[batch].nonzero(as_tuple=False).flatten()
        if not times.numel():
            continue
        candidates = torch.stack([torch.full_like(times, batch), times], -1)
        masks = model._valid_prefixes(data, candidates, model.options['horizons'])
        active = torch.as_tensor(model.options['horizon_weights'], device=times.device) > 0
        candidates = candidates[(masks & active[None]).any(-1)]
        if candidates.shape[0] > limit:
            indices = torch.linspace(0, candidates.shape[0] - 1, limit, device=times.device).round().long()
            candidates = candidates[indices]
        if candidates.numel():
            rows.append(candidates)
    return torch.cat(rows, 0) if rows else torch.empty(0, 2, dtype=torch.long, device=valid.device)


def _memory_capture(dynamics, prefix):
    if hasattr(dynamics, 'capture_memory_diagnostics'):
        return dynamics.capture_memory_diagnostics(prefix=prefix)
    return contextlib.nullcontext({})


def _recorded_probe(model, post, data, starts):
    from .evidence_model import _freeze_parameters

    options = model.options
    zero = post['deter'].new_zeros(())
    losses = {'fc': zero, 'rc': zero}
    stats = {'starts': float(starts.shape[0]), 'mc_samples': float(options['mc_samples'])}
    horizons = options['horizons']
    for scope in ['all_horizons', *['h' + str(k) for k in horizons]]:
        for population in ('all', 'nonzero', 'zero'):
            name = 'rc_' + scope + '_' + population
            stats[name + '_prefixes'] = zero
            for quantity in ('prediction_mean', 'target_mean', 'bias', 'mae', 'mse'):
                stats[name + '_' + quantity + '_numerator'] = zero
                stats[name + '_' + quantity + '_denominator'] = zero
    for name in ('fc_categorical', 'fc_hidden', 'fc_heat'):
        stats[name + '_numerator'] = zero
        stats[name + '_denominator'] = zero
    stats['fc_heat_available'] = float(model.task_heatmap is not None and 'heatmap' in data and 'heatmap_valid' in data)
    if not starts.numel():
        return losses, stats

    prefix = model._valid_prefixes(data, starts, horizons)
    active = [k for j, k in enumerate(horizons) if bool(prefix[:, j].any()) and options['horizon_weights'][j] > 0]
    if not active:
        return losses, stats
    batch, times = starts[:, 0], starts[:, 1]
    origin = {key: value[batch, times].detach() for key, value in post.items()}
    teacher = model._teacher(data) if stats['fc_heat_available'] else None
    teacher_valid = data.get('heatmap_valid', torch.zeros_like(data['is_first'])).bool()
    fc_sum, fc_count = zero, zero
    predictions = {k: [] for k in active}
    for _ in range(options['mc_samples']):
        state = origin
        cumulative = zero.expand(starts.shape[0])
        for offset in range(1, max(active) + 1):
            endpoint = (times + offset).clamp(max=data['is_first'].shape[1] - 1)
            # Rollout arithmetic remains identical to the training estimator.
            # Telemetry only counts rows contributing to a scored prefix;
            # padded and post-terminal computational rows are excluded.
            diagnostic_rows = torch.zeros_like(prefix[:, 0])
            for j, horizon in enumerate(horizons):
                if horizon >= offset and options['horizon_weights'][j] > 0:
                    diagnostic_rows |= prefix[:, j]
            with model.dynamics.memory_diagnostic_mask(diagnostic_rows):
                state = model.dynamics.img_step(state, data['action'][batch, endpoint])
            frozen = _freeze_parameters(model.heads['task_reward']) if options['freeze_rc_reward_head'] else contextlib.nullcontext()
            with frozen:
                prediction = model.heads['task_reward'](model.dynamics.get_feat(state)).mode()[..., 0]
            cumulative = cumulative + model._config.discount ** (offset - 1) * prediction
            if offset not in active:
                continue
            j = horizons.index(offset)
            mask, weight = prefix[:, j], float(options['horizon_weights'][j])
            predictions[offset].append(cumulative)
            target_logit = post['logit'][batch, endpoint, :model.dynamics._stoch // 2].detach()
            prediction_logit = state['logit'][:, :model.dynamics._stoch // 2]
            categorical = torch.distributions.kl_divergence(
                model.dynamics.s.get_dist({'logit': target_logit}),
                model.dynamics.s.get_dist({'logit': prediction_logit})) / (model.dynamics._stoch // 2)
            hidden = state['deter'][:, :model.dynamics._deter // 2]
            target_hidden = post['deter'][batch, endpoint, :model.dynamics._deter // 2].detach()
            hidden_error = (F.normalize(hidden, dim=-1) - F.normalize(target_hidden, dim=-1)).square().mean(-1)
            per_prefix = categorical + options['fc_hidden_weight'] * hidden_error
            _ratio(stats, 'fc_categorical', categorical, mask, weight)
            _ratio(stats, 'fc_hidden', hidden_error, mask, weight)
            fc_sum = fc_sum + weight * (per_prefix * mask).sum()
            if teacher is not None:
                map_mask = mask & teacher_valid[batch, endpoint]
                map_error = (model.heatmap_logits(state).sigmoid() - teacher[batch, endpoint]).square().mean((-1, -2))
                _ratio(stats, 'fc_heat', map_error, map_mask, weight)
                fc_sum = fc_sum + weight * options['fc_heat_weight'] * (map_error * map_mask).sum()
            fc_count = fc_count + weight * mask.sum()
    losses['fc'] = fc_sum / fc_count.clamp_min(1)

    rc_sum, rc_count = zero, zero
    for horizon in active:
        j = horizons.index(horizon)
        mask, weight = prefix[:, j], float(options['horizon_weights'][j])
        target = zero.expand(starts.shape[0])
        for offset in range(1, horizon + 1):
            endpoint = (times + offset).clamp(max=data['is_first'].shape[1] - 1)
            target = target + model._config.discount ** (offset - 1) * data['task_reward'][batch, endpoint, 0]
        target = target.detach()
        prediction = torch.stack(predictions[horizon]).mean(0)
        error = prediction - target
        nonzero = target.abs() > 1e-6
        quantities = {'prediction_mean': prediction, 'target_mean': target, 'bias': error,
                      'mae': error.abs(), 'mse': error.square()}
        for population, selected in (('all', mask), ('nonzero', mask & nonzero), ('zero', mask & ~nonzero)):
            for scope, scale in (('all_horizons', weight), ('h' + str(horizon), 1.0)):
                name = 'rc_' + scope + '_' + population
                stats[name + '_prefixes'] = stats[name + '_prefixes'] + selected.sum().detach()
                for quantity, value in quantities.items():
                    _ratio(stats, name + '_' + quantity, value, selected, scale)
        rc_sum = rc_sum + weight * (error.square() * mask).sum()
        rc_count = rc_count + weight * mask.sum()
    losses['rc'] = rc_sum / rc_count.clamp_min(1)
    return losses, stats


def _gradient_diagnostics(losses, parameters, weights):
    """Compare gradients on the same dynamics parameters, without .grad writes."""
    result = {'gradient_parameter_count': float(sum(p.numel() for p in parameters))}
    base_gradients = None
    for name in ('base', 'fc', 'rc'):
        loss = losses[name]
        gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True) if loss.requires_grad else tuple(None for _ in parameters)
        norm_sq = sum(float(g.detach().float().square().sum(dtype=torch.float64).cpu()) for g in gradients if g is not None)
        norm = math.sqrt(norm_sq)
        result['grad_' + ('base' if name == 'base' else name + '_raw') + '_norm'] = norm
        if name == 'base':
            base_gradients = gradients
        else:
            result['grad_' + name + '_weighted_norm'] = abs(weights[name]) * norm
            base_norm = result['grad_base_norm']
            dot = sum(float((a.detach().float() * b.detach().float()).sum(dtype=torch.float64).cpu())
                      for a, b in zip(base_gradients, gradients) if a is not None and b is not None)
            result['grad_' + name + '_base_cosine_numerator'] = dot
            result['grad_' + name + '_base_cosine_denominator'] = base_norm * norm
        del gradients
    # Tensor lists are intentionally short-lived; neither optimizer state nor
    # persistent parameter.grad buffers receive these gradients.
    return result


def diagnostic_metrics(model, batch, gradients=True, starts_per_sequence=None):
    if not isinstance(gradients, bool):
        raise ValueError('gradients must be boolean')
    limit = model.options['starts_per_sequence'] if starts_per_sequence is None else starts_per_sequence
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError('starts_per_sequence must be a positive integer')
    data = model.preprocess(batch)
    if 'task_reward' not in data:
        raise ValueError('Probe requires recorded task_reward labels')
    shape = data['is_first'].shape
    data.setdefault('sequence_valid', torch.ones(shape, device=data['image'].device))
    data.setdefault('loss_mask', data['sequence_valid'])
    mask = data['loss_mask'] * data['sequence_valid']
    if not bool(mask.any()):
        raise ValueError('Probe batch has no valid learning rows')
    stats = {'gradients_enabled': float(gradients), 'learning_rows': float(mask.sum()),
             'starts_per_sequence': float(limit),
             'configured_fc_weight': float(model.options['fc_weight']),
             'configured_rc_weight': float(model.options['rc_weight'])}
    grad_context = torch.enable_grad() if gradients else torch.no_grad()
    with _parameter_grad_flags(model, gradients), grad_context:
        with torch.cuda.amp.autocast(model._use_amp):
            embed = model.encoder(data)
            tokens, local_logits = model.writer(data['image']) if model.writer is not None else (None, None)
            with _memory_capture(model.dynamics, 'memory_posterior') as memory_posterior:
                post, prior = model.dynamics.observe(embed, data['action'], data['is_first'],
                    memory_tokens=tokens if model.options['memory_enabled'] else None,
                    sequence_valid=data['sequence_valid'], loss_mask=mask)
            stats.update(memory_posterior)
            kl_loss, _, _, _ = model.dynamics.kl_loss(post, prior, model._config.kl_free,
                                                     model._config.dyn_scale, model._config.rep_scale)
            feature = model.dynamics.get_feat(post)
            base = model._masked_mean(kl_loss, mask)
            for name, head in model.heads.items():
                grad_name = 'reward' if name in ('task_reward', 'clip_reward') else name
                inp = feature if grad_name in model._config.grad_heads else feature.detach()
                prediction = head(inp)
                predictions = prediction if isinstance(prediction, dict) else {name: prediction}
                for key, distribution in predictions.items():
                    loss = model._masked_mean(-distribution.log_prob(data[key]), mask)
                    base = base + loss * model._scales.get(key, 1.0)
                    stats[key + '_loss'] = loss.detach()
            if model.options['local_heat_weight'] > 0 or model.options['posterior_heat_weight'] > 0:
                target = model._teacher(data)
                map_mask = mask * data['heatmap_valid']
                if model.options['local_heat_weight'] > 0:
                    local = model._masked_mean((local_logits.sigmoid() - target).square().mean((-1, -2)), map_mask)
                    base = base + model.options['local_heat_weight'] * local
                    stats['local_heat_loss'] = local.detach()
                if model.options['posterior_heat_weight'] > 0:
                    posterior = model._masked_mean((model.heatmap_logits(post).sigmoid() - target).square().mean((-1, -2)), map_mask)
                    base = base + model.options['posterior_heat_weight'] * posterior
                    stats['posterior_heat_loss'] = posterior.detach()
            starts = _fixed_starts(model, data, limit)
            with _memory_capture(model.dynamics, 'memory_recorded') as memory_recorded:
                auxiliary, recorded_stats = _recorded_probe(model, post, data, starts)
            stats.update(memory_recorded)
            stats.update(recorded_stats)
        stats['base_loss'] = base.detach()
        for key in ('fc', 'rc'):
            stats[key + '_loss'] = auxiliary[key].detach()
            stats[key + '_weighted_loss'] = auxiliary[key].detach() * model.options[key + '_weight']
        if gradients:
            stats.update(_gradient_diagnostics(dict(base=base, **auxiliary), tuple(model.dynamics.parameters()),
                                              {'fc': model.options['fc_weight'], 'rc': model.options['rc_weight']}))
    output = {}
    for name, value in stats.items():
        number = float(value.detach().double().cpu()) if isinstance(value, torch.Tensor) else float(value)
        if not math.isfinite(number):
            raise FloatingPointError('Nonfinite diagnostic metric: ' + name)
        output[name] = number
    return output
