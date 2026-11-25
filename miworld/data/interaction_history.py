"""Past-only action/event inputs and training-only interaction pair sampling."""
import numpy as np
import torch


def interaction_events(events, agent, frames):
    result = np.zeros((frames, 2), np.float32)
    for event in events:
        t = int(event.get('observation_frame', -1))
        if event.get('actor', event.get('source')) != f'agent{agent}' or not 0 <= t < frames:
            continue
        kind = event.get('event')
        if kind in ('block_placed', 'block_dug'):
            result[t, 0 if kind == 'block_placed' else 1] += 1
    return result


def build_interaction_history(actions, health, health_valid, terminal, events,
                              agent, start, frames, length=64):
    """Slot t contains action[t-1] and events observed at t, never action[t].

    Include a prefix before the visual context; negative episode times are
    invalid padding. The model discards history across discontinuities.
    """
    n = len(health)
    incoming = np.zeros((n, 23), np.float32)
    incoming[1:] = actions[:n-1, agent]
    feedback = interaction_events(events, agent, n)
    live = np.asarray(health_valid[:, agent], bool) & (health[:, agent] > 0)
    valid = live.copy()
    done = np.asarray(terminal, bool)
    if done.ndim > 1:
        done = done.any(axis=tuple(range(1, done.ndim)))
    valid[1:] &= live[:-1] & ~done[:n-1]
    indices = np.arange(start-length+1, start+frames)
    inside = (indices >= 0) & (indices < n)
    clipped = indices.clip(0, n-1)
    result = np.concatenate((incoming, feedback), -1)[clipped]
    valid = valid[clipped] & inside
    result[~valid] = 0
    return torch.from_numpy(result), torch.from_numpy(valid)


def dropout_history_action_blocks(history, valid, probability, block_size=8,
                                  generator=None):
    """Drop complete action chunks while retaining observed event feedback.

    Closed-loop M3 receives its own previous eight-action chunks rather than a
    perfect teacher history.  Chunk dropout is a cheap robustness surrogate:
    it prevents the history encoder from treating every teacher action as
    exact, without fabricating place/dig success callbacks.  The last two
    channels are engine-observed event feedback and are therefore preserved.
    """
    probability = float(probability)
    if not 0 <= probability <= 1:
        raise ValueError('history action dropout probability must be in [0,1]')
    if block_size < 1:
        raise ValueError('history action dropout block size must be positive')
    if history.ndim != 2 or history.shape[-1] != 25 or valid.shape != history.shape[:1]:
        raise ValueError('history action dropout expects [time,25] history and [time] validity')
    if not probability:
        return history
    blocks = (len(history) + block_size - 1) // block_size
    dropped = torch.rand(blocks, generator=generator, device=history.device) < probability
    mask = dropped.repeat_interleave(block_size)[:len(history)] & valid.bool()
    result = history.clone()
    result[mask, :23] = 0
    return result


def successful_place_frames(actions, events, agent, tolerance=4):
    """One-to-one temporal association for sampling only, not a success label input."""
    commands = np.flatnonzero(actions[:, agent, 9] > .5)
    effects = sorted(int(e['transition_index']) for e in events
                     if e.get('event') == 'block_placed' and e.get('actor') == f'agent{agent}'
                     and 'transition_index' in e)
    matched, cursor = set(), 0
    for command in commands:
        while cursor < len(effects) and effects[cursor] < command:
            cursor += 1
        if cursor < len(effects) and effects[cursor] <= command+tolerance:
            matched.add(int(command)); cursor += 1
    return matched


def resample_interaction_pairs(target, valid, indices, context_start, successes, rng=None):
    """Nine anchors: three successful-click, three nearby no-click, three base.

    Uses existing masks and labels unchanged. No positive/negative pair means
    keep the original clip. Unconfirmed attempts remain in the base portion.
    """
    rng = np.random if rng is None else rng
    anchors = indices[:, -1].numpy() + context_start
    click = ((target[:, :8, 9] > .5) & valid[:, :8]).any(-1)
    positive = [i for i, anchor in enumerate(anchors) if any(
        valid[i, h] and target[i, h, 9] > .5 and int(anchor+h) in successes for h in range(8))]
    negative = np.flatnonzero(~click & valid[:, :8].all(-1)).tolist()
    if not positive or not negative:
        return target, valid, indices
    pairs = {p: [n for n in negative if abs(int(anchors[n]-anchors[p])) <= 24] for p in positive}
    positive = [p for p in positive if pairs[p]]
    if not positive:
        return target, valid, indices
    selected = []
    for _ in range(len(anchors)//3):
        p = int(rng.choice(positive))
        # Favor adjacent windows, but include longer confirmation intervals.
        candidates = pairs[p]
        weights = np.array([1 / abs(float(anchors[n]-anchors[p])) for n in candidates])
        n = int(rng.choice(candidates, p=weights/weights.sum()))
        selected.extend((p, n))
    base = rng.permutation(len(anchors)).tolist()
    selected.extend(base[:len(anchors)-len(selected)])
    selection = np.array(selected, np.int64)
    return target[selection], valid[selection], indices[torch.from_numpy(selection)]
