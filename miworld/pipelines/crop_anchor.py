"""Stateful M2-only crop recentering; never modifies body state or world voxels."""
from __future__ import annotations

import numpy as np


class CropAnchorTracker:
    """Keep a crop until an axis leaves center +/- (0.5 + margin) blocks.

    With margin=0.25, a foot just below n+0.5 does not move the crop down
    one layer. Real movement beyond the deadband recenters to the nearest
    voxel, including large movements in a single update. All coordinates and
    event addresses must use the returned anchor for the entire M2 block.

    Create/reset per episode, with a stable resident ordering. This is not a
    collision rule: no pose, velocity, action, or predicted residual is clamped.
    """

    def __init__(self, margin: float = 0.25):
        if not np.isfinite(margin) or not 0 <= margin <= 0.5:
            raise ValueError('crop margin must be finite and in [0, 0.5]')
        self.margin = float(margin)
        self._anchors = None

    def reset(self):
        self._anchors = None

    def update(self, positions):
        positions = np.asarray(positions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3 or not len(positions):
            raise ValueError('positions must be nonempty [residents, 3]')
        if not np.isfinite(positions).all() or np.abs(positions).max() > 2**40:
            raise ValueError('positions must be finite and safely representable')
        nearest = np.floor(positions + 0.5).astype(np.int64)
        if self._anchors is None:
            self._anchors = nearest
        elif self._anchors.shape != nearest.shape:
            raise ValueError('resident count changed; reset tracker for a new episode')
        elif self.margin == 0:
            self._anchors = nearest
        else:
            outside = np.abs(positions - self._anchors) > 0.5 + self.margin
            self._anchors = np.where(outside, nearest, self._anchors)
        return self._anchors.copy()

    def state_dict(self):
        return dict(margin=self.margin, anchors=None if self._anchors is None else self._anchors.tolist())

    def load_state_dict(self, state):
        if float(state['margin']) != self.margin:
            raise ValueError('saved crop margin does not match tracker configuration')
        anchors = state['anchors']
        if anchors is None:
            self.reset()
            return
        values = np.asarray(anchors, dtype=np.float64)
        if (values.ndim != 2 or values.shape[1] != 3 or not len(values)
                or not np.isfinite(values).all() or np.abs(values).max() > 2**40
                or not np.equal(values, np.floor(values)).all()):
            raise ValueError('saved anchors must be finite integer [residents, 3]')
        self._anchors = values.astype(np.int64)
