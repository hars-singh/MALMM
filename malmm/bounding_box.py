"""World-space bounds from native simulator object geometry."""

from itertools import product

import numpy as np


def world_bounds(box, matrix):
    """Transform eight local bounding-box corners into world-space bounds."""
    box, matrix = np.asarray(box), np.asarray(matrix)
    corners = np.array(list(product(box[:2], box[2:4], box[4:6])))
    corners = corners @ matrix[:3, :3].T + matrix[:3, 3]
    return corners.min(axis=0), corners.max(axis=0)


def is_nonphysical_plane(shape):
    """Identify static, nonrespondable, zero-thickness planes by properties alone."""
    box = np.asarray(shape.get_bounding_box()).reshape(3, 2)
    return (
        not shape.is_dynamic()
        and not shape.is_respondable()
        and np.min(box[:, 1] - box[:, 0]) <= 1e-6
    )


class WorldBoundingBox:
    """World-space bounds computed from an object's native bounding box."""

    def __init__(self, source):
        self.minimum, self.maximum = world_bounds(source.get_bounding_box(), source.get_matrix())
