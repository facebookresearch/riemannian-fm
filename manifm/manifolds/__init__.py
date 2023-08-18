"""Copyright (c) Meta Platforms, Inc. and affiliates."""

# wrap around the manifolds from geoopt
from geoopt import Euclidean
from geoopt import ProductManifold
from .torus import FlatTorus
from .spd import SPD
from .sphere import Sphere
from .hyperbolic import PoincareBall
from .mesh import Mesh
from .utils import geodesic
