"""
Gray-Scott Reaction-Diffusion Simulation using THRML
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
from jaxtyping import Array, Key, PyTree

from thrml.block_management import Block
from thrml.block_sampling import (
    BlockGibbsSpec,
    sample_states,
    SamplingSchedule,
)
from thrml.conditional_samplers import (
    _SamplerState,
    _State,
    AbstractConditionalSampler,
)
from thrml.factor import AbstractFactor, FactorSamplingProgram
from thrml.interaction import InteractionGroup
from thrml.pgm import AbstractNode


# ====== Node Types ======
class ChemicalU(AbstractNode):
    """First chemical species (activator)"""
    pass


class ChemicalV(AbstractNode):
    """Second chemical species (inhibitor)"""
    pass


# ====== Interactions ======
class DiffusionInteraction(eqx.Module):
    """Interaction that gathers neighbor concentrations for diffusion"""
    weights: Array


class ChemicalCouplingInteraction(eqx.Module):
    """Interaction that provides the other chemical's concentration at the same location"""
    pass


class SelfInteraction(eqx.Module):
    """Interaction that provides a node with its own current value"""
    pass


# ====== Factors ======
class DiffusionFactor(AbstractFactor):
    """Factor that creates diffusion interactions based on graph edges"""

    weights: Array

    def __init__(self, graph: nx.Graph, all_nodes: list):
        edges = list(graph.edges())
        if len(edges) == 0:
            super().__init__([Block(all_nodes)])
            object.__setattr__(self, 'weights', jnp.array([]))
            object.__setattr__(self, '_edge_blocks', (Block([]), Block([])))
            return

        u_nodes, v_nodes = zip(*edges)
        u_block = Block(list(u_nodes))
        v_block = Block(list(v_nodes))

        object.__setattr__(self, '_edge_blocks', (u_block, v_block))
        object.__setattr__(self, 'weights', jnp.ones(len(edges)))

        super().__init__([Block(all_nodes)])

    def to_interaction_groups(self) -> list[InteractionGroup]:
        if len(self.weights) == 0:
            return []

        u_block, v_block = self._edge_blocks

        return [
            InteractionGroup(
                interaction=DiffusionInteraction(self.weights),
                head_nodes=u_block,
                tail_nodes=[v_block],
            ),
            InteractionGroup(
                interaction=DiffusionInteraction(self.weights),
                head_nodes=v_block,
                tail_nodes=[u_block],
            ),
        ]


class ChemicalCouplingFactor(AbstractFactor):
    """Factor that couples U and V at the same spatial location"""

    def __init__(self, nodes_u: list, nodes_v: list):
        assert len(nodes_u) == len(nodes_v), "U and V must have same length"

        object.__setattr__(self, '_block_u', Block(nodes_u))
        object.__setattr__(self, '_block_v', Block(nodes_v))

        super().__init__([Block(nodes_u), Block(nodes_v)])

    def to_interaction_groups(self) -> list[InteractionGroup]:
        return [
            InteractionGroup(
                interaction=ChemicalCouplingInteraction(),
                head_nodes=self._block_u,
                tail_nodes=[self._block_v],
            ),
            InteractionGroup(
                interaction=ChemicalCouplingInteraction(),
                head_nodes=self._block_v,
                tail_nodes=[self._block_u],
            ),
        ]


class SelfCouplingFactor(AbstractFactor):
    """Factor that provides each node with access to its own current state"""

    def __init__(self, nodes: list):
        object.__setattr__(self, '_block', Block(nodes))
        super().__init__([Block(nodes)])

    def to_interaction_groups(self) -> list[InteractionGroup]:
        return [
            InteractionGroup(
                interaction=SelfInteraction(),
                head_nodes=self._block,
                tail_nodes=[self._block],
            ),
        ]


# ====== Sampler ======
class GrayScottSampler(AbstractConditionalSampler):
    """Implements Gray-Scott reaction-diffusion dynamics"""

    Du: float
    Dv: float
    F: float
    k: float
    dt: float
    noise_level: float
    is_u: bool

    def __init__(self, Du=0.16, Dv=0.08, F=0.060, k=0.062, dt=1.0,
                 noise_level=0.01, is_u=True):
        self.Du = Du
        self.Dv = Dv
        self.F = F
        self.k = k
        self.dt = dt
        self.noise_level = noise_level
        self.is_u = is_u

    def sample(
        self,
        key: Key,
        interactions: list[PyTree],
        active_flags: list[Array],
        states: list[list[_State]],
        sampler_state: _SamplerState,
        output_sd: PyTree[jax.ShapeDtypeStruct],
    ) -> tuple[Array, _SamplerState]:
        current_val = None
        neighbor_sum = None
        neighbor_count = None
        other_chemical = None

        for active, interaction, state in zip(active_flags, interactions, states):
            if isinstance(interaction, SelfInteraction):
                if len(state) > 0:
                    current_val = state[0].squeeze()

            elif isinstance(interaction, DiffusionInteraction):
                if len(state) > 0:
                    neighbor_vals = state[0]
                    if neighbor_sum is None:
                        neighbor_sum = jnp.sum(active * neighbor_vals, axis=-1)
                        neighbor_count = jnp.sum(active, axis=-1)
                    else:
                        neighbor_sum += jnp.sum(active * neighbor_vals, axis=-1)
                        neighbor_count += jnp.sum(active, axis=-1)

            elif isinstance(interaction, ChemicalCouplingInteraction):
                if len(state) > 0:
                    other_chemical = state[0].squeeze()

        if current_val is None:
            n_cells = output_sd.shape[0] if output_sd.shape else 1
            current_val = jnp.ones(n_cells, dtype=jnp.float32) if self.is_u else jnp.zeros(n_cells, dtype=jnp.float32)

        if neighbor_sum is None:
            neighbor_sum = jnp.zeros_like(current_val)
            neighbor_count = jnp.zeros_like(current_val)

        if other_chemical is None:
            other_chemical = jnp.zeros_like(current_val)

        avg_neighbor = jnp.where(
            neighbor_count > 0,
            neighbor_sum / neighbor_count,
            current_val
        )
        laplacian = avg_neighbor - current_val

        if self.is_u:
            U = current_val
            V = other_chemical
            dU = self.Du * laplacian - U * V * V + self.F * (1.0 - U)
            new_val = U + self.dt * dU
        else:
            V = current_val
            U = other_chemical
            dV = self.Dv * laplacian + U * V * V - (self.F + self.k) * V
            new_val = V + self.dt * dV

        noise = jax.random.normal(key, new_val.shape) * self.noise_level
        new_val = new_val + noise
        new_val = jnp.clip(new_val, 0.0, 1.0).astype(jnp.float32)

        return new_val, sampler_state

    def init(self) -> _SamplerState:
        return None


# ====== Grid Creation ======
def create_reaction_diffusion_grid(rows, cols, periodic=True):
    """Create a grid with two chemical species at each location"""
    grid_u = [[ChemicalU() for _ in range(cols)] for _ in range(rows)]
    grid_v = [[ChemicalV() for _ in range(cols)] for _ in range(rows)]

    G_u = nx.Graph()
    G_v = nx.Graph()

    for r in range(rows):
        for c in range(cols):
            G_u.add_node(grid_u[r][c], coords=(r, c))
            G_v.add_node(grid_v[r][c], coords=(r, c))

    for r in range(rows):
        for c in range(cols):
            neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

            for dr, dc in neighbor_offsets:
                if periodic:
                    nr = (r + dr) % rows
                    nc = (c + dc) % cols
                    G_u.add_edge(grid_u[r][c], grid_u[nr][nc])
                    G_v.add_edge(grid_v[r][c], grid_v[nr][nc])
                else:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        G_u.add_edge(grid_u[r][c], grid_u[nr][nc])
                        G_v.add_edge(grid_v[r][c], grid_v[nr][nc])

    all_u = [node for row in grid_u for node in row]
    all_v = [node for row in grid_v for node in row]

    return G_u, G_v, grid_u, grid_v, all_u, all_v


def create_reaction_diffusion_program(
    graph_u, graph_v, all_u, all_v,
    Du=0.16, Dv=0.08, F=0.060, k=0.062, dt=1.0, noise_level=0.01
):
    """Create a THRML sampling program for Gray-Scott reaction-diffusion"""

    node_shape_dtypes = {
        ChemicalU: jax.ShapeDtypeStruct((), jnp.float32),
        ChemicalV: jax.ShapeDtypeStruct((), jnp.float32),
    }

    block_u = Block(all_u)
    block_v = Block(all_v)
    free_blocks = [block_u, block_v]
    clamped_blocks = []

    spec = BlockGibbsSpec(free_blocks, clamped_blocks, node_shape_dtypes)

    diffusion_u = DiffusionFactor(graph_u, all_u)
    diffusion_v = DiffusionFactor(graph_v, all_v)
    self_u = SelfCouplingFactor(all_u)
    self_v = SelfCouplingFactor(all_v)
    coupling = ChemicalCouplingFactor(all_u, all_v)

    sampler_u = GrayScottSampler(
        Du=Du, Dv=Dv, F=F, k=k, dt=dt,
        noise_level=noise_level, is_u=True
    )
    sampler_v = GrayScottSampler(
        Du=Du, Dv=Dv, F=F, k=k, dt=dt,
        noise_level=noise_level, is_u=False
    )

    program = FactorSamplingProgram(
        gibbs_spec=spec,
        samplers=[sampler_u, sampler_v],
        factors=[diffusion_u, diffusion_v, self_u, self_v, coupling],
        other_interaction_groups=[]
    )

    return program


def create_initial_chemicals(rows, cols, pattern='center_square', seed=42):
    """Create initial chemical concentrations"""
    U = np.ones((rows, cols), dtype=np.float32)
    V = np.zeros((rows, cols), dtype=np.float32)

    rng = np.random.RandomState(seed)

    if pattern == 'center_square':
        r, c = rows // 2, cols // 2
        size = 5
        U[r-size:r+size, c-size:c+size] = 0.0
        V[r-size:r+size, c-size:c+size] = 1.0

    elif pattern == 'random':
        n_perturbations = (rows * cols) // 20
        for _ in range(n_perturbations):
            r = rng.randint(0, rows)
            c = rng.randint(0, cols)
            size = rng.randint(2, 6)
            U[r-size:r+size, c-size:c+size] = 0.0
            V[r-size:r+size, c-size:c+size] = 1.0

    elif pattern == 'stripes':
        for c in range(0, cols, 20):
            U[:, c:c+3] = 0.0
            V[:, c:c+3] = 1.0

    U += rng.uniform(-0.01, 0.01, U.shape)
    V += rng.uniform(-0.01, 0.01, V.shape)

    U = np.clip(U, 0, 1)
    V = np.clip(V, 0, 1)

    return jnp.array(U.flatten()), jnp.array(V.flatten())


# ====== Simulation Session ======
class SimulationSession:
    """Manages a simulation session with full history"""

    def __init__(self, rows=256, cols=256):
        self.rows = rows
        self.cols = cols

        # Create grid and program
        self.G_u, self.G_v, self.grid_u, self.grid_v, self.all_u, self.all_v = \
            create_reaction_diffusion_grid(rows, cols, periodic=True)

        # Initial parameters
        self.params = {
            'Du': 0.16,
            'Dv': 0.08,
            'F': 0.055,
            'k': 0.062,
            'dt': 1.0,
            'noise_level': 0.001
        }

        # Create program
        self.program = create_reaction_diffusion_program(
            self.G_u, self.G_v, self.all_u, self.all_v,
            **self.params
        )

        # Initialize state
        U_init, V_init = create_initial_chemicals(rows, cols, pattern='stripes')

        # History storage (starts with initial state)
        self.history_u = [np.array(U_init)]
        self.history_v = [np.array(V_init)]
        self.current_u = U_init
        self.current_v = V_init

    def update_params(self, **kwargs):
        """Update simulation parameters and recreate program"""
        self.params.update(kwargs)
        self.program = create_reaction_diffusion_program(
            self.G_u, self.G_v, self.all_u, self.all_v,
            **self.params
        )

    def reset(self, pattern='stripes'):
        """Reset simulation to initial state"""
        U_init, V_init = create_initial_chemicals(self.rows, self.cols, pattern=pattern)
        self.history_u = [np.array(U_init)]
        self.history_v = [np.array(V_init)]
        self.current_u = U_init
        self.current_v = V_init

    def simulate_steps(self, n_steps):
        """Run n simulation steps and add to history"""
        schedule = SamplingSchedule(
            n_warmup=0,
            n_samples=n_steps,
            steps_per_sample=1
        )

        key = jax.random.key(len(self.history_u))
        states = sample_states(
            key,
            self.program,
            schedule,
            [self.current_u, self.current_v],
            [],
            [Block(self.all_u), Block(self.all_v)]
        )

        # Add all new states to history
        for i in range(n_steps):
            self.history_u.append(np.array(states[0][i]))
            self.history_v.append(np.array(states[1][i]))

        # Update current state
        self.current_u = states[0][-1]
        self.current_v = states[1][-1]

    def get_state(self, step=None):
        """Get state at specific step (None = latest)"""
        if step is None:
            step = len(self.history_v) - 1

        if step < 0 or step >= len(self.history_v):
            return None

        return {
            'step': step,
            'maxStep': len(self.history_v) - 1,
            'u_grid': self.history_u[step].tolist(),
            'v_grid': self.history_v[step].tolist(),
            'rows': self.rows,
            'cols': self.cols
        }

    def get_history_range(self, start=0, end=None, stride=1):
        """Get a range of historical states"""
        if end is None:
            end = len(self.history_v)

        frames = []
        for i in range(start, min(end, len(self.history_v)), stride):
            frames.append({
                'step': i,
                'v_grid': self.history_v[i].tolist()
            })

        return frames

    def interact(self, x, y, brush_size=10):
        """Add chemicals at position, invalidates future history"""
        # Get current state as numpy arrays
        u = np.array(self.current_u).reshape(self.rows, self.cols)
        v = np.array(self.current_v).reshape(self.rows, self.cols)

        # Paint circle
        for dy in range(-brush_size, brush_size + 1):
            for dx in range(-brush_size, brush_size + 1):
                if dx * dx + dy * dy <= brush_size * brush_size:
                    px = (x + dx) % self.cols
                    py = (y + dy) % self.rows
                    u[py, px] = 0.0
                    v[py, px] = 1.0

        # Update current state
        self.current_u = jnp.array(u.flatten())
        self.current_v = jnp.array(v.flatten())

        # Add to history
        self.history_u.append(np.array(self.current_u))
        self.history_v.append(np.array(self.current_v))
