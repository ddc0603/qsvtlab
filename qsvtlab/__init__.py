"""qsvtlab: Quantum Singular Value Transformation applications in PennyLane."""

from .angles import QSPResult, qsp_response, solve_qsp_phases, wx_to_reflection
from .block_encodings import BlockEncoding, lcu_block_encoding, unitary_dilation
from .chemistry import EigenFilter, filter_poly, molecule_hamiltonian
from .circuits import qsvt_block, qsvt_phases_for, real_qsvt
from .hamiltonian_sim import (
    heisenberg,
    qsvt_min_cost,
    qsvt_truncation_error,
    qsvt_cnot_cost,
    qsvt_time_evolution,
    tfim,
    trotter_cnot_cost,
    trotter_error,
    trotter_min_cost,
)
from .linear_solver import LinearSolveResult, hermitian_embedding, qsvt_solve
from .polynomials import cos_poly, inverse_poly, inverse_poly_lstsq, rescale, sin_poly

__version__ = "0.2.0"
