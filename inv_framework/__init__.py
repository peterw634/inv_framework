"""inv_framework: a small framework for solving inverse problems.

Supports both linear (CT, MRI, deblurring, super-resolution, ...) and
non-linear (phase retrieval, saturated sensing, ...) forward models. The
top-level abstractions are:

    from inv_framework.operators.base import ForwardOperator, LinearOperator
    from inv_framework.operators.noise import GaussianNoise, PoissonLogDomainNoise
    from inv_framework.solvers.base import InverseProblemSolver

Solvers shipping in this version:
    - Classical (linear only):  FBPSolver, SIRTSolver, LandweberSolver
    - Model-based (general):    DIPSolver, INRSolver
    - Diffusion (general):      DPSSolver

CT is provided as a worked example under operators/ct/. Adding a new domain
means subclassing ForwardOperator or LinearOperator; the solvers do not need
to change.
"""

__version__ = "0.1.0"
