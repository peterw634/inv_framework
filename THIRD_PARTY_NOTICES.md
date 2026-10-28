# NaSn integration provenance

The NaSn integration was migrated from
https://github.com/peterw634/nasn-truncated-ct at commit
`35bb7088273dddb07495cafbf9fcc612c07ae930`.

- `inv_framework/applications/nasn_ct/` and the CUDA projector under
  `inv_framework/operators/ct/radon_cuda/` derive from that project's
  BSD-3-Clause code. See `LICENSES/NaSn-BSD-3-Clause.txt`.
- `inv_framework/models/pmf/` retains the upstream pMF MIT license,
  reproduced in `LICENSES/pMF-MIT.txt`.
- `radon_cuda/gridrec.py`, `gridrec_cuda.cpp` and `gridrec_cuda_kernel.cu`
  derive from TomoPy 1.15.x Gridrec (Copyright 2015 UChicago Argonne, LLC).
  See `LICENSES/TomoPy-BSD-3-Clause.txt`.
- Imported data manifests retain their CC BY 4.0 license metadata and
  `LICENSES/CC-BY-4.0-DATA.txt`. No dataset or model weight is bundled.

The pMF model and low-level CUDA source files are preserved byte-for-byte.
The application imports were relocated; the reconstruction and benchmark
now use the framework's `LinearOperator`, `DDSSolver`, `MeanFlowSchedule`,
`PMFVelocityPrior` and `CGDataConsistency`. The original NaSn project is not
a runtime dependency. No ASTRA runtime or source is included in this port.

## Diffusion reconstruction algorithms

The inference implementations in `solvers/diffusion/{mcg,pgdm,ddim_dds,dmplug,reddiff,hybridreg,psld,resample,diffstategrad,guidance}.py`
are adapted from https://github.com/DM4CT/DM4CT, commit
49b3e5907178b56338d55c17944de17168a7a0a1 (`pipelines.py`, `condition_methods.py`).
Algorithm attribution remains with the respective method authors. No standalone
license file was present in that reference snapshot; this notice does not assign
a new license to the upstream material.

`resample_schedule.py` preserves the reference extension of the Hugging Face
Diffusers DDIM scheduler. Diffusers is Copyright the HuggingFace Team, licensed
under Apache-2.0: https://github.com/huggingface/diffusers/blob/v0.32.2/LICENSE .
A copy is included in `LICENSES/diffusers-Apache-2.0.txt`.
The local adaptation separates sampling algorithms from pipeline loading, image
serialization and CT operators. See `docs/diffusion_methods.md` for the retained
reference behaviors and numerical validation protocol.
