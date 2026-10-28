"""Isolate sampler arithmetic from the reference CUDA BP's atomic rounding."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.reference_root).resolve() / "src"))
    import numpy as np
    import tifffile
    import torch
    from nasn_ct.operators import TruncatedRadon, TruncatedAdjoint
    from nasn_ct.weights import load_pmf
    from nasn_ct.random import BatchGenerator
    from nasn_ct.solvers.cg import cg_correct_full
    from inv_framework.operators.base import LinearOperator
    from inv_framework.solvers.diffusion import CGDataConsistency, MeanFlowSchedule

    model = load_pmf(args.weights)
    A = TruncatedRadon(2048, 2048, 512, 1024.).cuda()
    AT = TruncatedAdjoint(2048, 2048, 512, 1024.).cuda()
    image = torch.from_numpy(tifffile.imread(args.image).astype(np.float32))[None, None].cuda()
    records = []

    def record(kind, value):
        result = (A if kind == "forward" else AT)(value)
        records.append((kind, value.clone(), result.clone()))
        return result

    class Replay(LinearOperator):
        domain_shape = range_shape = (1, 2048, 2048)

        def __init__(self):
            self.position = 0
            self.maximum_input_difference = 0.

        def consume(self, kind, value):
            expected_kind, expected_input, output = records[self.position]
            assert kind == expected_kind
            difference = float((value - expected_input).abs().max())
            self.maximum_input_difference = max(self.maximum_input_difference, difference)
            self.position += 1
            # A replay is valid only if the consumer asks the same questions.
            assert difference == 0., (self.position, kind, difference)
            return output.clone()

        def forward(self, value):
            return self.consume("forward", value)

        def adjoint(self, value):
            return self.consume("adjoint", value)

    with torch.no_grad():
        target = A((image + 1) / 2)
        z = BatchGenerator("cuda", [0]).randn(tuple(image.shape)) * model.noise_scale
        t = torch.ones(1, device="cuda")
        u = model.u_fn(z, t, t, t, torch.zeros_like(t), t,
                       y=torch.zeros(1, dtype=torch.int32, device="cuda"))[0]
        x0 = z - u
        anchor = (x0 + 1) / 2
        reference = cg_correct_full(anchor, target,
                                    lambda x: record("forward", x),
                                    lambda y: record("adjoint", y), lam=5., max_iter=10)
        record("forward", reference)  # conditioning's residual logging
        repeat = cg_correct_full(anchor, target, A, AT, lam=5., max_iter=10)
        native_repeat_l2 = float((reference - repeat).norm() / reference.norm())
        schedule = MeanFlowSchedule(noise_scale=model.noise_scale)
        schedule.set_timesteps(16, device="cuda")
        uncorrected, estimate = schedule.step(u, t, z)
        replay = Replay()
        result, _ = CGDataConsistency(replay, lam=5., cg_iter=10,
                                      image_scale=.5, image_offset=.5).apply(
            uncorrected, estimate, target, z, schedule=schedule)
        expected = (1 - 15 / 16) * (2 * reference - 1) + (15 / 16) * z
        maximum_difference = float((result - expected).abs().max())
    report = {
        "scope": "one real-model synthetic DDS step; identical recorded CUDA operator responses",
        "operator_calls": len(records), "consumed_calls": replay.position,
        "operator_inputs_max_abs_difference": replay.maximum_input_difference,
        "next_iterate_max_abs_difference": maximum_difference,
        "reference_native_cg_repeat_relative_l2": native_repeat_l2,
        "passed": replay.position == len(records) and maximum_difference == 0.,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
