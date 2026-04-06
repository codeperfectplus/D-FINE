"""
yolo_to_coreml.py
=================
Converts an ultralytics YOLO .pt checkpoint to a CoreML .mlpackage.

Known issues and mitigations applied automatically:

1. TopK k > 21  –  CoreML ANE restricts TopK to ≤ 21.  Passing max_det=20 to
   model.export() propagates into the Exporter and sets detect.max_det before
   tracing.

2. int() cast on non-scalar tensor  –  PSA attention in YOLO26+ generates an
   aten::Int TorchScript op on a 1-element 1-D array, which coremltools rejects
   with "only 0-dimensional arrays can be converted to Python scalars".  We
   monkey-patch the coremltools _int handler to squeeze single-element arrays.

3. ANE compile abort  –  Some YOLO architectures can't be compiled by the ANE.
   Using compute_units=CPU_AND_GPU avoids the ANE path entirely.

Usage:
    python yolo_to_coreml.py                    # yolo26n.pt → yolo26n.mlpackage
    python yolo_to_coreml.py --checkpoint yolo11n.pt --max_det 20 --size 640
"""

import argparse
from pathlib import Path

import coremltools as ct
from ultralytics import YOLO


def _patch_coremltools_int_and_compute():
    """
    Two patches applied before ultralytics model.export():

    1. _int op fix  –  aten::Int on a single-element 1-D tensor crashes the
       torch→MIL converter.  We squeeze it to a 0-D scalar before mb.const,
       which is the safe interpretation for int() on a 1-element tensor.

    2. CPU+GPU compute units  –  Inject compute_units=CPU_AND_GPU into every
       ct.convert() call so the ANE compiler is never invoked.  This avoids
       SIGABRT crashes for architectures (e.g., PSA attention) that the ANE
       can't compile, while still letting GPU accelerate inference.
    """
    import coremltools.converters.mil.frontend.torch.ops as ct_ops
    from coremltools.converters.mil.frontend.torch.torch_op_registry import _TORCH_OPS_REGISTRY
    from coremltools.converters.mil import Builder as mb

    # -- patch 1: _int op --
    original_int = ct_ops._int

    def patched_int(context, node):
        x = context[node.inputs[0]]
        if x.val is not None:
            val = x.val
            if hasattr(val, "ndim") and val.ndim > 0 and val.size == 1:
                res = mb.const(val=int(val.flat[0]), name=node.name)
                context.add(res)
                return
        original_int(context, node)

    ct_ops._int = patched_int
    original_registry_int = _TORCH_OPS_REGISTRY.get_func("int")
    _TORCH_OPS_REGISTRY.set_func_by_name(patched_int, "int")
    _TORCH_OPS_REGISTRY.set_func_by_name(patched_int, "_int")

    # -- patch 2: ct.convert compute_units --
    original_convert = ct.convert

    def patched_convert(*args, **kwargs):
        kwargs.setdefault("compute_units", ct.ComputeUnit.CPU_AND_GPU)
        return original_convert(*args, **kwargs)

    ct.convert = patched_convert

    def restore():
        ct_ops._int = original_int
        if original_registry_int is not None:
            _TORCH_OPS_REGISTRY.set_func_by_name(original_registry_int, "int")
            _TORCH_OPS_REGISTRY.set_func_by_name(original_registry_int, "_int")
        ct.convert = original_convert

    return restore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="yolo26x.pt")
    parser.add_argument("--output", default=None, help="Defaults to <stem>.mlpackage")
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument(
        "--max_det", type=int, default=20,
        help="Max detections (must be ≤ 21 for CoreML ANE TopK)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run a lightweight CoreML load check after export (no prediction run)",
    )
    args = parser.parse_args()

    output = args.output or (Path(args.checkpoint).stem + ".mlpackage")

    restore_int = _patch_coremltools_int_and_compute()

    try:
        model = YOLO(args.checkpoint)
        result = model.export(
            format="coreml",
            imgsz=args.size,
            batch=1,
            max_det=args.max_det,
        )
    finally:
        restore_int()

    output_path = Path(result) if result else Path(Path(args.checkpoint).stem + ".mlpackage")
    if str(output_path) != str(output) and output_path.exists():
        output_path.rename(output)
        output_path = Path(output)

    print(f"\nCoreML model saved to: {output_path}")

    if args.verify:
        mlmodel = ct.models.MLModel(
            str(output_path),
            compute_units=ct.ComputeUnit.CPU_AND_GPU,
        )
        spec = mlmodel.get_spec()
        input_name = spec.description.input[0].name
        print(f"Verified load. CoreML input name: {input_name}")


if __name__ == "__main__":
    main()