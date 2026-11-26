"""Command line interface: validate, render artifacts, serve the UI."""
from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="stochastic_medium",
        description="Non-uniform participating-media stochastic tracking kernel")
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve", help="launch the experiment window")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8765)

    p_val = sub.add_parser("validate", help="run statistical validation + figures")
    p_val.add_argument("--quick", action="store_true")
    p_val.add_argument("--json-out", default=None)

    p_render = sub.add_parser("render", help="render one scene to a PNG")
    p_render.add_argument("--scene", default="complex")
    p_render.add_argument("--kappa", type=float, default=1.7)
    p_render.add_argument("--spp", type=int, default=16)
    p_render.add_argument("--width", type=int, default=384)
    p_render.add_argument("--height", type=int, default=256)
    p_render.add_argument("--partition", default="certified_grid")
    p_render.add_argument("--out", default="outputs/cli_render.png")

    args = parser.parse_args(argv)

    if args.command == "serve":
        from .server import serve
        serve(args.host, args.port)
        return 0

    if args.command == "validate":
        from .validate import run
        report = run(output_json=args.json_out, quick=args.quick)
        print("all statistical checks pass:",
              report["all_statistical_checks_pass"])
        print("report:", report["json_path"])
        return 0 if report["all_statistical_checks_pass"] else 1

    if args.command == "render":
        import base64
        from pathlib import Path
        from .params import RenderConfig
        from .scenes import load_scene
        from .batch import render_single, tonemap_png_base64
        cfg = RenderConfig(scene=args.scene, kappa=args.kappa, spp=args.spp,
                           width=args.width, height=args.height,
                           partition=args.partition)
        out = render_single(load_scene(args.scene), cfg)
        url = tonemap_png_base64(out["rgb"], cfg.exposure)
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(url.split(",", 1)[1]))
        print("wrote", path, "stats", out["stats"])
        return 0

    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
