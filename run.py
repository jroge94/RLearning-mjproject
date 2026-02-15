#!/usr/bin/env python3
"""
CLI to run experiments and generate plots.

Examples:
  python run.py experiment                    # run all algos × envs from config
  python run.py experiment --algos ppo --envs Hopper-v3
  python run.py experiment --algos ppo,impala --envs all --rounds 10
  python run.py experiment --list             # show available algos and envs
  python run.py plot                          # plot all from config paths
  python run.py plot --logs-dir logs --imgs-dir imgs
  python run.py plot --envs Hopper-v3 Walker2d-v3
"""
import argparse
import logging
import sys

import config

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _parse_algos(s: str) -> list[str]:
    if not s or s.lower() == "all":
        return config.get_algo_names()
    return [a.strip() for a in s.split(",") if a.strip()]


def _parse_envs(s: str) -> list[str]:
    if not s or s.lower() == "all":
        return config.get_env_names()
    return [e.strip() for e in s.split(",") if e.strip()]


def _validate_choices(algos: list[str], envs: list[str]) -> None:
    valid_algos = set(config.get_algo_names())
    valid_envs = set(config.get_env_names())
    bad_algos = set(algos) - valid_algos
    bad_envs = set(envs) - valid_envs
    if bad_algos:
        logger.error("Unknown algos: %s. Available: %s", bad_algos, valid_algos)
        sys.exit(1)
    if bad_envs:
        logger.error("Unknown envs: %s. Available: %s", bad_envs, valid_envs)
        sys.exit(1)


def cmd_list(_: argparse.Namespace) -> None:
    """Print available algos and envs from config."""
    print("Algos:", config.get_algo_names())
    print("Envs:", config.get_env_names())
    print("Logs dir:", config.LOGS_DIR)
    print("Imgs dir:", config.IMGS_DIR)


def cmd_experiment(args: argparse.Namespace) -> None:
    """Run RL experiments for selected algos and envs."""
    if args.list:
        cmd_list(args)
        return

    algos = _parse_algos(args.algos) if args.algos else config.get_algo_names()
    envs = _parse_envs(args.envs) if args.envs else config.get_env_names()
    max_rounds = args.rounds
    _validate_choices(algos, envs)

    import run_experiment
    run_experiment.main(algos=algos, envs=envs, max_rounds=max_rounds)


def cmd_plot(args: argparse.Namespace) -> None:
    """Generate plots from experiment logs."""
    import plot

    logs_dir = args.logs_dir or config.LOGS_DIR
    imgs_dir = args.imgs_dir or config.IMGS_DIR
    envs = _parse_envs(args.envs) if args.envs else config.get_env_names()
    algos = _parse_algos(args.algos) if args.algos else config.get_algo_names()

    _validate_choices(algos, envs)
    plot.main(logs_dir=logs_dir, imgs_dir=imgs_dir, envs=envs, models=algos)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run RL experiments and plot results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command")

    # experiment
    p_exp = subparsers.add_parser("experiment", help="Run training experiments")
    p_exp.add_argument(
        "--algos",
        type=str,
        metavar="NAMES",
        help="Comma-separated algo names, or 'all' (default: all from config)",
    )
    p_exp.add_argument(
        "--envs",
        type=str,
        metavar="NAMES",
        help="Comma-separated env names, or 'all' (default: all from config)",
    )
    p_exp.add_argument(
        "--rounds",
        type=int,
        metavar="N",
        default=None,
        help="Max training rounds (default: from config.stop_max_round)",
    )
    p_exp.add_argument(
        "--list",
        action="store_true",
        help="List available algos and envs, then exit",
    )
    p_exp.set_defaults(func=cmd_experiment)

    # plot
    p_plot = subparsers.add_parser("plot", help="Generate plots from logs")
    p_plot.add_argument(
        "--logs-dir",
        type=str,
        default=None,
        help="Log CSV directory (default: config.LOGS_DIR)",
    )
    p_plot.add_argument(
        "--imgs-dir",
        type=str,
        default=None,
        help="Output directory for plots (default: config.IMGS_DIR)",
    )
    p_plot.add_argument(
        "--envs",
        type=str,
        metavar="NAMES",
        help="Comma-separated env names to include, or 'all'",
    )
    p_plot.add_argument(
        "--algos",
        type=str,
        metavar="NAMES",
        help="Comma-separated algo names to include, or 'all'",
    )
    p_plot.set_defaults(func=cmd_plot)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
