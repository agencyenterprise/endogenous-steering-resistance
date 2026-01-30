#!/usr/bin/env python3
"""Allow running as: python -m experiment_06_sequential_activations"""
from . import main, parse_args

if __name__ == "__main__":
    import asyncio
    args = parse_args()
    asyncio.run(main(otd_file=args.otd_file))
