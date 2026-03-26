#!/usr/bin/env python3
"""Allow running as: python -m experiment_prefill_control"""
from . import main

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
