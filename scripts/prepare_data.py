#!/usr/bin/env python3
"""Minimal text normalizer for later phases; not needed to build Phase 1."""
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    text = args.input.read_text(encoding="utf-8")
    # Preserve paragraph boundaries; normalize line endings and trailing spaces.
    normalized = "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").split("\n"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(normalized, encoding="utf-8")
    print(f"Wrote {len(normalized)} characters to {args.output}")

if __name__ == "__main__":
    main()
