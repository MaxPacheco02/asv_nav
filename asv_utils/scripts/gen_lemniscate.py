#!/usr/bin/env python3
import math
import sys


def main():
    args = sys.argv[1:]

    ax = float(args[0]) if len(args) > 0 else 2.5
    ay = float(args[1]) if len(args) > 1 else ax * math.sqrt(2) / 4
    cx = float(args[2]) if len(args) > 2 else 0.0
    cy = float(args[3]) if len(args) > 3 else 0.0
    N = int(args[4]) if len(args) > 4 else 24

    print(f"# Lemniscate of Bernoulli  ax={ax}  ay={ay}  center=({cx}, {cy})  N={N}")
    print(f"# Generated with: python3 gen_lemniscate.py {ax} {ay} {cx} {cy} {N}")

    for i in range(N):
        t = 2 * math.pi * i / N
        denom = 1 + math.sin(t) ** 2
        x = ax * math.cos(t) / denom + cx
        y = ay * math.sin(t) * math.cos(t) / denom + cy
        deg = round(t * 180 / math.pi)
        comma = "," if i < N - 1 else " "
        print(f"    {x:+.4f},  {y:+.4f}{comma}   # t={deg:3d}°")


if __name__ == "__main__":
    main()
