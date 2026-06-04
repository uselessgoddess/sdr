#!/usr/bin/env python3
"""Analyze a binary STL: bounds, scale, and connected components.

Pure-Python (struct/array only) so it runs anywhere. Used to understand the
real sinus model handed to us in the issue: it is the *full* picture (many
sinuses) but only the maxillary (гайморова) one matters.
"""
import struct
import sys
from array import array


def load_binary_stl(path):
    with open(path, "rb") as f:
        data = f.read()
    n = struct.unpack_from("<I", data, 80)[0]
    print(f"header: {data[:80].split(bytes([0]))[0].decode('ascii', 'replace')!r}")
    print(f"triangles: {n}")
    # Each triangle: 12 floats normal+verts (50 bytes incl 2-byte attr).
    verts = array("f")
    # Read vertices only (skip normal[0:12], read 9 floats at +12).
    off = 84
    xs = array("f")
    ys = array("f")
    zs = array("f")
    tris = []  # (i0,i1,i2) into a deduped vertex list
    vmap = {}
    coords = []  # flat deduped coords
    quant = 1e-6  # 1 micron quantization for welding shared verts
    for t in range(n):
        base = off + t * 50
        idx = []
        for j in range(3):
            o = base + 12 + j * 12
            x, y, z = struct.unpack_from("<fff", data, o)
            key = (round(x / quant), round(y / quant), round(z / quant))
            vi = vmap.get(key)
            if vi is None:
                vi = len(coords) // 3
                vmap[key] = vi
                coords.extend((x, y, z))
            idx.append(vi)
        tris.append(tuple(idx))
    return coords, tris


def bounds(coords):
    xs = coords[0::3]
    ys = coords[1::3]
    zs = coords[2::3]
    return (
        (min(xs), min(ys), min(zs)),
        (max(xs), max(ys), max(zs)),
    )


def connected_components(n_verts, tris):
    """Union-find over welded vertices -> components."""
    parent = list(range(n_verts))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for (a, b, c) in tris:
        union(a, b)
        union(b, c)

    # group triangles by component root
    from collections import defaultdict

    comp_tris = defaultdict(list)
    for ti, (a, b, c) in enumerate(tris):
        comp_tris[find(a)].append(ti)
    return comp_tris


def main():
    path = sys.argv[1]
    coords, tris = load_binary_stl(path)
    n_verts = len(coords) // 3
    print(f"welded vertices: {n_verts}")
    (mn, mx) = bounds(coords)
    size = tuple(mx[i] - mn[i] for i in range(3))
    print(f"bounds min: {mn}")
    print(f"bounds max: {mx}")
    print(f"size (model units): {size}")
    print(f"  -> if mm: {tuple(round(s,2) for s in size)} mm")
    print(f"  -> if m:  {tuple(round(s*1000,2) for s in size)} mm")

    comp_tris = connected_components(n_verts, tris)
    comps = sorted(comp_tris.items(), key=lambda kv: -len(kv[1]))
    print(f"\nconnected components: {len(comps)}")
    for ci, (root, tlist) in enumerate(comps[:20]):
        # component bounds + signed volume
        vset = set()
        vol = 0.0
        for ti in tlist:
            a, b, c = tris[ti]
            vset.update((a, b, c))
            ax, ay, az = coords[3 * a], coords[3 * a + 1], coords[3 * a + 2]
            bx, by, bz = coords[3 * b], coords[3 * b + 1], coords[3 * b + 2]
            cx, cy, cz = coords[3 * c], coords[3 * c + 1], coords[3 * c + 2]
            # signed tetra volume to origin
            vol += (
                ax * (by * cz - bz * cy)
                - ay * (bx * cz - bz * cx)
                + az * (bx * cy - by * cx)
            ) / 6.0
        xs = [coords[3 * v] for v in vset]
        ys = [coords[3 * v + 1] for v in vset]
        zs = [coords[3 * v + 2] for v in vset]
        cmn = (min(xs), min(ys), min(zs))
        cmx = (max(xs), max(ys), max(zs))
        csize = tuple(round(cmx[i] - cmn[i], 2) for i in range(3))
        cen = tuple(round((cmx[i] + cmn[i]) / 2, 2) for i in range(3))
        print(
            f"  comp {ci:2d}: tris={len(tlist):7d} verts={len(vset):7d} "
            f"vol={vol:+.2f} size={csize} center={cen}"
        )


if __name__ == "__main__":
    main()
