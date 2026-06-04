#!/usr/bin/env python3
"""Extract the *compressed* sinus mesh's vertex bounds from prepare.blend.

The handover note warns the blend model is vertex-merged AND "масштаб не тот"
(wrong scale). To know whether the blend object transform (loc/rot/scale of
OBsinuses_smooth) can be reused to place the needle on the *full* STL, we must
compare the blend mesh's local bounds against the STL's local bounds
(min (-40,-44,0) .. max (40,44,94)). If they match, the frame is shared and the
needle mapping is valid; if not, the scale factor between them is the fix.

Usage: python3 blend_mesh_bounds.py /path/to/prepare_decompressed.blend
"""
import struct
import sys
from blend_parse import Blend


def main():
    path = sys.argv[1]
    bl = Blend(path)
    e, d = bl.endian, bl.data

    obo, _ = bl.struct_field_offsets("Object")
    data_off_in_ob = obo["data"][0] if "data" in obo else None
    ido, _ = bl.struct_field_offsets("ID")
    name_off = ido["name"][0]
    mo, _ = bl.struct_field_offsets("Mesh")
    totvert_off = mo["totvert"][0]
    mvert_off = mo["mvert"][0]
    mvo, mv_size = bl.struct_field_offsets("MVert")
    co_off = mvo["co"][0]
    sys.stderr.write(f"Object.data off={data_off_in_ob}; MVert size={mv_size} co off={co_off}\n")

    # index data blocks by their `old` address
    by_old = {}
    for (code, doff, length, sdna, nr, old) in bl.blocks:
        by_old[old] = (code, doff, length, sdna, nr)

    # find OBsinuses_smooth -> its mesh data pointer
    target = None
    for (code, doff, length, sdna, nr, old) in bl.iter_blocks("OB"):
        nstart = doff + name_off
        nend = d.index(b"\0", nstart)
        name = d[nstart:nend].decode("ascii", "replace")
        if "sinuses_smooth" in name:
            data_ptr = struct.unpack_from(e + "Q", d, doff + data_off_in_ob)[0]
            target = (name, data_ptr)
            sys.stderr.write(f"found {name}: data ptr -> {data_ptr}\n")
            break
    if not target:
        print("OBsinuses_smooth not found")
        return

    _, data_ptr = target
    if data_ptr not in by_old:
        print(f"mesh block {data_ptr} not found")
        return
    code, mdoff, mlen, _, _ = by_old[data_ptr]
    totvert = struct.unpack_from(e + "i", d, mdoff + totvert_off)[0]
    mvert_ptr = struct.unpack_from(e + "Q", d, mdoff + mvert_off)[0]
    sys.stderr.write(f"mesh block code={code} totvert={totvert} mvert_ptr={mvert_ptr}\n")

    def report(coords):
        xs = [c[0] for c in coords]; ys = [c[1] for c in coords]; zs = [c[2] for c in coords]
        print(f"verts: {len(coords)}")
        print(f"bounds min: ({min(xs):.3f}, {min(ys):.3f}, {min(zs):.3f})")
        print(f"bounds max: ({max(xs):.3f}, {max(ys):.3f}, {max(zs):.3f})")
        print(f"size      : ({max(xs)-min(xs):.3f}, {max(ys)-min(ys):.3f}, {max(zs)-min(zs):.3f})")
        cx=(min(xs)+max(xs))/2; cy=(min(ys)+max(ys))/2; cz=(min(zs)+max(zs))/2
        print(f"center    : ({cx:.3f}, {cy:.3f}, {cz:.3f})")

    coords = []
    if mvert_ptr and mvert_ptr in by_old:
        _, vdoff, vlen, _, _ = by_old[mvert_ptr]
        for i in range(totvert):
            base = vdoff + i * mv_size + co_off
            coords.append(struct.unpack_from(e + "3f", d, base))
        print("=== compressed sinus mesh (MVert) ===")
        report(coords)
    else:
        # new-style: positions in vdata CustomData "position" layer (float3)
        print("mvert pointer NULL -> positions in CustomData; scanning float3 blocks near mesh")
        # Fallback: find a DATA block sized totvert*12 (vec3f) reachable; brute force
        for (code2, doff2, length2, sdna2, nr2, old2) in bl.blocks:
            if code2 == "DATA" and length2 >= totvert * 12 and nr2 == totvert:
                cs = [struct.unpack_from(e + "3f", d, doff2 + i * 12) for i in range(totvert)]
                xs = [c[0] for c in cs]
                if -200 < min(xs) < 200 and max(xs) - min(xs) > 1:
                    print(f"candidate block old={old2} nr={nr2} len={length2}")
                    report(cs)
                    break


if __name__ == "__main__":
    main()
