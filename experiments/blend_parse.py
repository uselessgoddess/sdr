#!/usr/bin/env python3
"""Minimal .blend parser for the new (BLENDER17) header format.

Goal: recover object transforms (world matrices) so we can find where the
irrigation needle (a FLIP-Fluids *inflow* object) enters the sinus, and how the
`sinuses_smooth` mesh was scaled/placed in the author's Blender simulation.

Handles: 17-byte header, 32-byte block heads (code8 + len8 + old8 + sdna4 +
nr4), little-endian, 64-bit pointers. Parses SDNA so struct field offsets are
read from the file itself (version-robust).
"""
import struct
import sys
import re
from collections import OrderedDict


class Blend:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.data = f.read()
        assert self.data[:7] == b"BLENDER", "not a blend file"
        # New header: "BLENDER17-01v0500" then first block at offset 17.
        # endian char at offset 12 ('v' little / 'V' big); 64-bit pointers.
        self.endian = "<" if self.data[12:13] == b"v" else ">"
        self.ptr = 8
        self.blocks = []  # (code, data_offset, length, sdna, nr)
        self._scan_blocks()
        self._parse_dna()

    def _scan_blocks(self):
        off = 17
        e = self.endian
        d = self.data
        # New (BLENDER17) 32-byte BHead: code(8) + old(8) + len(8) + sdna(4) + nr(4).
        while off + 32 <= len(d):
            code = d[off : off + 4].split(b"\0")[0].decode("ascii", "replace")
            old = struct.unpack_from(e + "Q", d, off + 8)[0]
            length = struct.unpack_from(e + "Q", d, off + 16)[0]
            sdna = struct.unpack_from(e + "i", d, off + 24)[0]
            nr = struct.unpack_from(e + "i", d, off + 28)[0]
            data_off = off + 32
            self.blocks.append((code, data_off, length, sdna, nr, old))
            if code == "ENDB":
                break
            off = data_off + length
        codes = [b[0] for b in self.blocks]
        sys.stderr.write(
            f"scanned {len(self.blocks)} blocks; last={codes[-1]}; "
            f"has DNA1={'DNA1' in codes} has ENDB={'ENDB' in codes}\n"
        )

    def _align4(self, off):
        return (off + 3) & ~3

    def _seek(self, off, marker):
        """Snap `off` forward to the next occurrence of a 4-byte marker
        (handles whatever padding the new DNA layout uses)."""
        i = self.data.index(marker, off, off + 16)
        return i

    def _parse_dna(self):
        # Locate DNA1 block.
        dna = next(b for b in self.blocks if b[0] == "DNA1")
        off = dna[1]
        d = self.data
        e = self.endian
        assert d[off : off + 4] == b"SDNA"
        off += 4
        assert d[off : off + 4] == b"NAME"
        off += 4
        n_names = struct.unpack_from(e + "i", d, off)[0]
        off += 4
        names = []
        for _ in range(n_names):
            end = d.index(b"\0", off)
            names.append(d[off:end].decode("ascii"))
            off = end + 1
        off = self._seek(off, b"TYPE")
        assert d[off : off + 4] == b"TYPE", d[off : off + 4]
        off += 4
        n_types = struct.unpack_from(e + "i", d, off)[0]
        off += 4
        types = []
        for _ in range(n_types):
            end = d.index(b"\0", off)
            types.append(d[off:end].decode("ascii"))
            off = end + 1
        off = self._seek(off, b"TLEN")
        assert d[off : off + 4] == b"TLEN"
        off += 4
        tlens = list(struct.unpack_from(e + "%dh" % n_types, d, off))
        off += 2 * n_types
        off = self._seek(off, b"STRC")
        assert d[off : off + 4] == b"STRC"
        off += 4
        n_structs = struct.unpack_from(e + "i", d, off)[0]
        off += 4
        structs = []  # (type_index, [(field_type_index, field_name_index)])
        for _ in range(n_structs):
            t_idx, n_fields = struct.unpack_from(e + "hh", d, off)
            off += 4
            fields = []
            for _ in range(n_fields):
                ft, fn = struct.unpack_from(e + "hh", d, off)
                off += 4
                fields.append((ft, fn))
            structs.append((t_idx, fields))

        self.names = names
        self.types = types
        self.tlens = tlens
        self.structs = structs
        # Map type name -> struct index
        self.struct_by_type = {types[s[0]]: i for i, s in enumerate(structs)}

    def _name_size(self, name_idx, type_idx):
        """Size in bytes of a single field with given DNA name & type."""
        name = self.names[name_idx]
        base = self.ptr if name.startswith("*") else self.tlens[type_idx]
        # array dims, e.g. obmat[4][4], loc[3]
        count = 1
        for dim in re.findall(r"\[(\d+)\]", name):
            count *= int(dim)
        return base * count

    def struct_field_offsets(self, type_name):
        """Return OrderedDict: clean_field_name -> (offset, type_name, name)."""
        si = self.struct_by_type[type_name]
        _, fields = self.structs[si]
        offsets = OrderedDict()
        off = 0
        for ft, fn in fields:
            name = self.names[fn]
            sz = self._name_size(fn, ft)
            clean = name.lstrip("*").split("[")[0]
            offsets[clean] = (off, self.types[ft], name, sz)
            off += sz
        return offsets, off

    def iter_blocks(self, code):
        for b in self.blocks:
            if b[0] == code:
                yield b


def main():
    path = sys.argv[1]
    bl = Blend(path)

    # Dump the Object struct layout (fields we care about).
    obo, obsize = bl.struct_field_offsets("Object")
    sys.stderr.write(f"\nObject struct size = {obsize} bytes\n")
    for k in ("id", "loc", "dloc", "size", "dsize", "rot", "obmat",
              "object_to_world", "world_to_object", "imat"):
        if k in obo:
            sys.stderr.write(f"  {k:18s} off={obo[k][0]:5d} type={obo[k][1]} raw={obo[k][2]}\n")

    # ID struct: name is at offset of 'name' field.
    ido, _ = bl.struct_field_offsets("ID")
    name_off = ido["name"][0]

    e = bl.endian
    d = bl.data

    # Which transform field holds the world matrix?
    mat_field = "obmat" if "obmat" in obo else "object_to_world"
    mat_off = obo[mat_field][0] if mat_field in obo else None
    loc_off = obo["loc"][0] if "loc" in obo else None
    size_off = obo["size"][0] if "size" in obo else None
    rot_off = obo["rot"][0] if "rot" in obo else None
    quat_off = obo["quat"][0] if "quat" in obo else None
    rotmode_off = obo["rotmode"][0] if "rotmode" in obo else None

    print("\n=== Objects (name, world position, scale) ===")
    rows = []
    for (code, data_off, length, sdna, nr, old) in bl.iter_blocks("OB"):
        # name within ID at start of Object data
        nstart = data_off + name_off
        nend = d.index(b"\0", nstart)
        name = d[nstart:nend].decode("ascii", "replace")
        rec = {"name": name}
        if loc_off is not None:
            loc = struct.unpack_from(e + "3f", d, data_off + loc_off)
            rec["loc"] = tuple(round(x, 4) for x in loc)
        if size_off is not None:
            size = struct.unpack_from(e + "3f", d, data_off + size_off)
            rec["size"] = tuple(round(x, 5) for x in size)
        if rot_off is not None:
            rot = struct.unpack_from(e + "3f", d, data_off + rot_off)
            rec["rot"] = tuple(round(x, 5) for x in rot)
        if quat_off is not None:
            quat = struct.unpack_from(e + "4f", d, data_off + quat_off)
            rec["quat"] = tuple(round(x, 5) for x in quat)
        if rotmode_off is not None:
            rec["rotmode"] = struct.unpack_from(e + "h", d, data_off + rotmode_off)[0]
        if mat_off is not None:
            m = struct.unpack_from(e + "16f", d, data_off + mat_off)
            # Blender stores column-major 4x4: translation = m[12], m[13], m[14]
            rec["world_t"] = (round(m[12], 4), round(m[13], 4), round(m[14], 4))
            # scale = column lengths
            import math
            sx = math.sqrt(m[0] ** 2 + m[1] ** 2 + m[2] ** 2)
            sy = math.sqrt(m[4] ** 2 + m[5] ** 2 + m[6] ** 2)
            sz = math.sqrt(m[8] ** 2 + m[9] ** 2 + m[10] ** 2)
            rec["world_s"] = (round(sx, 5), round(sy, 5), round(sz, 5))
        rows.append(rec)
        print(
            f"  {name:28s} loc={rec.get('loc')} size={rec.get('size')} "
            f"rot={rec.get('rot')} rotmode={rec.get('rotmode')} "
            f"quat={rec.get('quat')}"
        )
    return rows


if __name__ == "__main__":
    main()
