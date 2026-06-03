# `assets/` — committed meshes

## `maxillary_sinus.stl`

The **right maxillary sinus** (гайморова пазуха) — the cavity issue #3 irrigates
through the needle. Binary STL, **millimetres**, in the same Blender mesh-local
frame as the recovered needle geometry (see `experiments/README.md`).

| property        | value                                            |
| --------------- | ------------------------------------------------ |
| triangles       | 13 804                                           |
| bounds min (mm) | `[ 7.34, -22.57,  5.51]`                         |
| bounds max (mm) | `[24.06,  24.22, 38.48]`                         |
| size (mm)       | `[16.72,  46.79, 32.97]`                         |
| signed volume   | **+2.228 ml** (positive ⇒ outward-wound)         |
| watertight      | yes (edge-manifold, every edge shared by 2 faces)|
| file size       | 690 284 B (≈ 674 KiB)                            |

It is loaded by `examples/maxillary_real.toml` and exercised in CI by the
`real_mesh_*` tests in `src/scene.rs`.

### Provenance

This mesh is **derived**, not hand-authored. The source is the author's patient
scan `sinuses_smooth.stl` (≈900 k triangles) — the *whole* nasal airspace: both
maxillary sinuses, the ethmoid air cells, the nasal cavity and the sphenoid, all
one connected void (the sinuses drain into the nose through their ostia). That
source is large and patient-derived, so it is **not committed**; only the small
extracted right-antrum cavity above is.

The extraction tool is `examples/prep_maxillary.rs`. Because the whole airspace
is a single connected component, the maxillary sinus cannot be separated by mesh
connectivity — only **spatially**, by a crop box whose faces fall in bone and
sever the thin medial ostium and the superomedial ethmoid bridge. The tool:

1. builds the **exact** signed-distance field of the full closed mesh
   (generalised winding number) over a generous right-side box
   `(2,-28,2)..(30,28,46)` mm at 1 mm (cached to disk — the slow step);
2. **clips** that field to the maxillary box `(8,-22,6)..(27,24,38)` mm
   (everything outside → bone);
3. **flood-fills** the air voxels into pockets and picks the one reaching the
   recovered needle entry `(24.6, 15.7, 18.9)` mm;
4. **masks** every other pocket to bone, leaving a single cavity;
5. polygonises with **Naive Surface Nets** and keeps the largest component — a
   guaranteed-watertight, outward-wound triangle mesh, far smaller than the scan.

### Regenerating

With the source scan in hand:

```sh
cargo run --release --example prep_maxillary -- \
    path/to/sinuses_smooth.stl assets/maxillary_sinus.stl
```

The tool prints the pockets it found, the chosen cavity's volume and a
watertightness/winding report, and writes sanity slices to `/tmp/sinus_out/`.
Pass six extra numbers `minx miny minz maxx maxy maxz` to override the clip box.
