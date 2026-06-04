# Experiments — recovering the real geometry from `prepare.blend`

These scripts are the reproducibility trail for issue #3: how the needle-entry
point and the patient mesh frame were recovered from the Blender study the
author shared (`sinuses_smooth.zip`, `prepare.zip`). They are **forensic
one-offs**, not part of the build; they need the issue attachments unpacked
under `/tmp/sinus_assets/` (the raw `.blend` and the 45 MB STL are far too big
to commit).

| file | what it does |
|------|--------------|
| `blend_parse.py` | minimal `.blend` reader: SDNA structs, block table, field offsets (BLENDER17 / 32-byte block heads). |
| `blend_objects.txt` | dump of every object's `loc` / `size` / `rot` — the transforms used to map Blender world → mesh-local. |
| `blend_mesh_bounds.py` | reads the *compressed* `OBsinuses_smooth` vertex bounds and shows they equal the full STL bounds → the STL and the blend mesh share one local frame, so the needle mapping is valid. |
| `analyze_stl.py` / `stl_analysis.txt` | sanity-check of the raw `sinuses_smooth.stl` (triangle count, bounds, watertightness). |

The numbers these produced are baked, as named constants with comments, into
`examples/inspect_real.rs` and `examples/locate_cavity.rs`, so the Rust side is
self-contained and reproducible without the attachments.

## Key recovered facts

* `OBsinuses_smooth`: `loc=(8.62187,-16.25081,26.75974)`, `rot=(-0.96002,0,0)`
  (X only), uniform `scale=0.04183`. Inverse map
  `world→local = Rx(+0.96002)·(world − loc) / 0.04183`.
* Needle emitter `OBIcosphere` world `(9.65111,-15.22509,26.67662)` →
  **local `(24.6, 15.7, 18.9)` mm**; needle shaft `OBCylinder.001`
  (`rot_x=0.41119`) gives a firing axis of local `(0, -0.98, 0.20)`.
* Compressed-mesh bounds `(-40.03,-44.61,0.02)..(40.05,44.46,94.12)` **=** STL
  bounds `(-40.06,-44.62,0)..(40.06,44.62,94.12)` → shared frame confirmed.
* Blender world gravity `(0,0,-9.81)` → local `(0, +8.03, -5.63) m/s²`.
