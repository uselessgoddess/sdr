//! Extract a watertight **maxillary sinus** mesh from the full-airway STL.
//!
//! The patient scan (`sinuses_smooth.stl`, ~900k triangles) is the *whole* nasal
//! airspace: both maxillary sinuses, the ethmoid air cells, the nasal cavity,
//! the sphenoid — all one connected void (the sinuses drain into the nose
//! through their ostia). Issue #3 only cares about the **right maxillary sinus**
//! (гайморова пазуха), the one the author irrigates through the needle.
//!
//! Because the whole airspace is a single connected component, the maxillary
//! sinus can't be separated by a mesh-connectivity filter — it can only be
//! isolated **spatially**, by a crop box whose faces fall in bone and sever the
//! thin ostium (medially, to the nasal cavity) and the ethmoid bridge
//! (superomedially). This tool does exactly that, robustly:
//!
//!   1. Build the *exact* signed-distance field of the full closed mesh
//!      (generalised winding number) over a generous right-side box. This is the
//!      slow step (minutes); it is **cached** to disk so clip-box tuning is fast.
//!   2. **Clip** the field to a maxillary box (everything outside → bone).
//!   3. **Flood-fill** the air voxels into pockets; pick the one at the needle.
//!   4. **Mask** every other pocket to bone, leaving a single cavity.
//!   5. Polygonise with **Naive Surface Nets** → a guaranteed-watertight,
//!      outward-wound triangle mesh, far smaller than the source scan.
//!
//! Lengths are millimetres in the STL-native (Blender mesh-local) frame, the
//! same frame the recovered needle point lives in (see `examples/locate_cavity`
//! and `experiments/README.md`).
//!
//! Run:
//!   cargo run --release --example prep_maxillary -- <in.stl> <out.stl> \
//!       [minx miny minz maxx maxy maxz]      # clip box, mm (optional)
//!
//! With no clip box it uses the tuned default below and reports the pockets it
//! found so the box can be refined.

use image::{Rgb, RgbImage};
use sdr::math::{Aabb, Vec3};
use sdr::mesh::TriMesh;
use sdr::sdf::Sdf;
use std::io::{Read, Write};

// --- Generous SDF build box (mm, STL-native frame), right maxillary side. ---
// Wide enough to contain any reasonable maxillary clip box plus the needle. The
// exact winding-number field over this box is the cached, expensive artefact.
const BUILD_MIN: Vec3 = Vec3::new(2.0, -28.0, 2.0);
const BUILD_MAX: Vec3 = Vec3::new(30.0, 28.0, 46.0);
const BUILD_DX: f64 = 1.0;
// Vertex-cluster decimation before the winding pass: the source is already a
// compressed (vertex-merged) model, so 1 mm clustering costs no real fidelity
// and makes the field ~6x cheaper. Confirmed earlier to agree with the raw mesh
// on the needle-in-bone test.
const DEC_CELL: f64 = 1.0;

// --- Recovered needle entry (see experiments/README.md). ---
const NEEDLE: Vec3 = Vec3::new(24.6, 15.7, 18.9);
const NEEDLE_AXIS: Vec3 = Vec3::new(0.0, -0.98, 0.20);

const CACHE: &str = "/tmp/sinus_out/sdf_full_cache.bin";

fn fmt(v: Vec3) -> String {
    format!("({:.2}, {:.2}, {:.2})", v.x, v.y, v.z)
}
fn arg_f64(i: usize, default: f64) -> f64 {
    std::env::args()
        .nth(i)
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Serialise an `Sdf` plus a small provenance header (build box / dx / cell /
/// triangle count) so a cached field is only reused when it matches.
fn cache_save(sdf: &Sdf, cell: f64, ntris: usize) -> std::io::Result<()> {
    if let Some(p) = std::path::Path::new(CACHE).parent() {
        std::fs::create_dir_all(p).ok();
    }
    let mut w = std::io::BufWriter::new(std::fs::File::create(CACHE)?);
    w.write_all(b"SDFCACHE2")?;
    for v in [
        sdf.origin.x,
        sdf.origin.y,
        sdf.origin.z,
        sdf.dx,
        cell,
        ntris as f64,
    ] {
        w.write_all(&v.to_le_bytes())?;
    }
    for n in [sdf.nx as u64, sdf.ny as u64, sdf.nz as u64] {
        w.write_all(&n.to_le_bytes())?;
    }
    for &d in &sdf.data {
        w.write_all(&d.to_le_bytes())?;
    }
    Ok(())
}

/// Load a cached field iff its header matches the expected build params.
fn cache_load(cell: f64, ntris: usize) -> Option<Sdf> {
    let mut buf = Vec::new();
    std::fs::File::open(CACHE)
        .ok()?
        .read_to_end(&mut buf)
        .ok()?;
    if buf.len() < 9 + 6 * 8 + 3 * 8 || &buf[..9] != b"SDFCACHE2" {
        return None;
    }
    let f = |o: usize| f64::from_le_bytes(buf[o..o + 8].try_into().unwrap());
    let u = |o: usize| u64::from_le_bytes(buf[o..o + 8].try_into().unwrap()) as usize;
    let origin = Vec3::new(f(9), f(17), f(25));
    let (dx, c, nt) = (f(33), f(41), f(49) as usize);
    let (nx, ny, nz) = (u(57), u(65), u(73));
    if (origin - BUILD_MIN).length() > 1e-9
        || (dx - BUILD_DX).abs() > 1e-12
        || (c - cell).abs() > 1e-12
        || nt != ntris
    {
        return None;
    }
    let mut data = vec![0.0f64; nx * ny * nz];
    let base = 81;
    if buf.len() < base + data.len() * 8 {
        return None;
    }
    for (i, d) in data.iter_mut().enumerate() {
        *d = f64::from_le_bytes(buf[base + i * 8..base + i * 8 + 8].try_into().unwrap());
    }
    Some(Sdf {
        origin,
        dx,
        nx,
        ny,
        nz,
        data,
    })
}

/// A flood-fill pocket: voxel count, position sum, bbox, and whether it touches
/// the clip boundary (i.e. was severed rather than fully enclosed).
struct Pocket {
    count: usize,
    sum: Vec3,
    bmin: Vec3,
    bmax: Vec3,
    touches: bool,
    label: u32,
}
impl Pocket {
    fn centroid(&self) -> Vec3 {
        self.sum / self.count as f64
    }
    fn volume_ml(&self, dx: f64) -> f64 {
        self.count as f64 * dx * dx * dx / 1000.0
    }
}

/// 6-connected flood-fill of the air voxels (`field[idx] < 0`) into pockets,
/// returning the per-voxel labels and the pocket list. Deterministic scan order.
fn flood(
    field: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
    origin: Vec3,
    dx: f64,
) -> (Vec<u32>, Vec<Pocket>) {
    let air = |idx: usize| field[idx] < 0.0;
    let mut label = vec![u32::MAX; nx * ny * nz];
    let mut pockets: Vec<Pocket> = Vec::new();
    let mut stack: Vec<(usize, usize, usize)> = Vec::new();
    for k0 in 0..nz {
        for j0 in 0..ny {
            for i0 in 0..nx {
                let id0 = (k0 * ny + j0) * nx + i0;
                if !air(id0) || label[id0] != u32::MAX {
                    continue;
                }
                let next = pockets.len() as u32;
                let (mut cnt, mut sum) = (0usize, Vec3::ZERO);
                let (mut bmin, mut bmax) =
                    (Vec3::splat(f64::INFINITY), Vec3::splat(f64::NEG_INFINITY));
                let mut touches = false;
                label[id0] = next;
                stack.push((i0, j0, k0));
                while let Some((i, j, k)) = stack.pop() {
                    cnt += 1;
                    let p = origin + Vec3::new(i as f64 * dx, j as f64 * dx, k as f64 * dx);
                    sum += p;
                    bmin = bmin.min(p);
                    bmax = bmax.max(p);
                    if i == 0 || j == 0 || k == 0 || i == nx - 1 || j == ny - 1 || k == nz - 1 {
                        touches = true;
                    }
                    let visit =
                        |ni: usize, nj: usize, nk: usize, st: &mut Vec<_>, lab: &mut Vec<u32>| {
                            let id = (nk * ny + nj) * nx + ni;
                            if air(id) && lab[id] == u32::MAX {
                                lab[id] = next;
                                st.push((ni, nj, nk));
                            }
                        };
                    if i > 0 {
                        visit(i - 1, j, k, &mut stack, &mut label);
                    }
                    if i + 1 < nx {
                        visit(i + 1, j, k, &mut stack, &mut label);
                    }
                    if j > 0 {
                        visit(i, j - 1, k, &mut stack, &mut label);
                    }
                    if j + 1 < ny {
                        visit(i, j + 1, k, &mut stack, &mut label);
                    }
                    if k > 0 {
                        visit(i, j, k - 1, &mut stack, &mut label);
                    }
                    if k + 1 < nz {
                        visit(i, j, k + 1, &mut stack, &mut label);
                    }
                }
                pockets.push(Pocket {
                    count: cnt,
                    sum,
                    bmin,
                    bmax,
                    touches,
                    label: next,
                });
            }
        }
    }
    (label, pockets)
}

/// Count how many triangle edges are shared by exactly two faces. A closed,
/// edge-manifold mesh has *every* edge shared by exactly two — a quick, honest
/// watertightness check to print alongside the signed volume.
fn manifold_report(m: &TriMesh) -> (usize, usize, usize) {
    use std::collections::HashMap;
    let mut edges: HashMap<(u32, u32), u32> = HashMap::new();
    for t in &m.indices {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            let key = if a < b { (a, b) } else { (b, a) };
            *edges.entry(key).or_insert(0) += 1;
        }
    }
    let (mut boundary, mut nonmanifold) = (0, 0);
    for &c in edges.values() {
        if c == 1 {
            boundary += 1;
        } else if c > 2 {
            nonmanifold += 1;
        }
    }
    (edges.len(), boundary, nonmanifold)
}

/// One in-plane slice of a field for a quick visual sanity check.
fn slice_png(
    sdf: &Sdf,
    hax: usize,
    vax: usize,
    oax: usize,
    at: Vec3,
    mark: Vec3,
    px: u32,
) -> RgbImage {
    const AIR: Rgb<u8> = Rgb([232, 236, 244]);
    const BONE: Rgb<u8> = Rgb([40, 40, 50]);
    const MARK: Rgb<u8> = Rgb([255, 70, 70]);
    let bb = sdf.bounds();
    let (mn, mx, at, mk) = (
        bb.min.to_array(),
        bb.max.to_array(),
        at.to_array(),
        mark.to_array(),
    );
    let wm = (mx[hax] - mn[hax]).max(1e-9);
    let hm = (mx[vax] - mn[vax]).max(1e-9);
    let (w, h) = (px, (px as f64 * hm / wm).round().max(1.0) as u32);
    let mut img = RgbImage::from_pixel(w, h, BONE);
    for py in 0..h {
        for pxi in 0..w {
            let mut c = [0.0; 3];
            c[hax] = mn[hax] + (pxi as f64 + 0.5) / w as f64 * wm;
            c[vax] = mx[vax] - (py as f64 + 0.5) / h as f64 * hm;
            c[oax] = at[oax];
            if sdf.sample(Vec3::new(c[0], c[1], c[2])) < 0.0 {
                img.put_pixel(pxi, py, AIR);
            }
        }
    }
    let mxp = ((mk[hax] - mn[hax]) / wm * w as f64) as i64;
    let myp = ((mx[vax] - mk[vax]) / hm * h as f64) as i64;
    for dy in -4..=4i64 {
        for dxp in -4..=4i64 {
            let r2 = dxp * dxp + dy * dy;
            if (6..=16).contains(&r2) {
                let (x, y) = (mxp + dxp, myp + dy);
                if x >= 0 && y >= 0 && (x as u32) < w && (y as u32) < h {
                    img.put_pixel(x as u32, y as u32, MARK);
                }
            }
        }
    }
    img
}

fn main() -> anyhow::Result<()> {
    let in_path = std::env::args()
        .nth(1)
        .expect("usage: prep_maxillary <in.stl> <out.stl> [clip box]");
    let out_path = std::env::args()
        .nth(2)
        .expect("usage: prep_maxillary <in.stl> <out.stl> [clip box]");

    // Tuned default clip box (mm, STL-native frame). Faces chosen to sit in bone
    // and sever the medial ostium and the superior ethmoid bridge.
    let clip = Aabb::new(
        Vec3::new(arg_f64(3, 8.0), arg_f64(4, -22.0), arg_f64(5, 6.0)),
        Vec3::new(arg_f64(6, 27.0), arg_f64(7, 24.0), arg_f64(8, 38.0)),
    );

    // --- 1. exact full-mesh SDF over the generous build box (cached) ---
    eprintln!("loading {in_path} ...");
    let raw = TriMesh::load(&in_path)?;
    let dec = raw.decimated_vertex_cluster(DEC_CELL);
    let ntris = dec.triangle_count();
    eprintln!("decimated to {ntris} tris ({} verts)", dec.vertices.len());

    let sdf = if let Some(s) = cache_load(DEC_CELL, ntris) {
        eprintln!("loaded cached SDF {}x{}x{}", s.nx, s.ny, s.nz);
        s
    } else {
        eprintln!(
            "building exact SDF over {}..{} dx={BUILD_DX} (slow, one-off) ...",
            fmt(BUILD_MIN),
            fmt(BUILD_MAX)
        );
        let t0 = std::time::Instant::now();
        let s = Sdf::from_mesh_in_bounds(&dec, BUILD_DX, Aabb::new(BUILD_MIN, BUILD_MAX));
        eprintln!("built {}x{}x{} in {:?}", s.nx, s.ny, s.nz, t0.elapsed());
        cache_save(&s, DEC_CELL, ntris).ok();
        s
    };
    let (nx, ny, nz, dx, origin) = (sdf.nx, sdf.ny, sdf.nz, sdf.dx, sdf.origin);

    // --- 2. clip to the maxillary box (outside -> bone) ---
    let mut clipped = sdf.data.clone();
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let p = origin + Vec3::new(i as f64 * dx, j as f64 * dx, k as f64 * dx);
                if !clip.contains(p) {
                    clipped[(k * ny + j) * nx + i] = 1.0;
                }
            }
        }
    }

    // --- 3. flood-fill into pockets, report them ---
    let (label, pockets) = flood(&clipped, nx, ny, nz, origin, dx);
    let mut order: Vec<usize> = (0..pockets.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(pockets[i].count));
    println!("\nclip box {} .. {}", fmt(clip.min), fmt(clip.max));
    println!("{} pockets within clip. Top 8 by volume:", pockets.len());
    println!(
        "  {:>3} {:>8} {:>22} {:>6}  bbox",
        "#", "vol(ml)", "centroid", "d2ndl"
    );
    for (rank, &pi) in order.iter().take(8).enumerate() {
        let p = &pockets[pi];
        let d = (NEEDLE - NEEDLE.clamp(p.bmin, p.bmax)).length();
        println!(
            "  {:>3} {:>8.3} {:>22} {:>6.1}  {}..{}{}",
            rank,
            p.volume_ml(dx),
            fmt(p.centroid()),
            d,
            fmt(p.bmin),
            fmt(p.bmax),
            if p.touches { " [open]" } else { " [closed]" }
        );
    }

    // --- 4. choose the maxillary pocket: the largest one whose bbox reaches the
    // needle entry (within 3 mm). Falls back to the largest pocket overall. ---
    let pick = order
        .iter()
        .copied()
        .find(|&pi| (NEEDLE - NEEDLE.clamp(pockets[pi].bmin, pockets[pi].bmax)).length() <= 3.0)
        .or_else(|| order.first().copied())
        .expect("no air pockets in clip box");
    let chosen = &pockets[pick];
    println!(
        "\nchosen maxillary pocket: vol {:.3} ml  centroid {}  {}",
        chosen.volume_ml(dx),
        fmt(chosen.centroid()),
        if chosen.touches {
            "[severed at a clip face]"
        } else {
            "[fully enclosed]"
        }
    );

    // --- 5. mask every other pocket to bone, leaving a single cavity ---
    let mut masked_data = clipped.clone();
    for idx in 0..masked_data.len() {
        if masked_data[idx] < 0.0 && label[idx] != chosen.label {
            masked_data[idx] = 1.0;
        }
    }
    let masked = Sdf {
        origin,
        dx,
        nx,
        ny,
        nz,
        data: masked_data,
    };

    // --- 6. polygonise (Naive Surface Nets) -> watertight, outward-wound ---
    let mesh = sdr::surface_nets::surface_nets(origin, dx, (nx, ny, nz), |p| masked.sample(p));
    let mesh = mesh.largest_component();
    let vol = mesh.signed_volume();
    let (edges, boundary, nonman) = manifold_report(&mesh);
    println!(
        "\nsurface-nets mesh: {} tris, {} verts\n  signed volume = {:+.0} mm^3 ({})\n  edges={edges} boundary={boundary} non-manifold={nonman}  => {}",
        mesh.triangle_count(), mesh.vertices.len(), vol,
        if vol > 0.0 { "outward winding OK" } else { "INWARD — would invert the SDF sign!" },
        if boundary == 0 && nonman == 0 { "WATERTIGHT" } else { "NOT closed" }
    );
    mesh.save(&out_path)?;
    println!("wrote {out_path}");

    // --- needle placement: nearest interior point to the recovered entry,
    // nudged a couple of mm toward the cavity centroid so it is safely inside. ---
    let mut best = (f64::INFINITY, NEEDLE);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let idx = (k * ny + j) * nx + i;
                if masked.data[idx] < 0.0 {
                    let p = origin + Vec3::new(i as f64 * dx, j as f64 * dx, k as f64 * dx);
                    let d = (p - NEEDLE).length();
                    if d < best.0 {
                        best = (d, p);
                    }
                }
            }
        }
    }
    let inward = (chosen.centroid() - best.1).normalize_or_zero();
    let mut tip = best.1 + inward * 2.0;
    // Guarantee the tip samples inside (step further in if a 2 mm nudge overshot).
    for _ in 0..8 {
        if masked.sample(tip) < -0.3 {
            break;
        }
        tip += inward * 1.0;
    }
    let jet_into_air = masked.sample(tip + NEEDLE_AXIS.normalize_or_zero() * 2.0) < 0.0;
    println!(
        "\nrecovered entry {} is {:.1} mm from the nearest cavity voxel.",
        fmt(NEEDLE),
        best.0
    );
    println!(
        "suggested needle.tip_mm = [{:.2}, {:.2}, {:.2}]  (phi {:+.2} mm)",
        tip.x,
        tip.y,
        tip.z,
        masked.sample(tip)
    );
    println!(
        "recovered axis {} fires {} the cavity.",
        fmt(NEEDLE_AXIS),
        if jet_into_air {
            "INTO"
        } else {
            "toward a wall of"
        }
    );

    // --- quick visual sanity slices through the chosen pocket centroid ---
    let outdir = std::env::var("OUTDIR").unwrap_or_else(|_| "/tmp/sinus_out".into());
    std::fs::create_dir_all(&outdir).ok();
    let c = chosen.centroid();
    slice_png(&masked, 0, 1, 2, c, tip, 540)
        .save(format!("{outdir}/maxillary_xy.png"))
        .ok();
    slice_png(&masked, 0, 2, 1, c, tip, 540)
        .save(format!("{outdir}/maxillary_xz.png"))
        .ok();
    slice_png(&masked, 1, 2, 0, c, tip, 540)
        .save(format!("{outdir}/maxillary_yz.png"))
        .ok();
    println!("wrote sanity slices to {outdir}/maxillary_*.png");
    Ok(())
}
