//! Quantitative locator for the maxillary air cavity around the needle entry.
//!
//! Eyeballing thin SDF slices is ambiguous, so this tool *measures*:
//!   * the exact signed distance at the recovered needle point;
//!   * a 6-direction ray-march from the needle that reports where the cavity
//!     wall is and how wide the air span is in each direction;
//!   * a flood-fill of the air voxels into connected pockets, each reported
//!     with voxel count, volume, centroid and bounding box, so the maxillary
//!     sinus (a large compact pocket near the needle) is unmistakable.
//!
//! Run (lengths in mm, STL-native frame):
//!   cargo run --release --example locate_cavity -- <in.stl> \
//!       [cell] [dx] [minx miny minz maxx maxy maxz]

use image::{Rgb, RgbImage};
use sdr::math::{Aabb, Vec3};
use sdr::mesh::TriMesh;
use sdr::sdf::Sdf;

const AIR: Rgb<u8> = Rgb([232, 236, 244]);
const BONE: Rgb<u8> = Rgb([40, 40, 50]);
const MARK: Rgb<u8> = Rgb([255, 70, 70]);

/// One in-plane slice; `oax` out-of-plane, `hax`/`vax` in-plane (v flipped up).
fn slice_png(sdf: &Sdf, hax: usize, vax: usize, oax: usize, at: Vec3, px: u32) -> RgbImage {
    let bb = sdf.bounds();
    let (mn, mx, at) = (bb.min.to_array(), bb.max.to_array(), at.to_array());
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
    let mxp = ((at[hax] - mn[hax]) / wm * w as f64) as i64;
    let myp = ((mx[vax] - at[vax]) / hm * h as f64) as i64;
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

// --- Blender transform of the sinus object (same as inspect_real) ---
const SIN_LOC: Vec3 = Vec3::new(8.62187, -16.25081, 26.75974);
const SIN_ROT_X: f64 = -0.96002;
const SIN_SCALE: f64 = 0.04183;
const ICO_LOC: Vec3 = Vec3::new(9.65111, -15.22509, 26.67662);
const DOM_LOC: Vec3 = Vec3::new(9.71984, -14.92206, 27.23915);
const DOM_SIZE: Vec3 = Vec3::new(0.15299, 0.21845, 0.40712);

fn rot_x(p: Vec3, a: f64) -> Vec3 {
    let (s, c) = a.sin_cos();
    Vec3::new(p.x, p.y * c - p.z * s, p.y * s + p.z * c)
}
fn world_to_local(w: Vec3) -> Vec3 {
    rot_x(w - SIN_LOC, -SIN_ROT_X) / SIN_SCALE
}
fn fmt(v: Vec3) -> String {
    format!("({:.2}, {:.2}, {:.2})", v.x, v.y, v.z)
}
fn arg_f64(i: usize, default: f64) -> f64 {
    std::env::args()
        .nth(i)
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn main() -> anyhow::Result<()> {
    let in_path = std::env::args()
        .nth(1)
        .expect("usage: locate_cavity <in.stl> ...");
    let cell = arg_f64(2, 2.0);
    let dx = arg_f64(3, 1.0);

    let needle = world_to_local(ICO_LOC);
    // FLIP domain box -> local AABB.
    let (mut dmin, mut dmax) = (Vec3::splat(f64::INFINITY), Vec3::splat(f64::NEG_INFINITY));
    for sx in [-1.0, 1.0] {
        for sy in [-1.0, 1.0] {
            for sz in [-1.0, 1.0] {
                let c = world_to_local(
                    DOM_LOC + Vec3::new(sx * DOM_SIZE.x, sy * DOM_SIZE.y, sz * DOM_SIZE.z),
                );
                dmin = dmin.min(c);
                dmax = dmax.max(c);
            }
        }
    }

    // Default analysis box: union of needle neighbourhood and FLIP domain, padded.
    let pad = Vec3::splat(6.0);
    let def_min = needle.min(dmin) - pad;
    let def_max = needle.max(dmax) + pad;
    let min = Vec3::new(
        arg_f64(4, def_min.x),
        arg_f64(5, def_min.y),
        arg_f64(6, def_min.z),
    );
    let max = Vec3::new(
        arg_f64(7, def_max.x),
        arg_f64(8, def_max.y),
        arg_f64(9, def_max.z),
    );

    eprintln!("loading {in_path} ...");
    let raw = TriMesh::load(&in_path)?;
    // cell <= 0 => use the raw mesh (no decimation): the winding number is only
    // *exact* on the original closed surface, so this is the ground truth.
    let dec = if cell > 0.0 {
        let d = raw.decimated_vertex_cluster(cell);
        eprintln!(
            "decimated to {} tris ({} verts)",
            d.triangle_count(),
            d.vertices.len()
        );
        d
    } else {
        eprintln!(
            "using RAW mesh: {} tris (no decimation)",
            raw.triangle_count()
        );
        raw
    };

    println!("needle (local) : {}", fmt(needle));
    println!("domain  min    : {}", fmt(dmin));
    println!("domain  max    : {}", fmt(dmax));
    println!("analysis box   : {} .. {}", fmt(min), fmt(max));

    let t0 = std::time::Instant::now();
    let sdf = Sdf::from_mesh_in_bounds(&dec, dx, Aabb::new(min, max));
    let n_in = sdf.data.iter().filter(|&&v| v < 0.0).count();
    println!(
        "exact SDF {}x{}x{} in {:?}: inside={:.1}%",
        sdf.nx,
        sdf.ny,
        sdf.nz,
        t0.elapsed(),
        100.0 * n_in as f64 / sdf.data.len() as f64
    );

    // --- SDF at needle + 6-direction ray-march ---
    println!(
        "\nphi(needle) = {:+.3} mm  ({})",
        sdf.sample(needle),
        if sdf.sample(needle) < 0.0 {
            "AIR"
        } else {
            "BONE"
        }
    );
    let dirs = [
        ("+x", Vec3::new(1.0, 0.0, 0.0)),
        ("-x", Vec3::new(-1.0, 0.0, 0.0)),
        ("+y", Vec3::new(0.0, 1.0, 0.0)),
        ("-y", Vec3::new(0.0, -1.0, 0.0)),
        ("+z", Vec3::new(0.0, 0.0, 1.0)),
        ("-z", Vec3::new(0.0, 0.0, -1.0)),
    ];
    let step = dx * 0.5;
    println!("ray-march from needle (first air crossing / air span within 30mm):");
    for (name, d) in dirs {
        let mut first_air: Option<f64> = None;
        let mut last_air: Option<f64> = None;
        let mut t = 0.0;
        while t <= 30.0 {
            let phi = sdf.sample(needle + d * t);
            if phi < 0.0 {
                if first_air.is_none() {
                    first_air = Some(t);
                }
                last_air = Some(t);
            }
            t += step;
        }
        match (first_air, last_air) {
            (Some(f), Some(l)) => println!("  {name}: air at {f:5.1}..{l:5.1} mm"),
            _ => println!("  {name}: no air within 30mm"),
        }
    }

    // --- flood-fill air voxels (6-connectivity) into pockets ---
    let (nx, ny, nz) = (sdf.nx, sdf.ny, sdf.nz);
    let air = |i: usize, j: usize, k: usize| sdf.data[(k * ny + j) * nx + i] < 0.0;
    let mut label = vec![u32::MAX; nx * ny * nz];
    let mut pockets: Vec<(usize, Vec3, Vec3, Vec3, bool)> = Vec::new(); // count, sum, min, max, touches_boundary
    let mut next = 0u32;
    let mut stack: Vec<(usize, usize, usize)> = Vec::new();
    for k0 in 0..nz {
        for j0 in 0..ny {
            for i0 in 0..nx {
                let id0 = (k0 * ny + j0) * nx + i0;
                if !air(i0, j0, k0) || label[id0] != u32::MAX {
                    continue;
                }
                // BFS/DFS flood
                let (mut cnt, mut sum) = (0usize, Vec3::ZERO);
                let (mut bmin, mut bmax) =
                    (Vec3::splat(f64::INFINITY), Vec3::splat(f64::NEG_INFINITY));
                let mut touches = false;
                label[id0] = next;
                stack.push((i0, j0, k0));
                while let Some((i, j, k)) = stack.pop() {
                    cnt += 1;
                    let p = sdf.origin + Vec3::new(i as f64 * dx, j as f64 * dx, k as f64 * dx);
                    sum += p;
                    bmin = bmin.min(p);
                    bmax = bmax.max(p);
                    if i == 0 || j == 0 || k == 0 || i == nx - 1 || j == ny - 1 || k == nz - 1 {
                        touches = true;
                    }
                    let push = |ni: usize,
                                nj: usize,
                                nk: usize,
                                st: &mut Vec<(usize, usize, usize)>,
                                lab: &mut Vec<u32>| {
                        let id = (nk * ny + nj) * nx + ni;
                        if air(ni, nj, nk) && lab[id] == u32::MAX {
                            lab[id] = next;
                            st.push((ni, nj, nk));
                        }
                    };
                    if i > 0 {
                        push(i - 1, j, k, &mut stack, &mut label);
                    }
                    if i + 1 < nx {
                        push(i + 1, j, k, &mut stack, &mut label);
                    }
                    if j > 0 {
                        push(i, j - 1, k, &mut stack, &mut label);
                    }
                    if j + 1 < ny {
                        push(i, j + 1, k, &mut stack, &mut label);
                    }
                    if k > 0 {
                        push(i, j, k - 1, &mut stack, &mut label);
                    }
                    if k + 1 < nz {
                        push(i, j, k + 1, &mut stack, &mut label);
                    }
                }
                pockets.push((cnt, sum, bmin, bmax, touches));
                next += 1;
            }
        }
    }

    let vox_ml = dx * dx * dx / 1000.0;
    let mut order: Vec<usize> = (0..pockets.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(pockets[i].0));
    println!(
        "\n{} air pockets (flood-fill, 6-conn). Top 10 by volume:",
        pockets.len()
    );
    println!(
        "  {:>3}  {:>7} {:>8}  {:>22}  {:>6}  bbox",
        "#", "voxels", "vol(ml)", "centroid", "d2ndl"
    );
    for (rank, &pi) in order.iter().take(10).enumerate() {
        let (cnt, sum, bmin, bmax, touches) = &pockets[pi];
        let centroid = *sum / *cnt as f64;
        // distance from needle to this pocket's bbox (0 if inside)
        let cl = needle.clamp(*bmin, *bmax);
        let d = (needle - cl).length();
        println!(
            "  {:>3}  {:>7} {:>8.3}  {:>22}  {:>6.1}  {}..{}{}",
            rank,
            cnt,
            *cnt as f64 * vox_ml,
            fmt(centroid),
            d,
            fmt(*bmin),
            fmt(*bmax),
            if *touches { " [open]" } else { " [closed]" }
        );
    }

    // which pocket is nearest the needle?
    if let Some(&best) = order.iter().min_by(|&&a, &&b| {
        let da = (needle - needle.clamp(pockets[a].2, pockets[a].3)).length();
        let db = (needle - needle.clamp(pockets[b].2, pockets[b].3)).length();
        da.partial_cmp(&db).unwrap()
    }) {
        let (cnt, sum, bmin, bmax, _) = &pockets[best];
        println!(
            "\nnearest pocket to needle: centroid {} vol {:.3} ml bbox {}..{}",
            fmt(*sum / *cnt as f64),
            *cnt as f64 * vox_ml,
            fmt(*bmin),
            fmt(*bmax)
        );
    }

    // Slice images: through the needle and through the largest pocket centroid.
    let outdir = std::env::var("OUTDIR").unwrap_or_else(|_| "/tmp/sinus_out".into());
    std::fs::create_dir_all(&outdir).ok();
    let big = order
        .first()
        .map(|&i| pockets[i].1 / pockets[i].0 as f64)
        .unwrap_or(needle);
    for (tag, at) in [("needle", needle), ("cavity", big)] {
        slice_png(&sdf, 0, 1, 2, at, 540)
            .save(format!("{outdir}/loc_{tag}_xy.png"))
            .ok(); // z fixed (top-down)
        slice_png(&sdf, 0, 2, 1, at, 540)
            .save(format!("{outdir}/loc_{tag}_xz.png"))
            .ok(); // y fixed (coronal)
        slice_png(&sdf, 1, 2, 0, at, 540)
            .save(format!("{outdir}/loc_{tag}_yz.png"))
            .ok(); // x fixed (sagittal)
    }
    println!("\nwrote slices to {outdir}/loc_*.png");
    Ok(())
}
