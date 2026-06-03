//! One-off analysis helper (kept per the repo's "experiments live in the tree"
//! convention): load the raw `sinuses_smooth.stl` airway scan and map the
//! Blender `prepare.blend` object transforms into the STL's own coordinate
//! frame, so we can reproduce exactly where the irrigation needle enters and
//! which sub-box is the maxillary sinus.
//!
//! Run: `cargo run --release --example inspect_real -- /path/to/sinuses_smooth.stl`
//!
//! Blender object state (extracted from prepare.blend via experiments/blend_parse.py):
//!   sinuses_smooth: loc=(8.62187,-16.25081,26.75974) rot_x=-0.96002 scale=0.04183
//!   Icosphere(emitter): loc=(9.65111,-15.22509,26.67662)
//!   FLIP Domain: loc=(9.71984,-14.92206,27.23915) size=(0.15299,0.21845,0.40712)

use sdr::math::Vec3;
use sdr::mesh::TriMesh;

// --- Blender transform of the sinus object (world = loc + Rx(rot_x)*scale*local) ---
const SIN_LOC: Vec3 = Vec3::new(8.62187, -16.25081, 26.75974);
const SIN_ROT_X: f64 = -0.96002;
const SIN_SCALE: f64 = 0.04183;

// Emitter (needle source) and domain, in Blender world space.
const ICO_LOC: Vec3 = Vec3::new(9.65111, -15.22509, 26.67662);
const DOM_LOC: Vec3 = Vec3::new(9.71984, -14.92206, 27.23915);
const DOM_SIZE: Vec3 = Vec3::new(0.15299, 0.21845, 0.40712);

fn rot_x(p: Vec3, a: f64) -> Vec3 {
    let (s, c) = a.sin_cos();
    Vec3::new(p.x, p.y * c - p.z * s, p.y * s + p.z * c)
}

/// world -> STL-local: local = (1/scale) * Rx(-rot_x) * (world - loc)
fn world_to_local(w: Vec3) -> Vec3 {
    rot_x(w - SIN_LOC, -SIN_ROT_X) / SIN_SCALE
}

fn fmt(v: Vec3) -> String {
    format!("({:.3}, {:.3}, {:.3})", v.x, v.y, v.z)
}

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("usage: inspect_real <stl>");
    eprintln!("loading {path} ...");
    let mesh = TriMesh::load(&path)?;
    let bb = mesh.bounds();
    let size = bb.size();
    println!("=== raw STL ===");
    println!("triangles : {}", mesh.triangle_count());
    println!("vertices  : {}", mesh.vertices.len());
    println!("bounds min: {}", fmt(bb.min));
    println!("bounds max: {}", fmt(bb.max));
    println!("size      : {}  (native units)", fmt(size));
    println!("signed vol: {:.3} (native^3)", mesh.signed_volume());

    // Map the emitter (needle entry) into STL-local coordinates.
    let ico_local = world_to_local(ICO_LOC);
    println!("\n=== needle emitter (Icosphere) in STL coords ===");
    println!("world : {}", fmt(ICO_LOC));
    println!("local : {}", fmt(ico_local));
    println!("inside STL bbox? {}", bb.contains(ico_local));

    // Map the FLIP domain box (8 corners) into STL coords -> crop region.
    println!("\n=== FLIP domain box in STL coords (crop region) ===");
    let mut cmin = Vec3::splat(f64::INFINITY);
    let mut cmax = Vec3::splat(f64::NEG_INFINITY);
    for sx in [-1.0, 1.0] {
        for sy in [-1.0, 1.0] {
            for sz in [-1.0, 1.0] {
                let corner = DOM_LOC + Vec3::new(sx * DOM_SIZE.x, sy * DOM_SIZE.y, sz * DOM_SIZE.z);
                let l = world_to_local(corner);
                cmin = cmin.min(l);
                cmax = cmax.max(l);
            }
        }
    }
    println!("crop min : {}", fmt(cmin));
    println!("crop max : {}", fmt(cmax));
    println!("crop size: {}", fmt(cmax - cmin));

    // How many triangles fall (any vertex) inside the crop box?
    let mut inside = 0usize;
    for i in 0..mesh.triangle_count() {
        let [a, b, c] = mesh.triangle(i);
        let center = (a + b + c) / 3.0;
        if center.x >= cmin.x
            && center.x <= cmax.x
            && center.y >= cmin.y
            && center.y <= cmax.y
            && center.z >= cmin.z
            && center.z <= cmax.z
        {
            inside += 1;
        }
    }
    println!("triangles with centroid in crop box: {inside}");

    // Emitter position relative to crop box (fractional), to sanity-check side.
    let frac = Vec3::new(
        (ico_local.x - cmin.x) / (cmax.x - cmin.x),
        (ico_local.y - cmin.y) / (cmax.y - cmin.y),
        (ico_local.z - cmin.z) / (cmax.z - cmin.z),
    );
    println!("emitter fractional pos in crop box: {}", fmt(frac));
    Ok(())
}
