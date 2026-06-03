//! Re-render saved particle frames as the amber gravity "hero" view, *offline* —
//! without re-running the (minutes-long) simulation.
//!
//! `simulate` writes each frame's particle cloud to `frames/particles_*.vtk`
//! (positions only). Those positions, plus the scene's solid SDF / needle /
//! gravity (rebuilt from the same TOML), are everything [`render_slice`] needs.
//! So we can retune the pool look and regenerate the PNGs in seconds, and the
//! committed hero image stays reproducible from the committed `.vtk` frames.
//!
//! Run:
//!   cargo run --release --example render_pool -- \
//!       examples/maxillary_real.toml out/maxillary/frames out/maxillary/pool [px]
//!
//! Writes `pool_000000.png …` into the output directory (one per input frame).

use anyhow::{bail, Context, Result};
use sdr::output::read_points_vtk;
use sdr::render::{render_slice, save_png, SlicePlane};
use sdr::scene::SceneConfig;
use std::path::PathBuf;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        bail!(
            "usage: render_pool <scene.toml> <frames_dir> <out_dir> [preview_px]\n\
             re-renders frames/particles_*.vtk as the amber gravity pool view"
        );
    }
    let scene = &args[0];
    let frames_dir = PathBuf::from(&args[1]);
    let out_dir = PathBuf::from(&args[2]);
    let px: u32 = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(360);

    // Rebuild the scene to get the solid SDF, needle and gravity. The fluid
    // particles come from the saved frames instead of a fresh simulation.
    let mut built = SceneConfig::from_toml_file(scene)
        .with_context(|| format!("loading scene {scene}"))?
        .build()?;
    let plane = SlicePlane::projected_along_gravity(built.solver.fluid.gravity);

    // Collect frames/particles_*.vtk, sorted by name (== frame order).
    let mut frames: Vec<PathBuf> = std::fs::read_dir(&frames_dir)
        .with_context(|| format!("reading {}", frames_dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension().map(|e| e == "vtk").unwrap_or(false)
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("particles_"))
                    .unwrap_or(false)
        })
        .collect();
    frames.sort();
    if frames.is_empty() {
        bail!("no particles_*.vtk frames in {}", frames_dir.display());
    }

    std::fs::create_dir_all(&out_dir).with_context(|| format!("creating {}", out_dir.display()))?;

    for (i, frame) in frames.iter().enumerate() {
        let positions = read_points_vtk(frame)?;
        let n = positions.len();
        // Swap in this frame's particles (velocities unused by the amber pool).
        built.solver.particles.positions = positions;
        built.solver.particles.velocities = vec![sdr::math::Vec3::ZERO; n];

        let img = render_slice(&built.solver, &plane, px, 0.0);
        let out = out_dir.join(format!("pool_{i:06}.png"));
        save_png(&img, &out)?;
        println!("{} -> {} ({n} particles)", frame.display(), out.display());
    }
    Ok(())
}
