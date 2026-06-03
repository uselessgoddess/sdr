//! Surface reconstruction via the external [`splashsurf`] tool.
//!
//! The solver emits particle clouds (see [`crate::output`]); `splashsurf` turns
//! a cloud into a smooth triangle mesh via SPH density + Marching Cubes, exactly
//! as in the thesis pipeline. We shell out to the installed `splashsurf` binary
//! rather than re-implementing it.
//!
//! Install with `cargo install splashsurf`.
//!
//! [`splashsurf`]: https://github.com/InteractiveComputerGraphics/splashsurf

use anyhow::{bail, Context, Result};
use std::path::Path;
use std::process::Command;

/// Parameters for a `splashsurf reconstruct` call. Lengths marked "×radius" are
/// in multiples of the particle radius, matching the tool's own convention.
#[derive(Debug, Clone, Copy)]
pub struct ReconParams {
    /// Particle radius, metres (absolute).
    pub particle_radius: f64,
    /// SPH smoothing length, ×radius.
    pub smoothing_length: f64,
    /// Marching-cubes cell size, ×radius.
    pub cube_size: f64,
    /// Iso-surface threshold.
    pub surface_threshold: f64,
}

impl Default for ReconParams {
    fn default() -> Self {
        ReconParams {
            particle_radius: 0.0, // caller must set (see `radius_for_volume`)
            smoothing_length: 2.0,
            cube_size: 0.5,
            surface_threshold: 0.6,
        }
    }
}

/// A physically-grounded default particle radius: the radius of a sphere whose
/// volume equals the volume one solver particle represents. This matches the
/// particle spacing well enough that `splashsurf` produces a watertight surface.
pub fn radius_for_volume(particle_volume: f64) -> f64 {
    (0.75 * particle_volume / std::f64::consts::PI).cbrt()
}

/// Is the `splashsurf` binary available on `PATH`?
pub fn available() -> bool {
    Command::new("splashsurf")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Run `splashsurf reconstruct <input> -o <output> ...`.
///
/// Errors if the binary is missing or the reconstruction fails.
pub fn reconstruct(
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
    params: ReconParams,
) -> Result<()> {
    if params.particle_radius <= 0.0 {
        bail!("particle_radius must be positive (try recon::radius_for_volume)");
    }
    let input = input.as_ref();
    let output = output.as_ref();
    let status = Command::new("splashsurf")
        .arg("reconstruct")
        .arg(input)
        .arg("-o")
        .arg(output)
        .arg(format!("-r={}", params.particle_radius))
        .arg(format!("-l={}", params.smoothing_length))
        .arg(format!("-c={}", params.cube_size))
        .arg(format!("-t={}", params.surface_threshold))
        .status()
        .context("running `splashsurf` (install with `cargo install splashsurf`)")?;
    if !status.success() {
        bail!("splashsurf exited with {status}");
    }
    if !output.exists() {
        bail!("splashsurf reported success but produced no output file");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_volume_radius_is_correct() {
        // Sphere of volume V = 4/3 π r³  ⇒  r = (3V/4π)^(1/3).
        let r = 0.001_f64;
        let v = 4.0 / 3.0 * std::f64::consts::PI * r.powi(3);
        assert!((radius_for_volume(v) - r).abs() < 1e-12);
    }

    #[test]
    fn zero_radius_is_rejected() {
        let err = reconstruct("in.vtk", "out.obj", ReconParams::default());
        assert!(err.is_err());
    }
}
