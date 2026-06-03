//! Doctor-facing scene configuration.
//!
//! Clinicians think in millimetres, millilitres-per-second and needle gauges —
//! not in SI base units, world-space metres or grid cells. This module is the
//! translation layer: a [`SceneConfig`] is a small, well-commented TOML file in
//! clinical units that [`SceneConfig::build`] turns into a ready-to-run
//! [`Solver`] in SI units.
//!
//! A minimal config is enough to get going (everything has a sensible default);
//! a fuller one lets the user place a needle precisely, load a patient mesh, or
//! tune the fluid. See `examples/scene.toml` for an annotated template.

use crate::math::{Aabb, Vec3};
use crate::mesh::TriMesh;
use crate::pressure::SolveParams;
use crate::sdf::Sdf;
use crate::sinus::SinusParams;
use crate::solver::{FluidParams, Needle, Solver};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

const MM: f64 = 1e-3; // millimetres -> metres
const ML_PER_S: f64 = 1e-6; // millilitres/second -> m³/s

/// Top-level scene description (deserialised from TOML).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SceneConfig {
    pub sinus: SinusConfig,
    pub needle: NeedleConfig,
    pub fluid: FluidConfig,
    pub sim: SimConfig,
    /// Optional explicit drainage box (millimetres). If omitted, a parametric
    /// sinus derives one automatically at the socket opening.
    pub outlet: Option<BoxMm>,
}

/// Cavity geometry: either parametric, or a loaded watertight mesh.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SinusConfig {
    /// Path to a closed cavity mesh (`.obj`/`.stl`), in millimetres. When set,
    /// the parametric fields below are ignored.
    pub mesh: Option<String>,
    /// Body half-axes `[anteroposterior, vertical, mediolateral]`, mm.
    pub semi_axes_mm: [f64; 3],
    /// Cavity centre, mm.
    pub center_mm: [f64; 3],
    /// Floor taper (1 = none, 0.6 ≈ pyramidal).
    pub taper: f64,
    /// Wall undulation amplitude (fraction of radius).
    pub bumpiness: f64,
    /// Socket (oroantral communication) position on the floor `[x, z]`, mm.
    pub socket_xz_mm: [f64; 2],
    /// Socket channel radius, mm.
    pub socket_radius_mm: f64,
    /// Socket protrusion below the floor, mm.
    pub socket_depth_mm: f64,
    /// Polygonisation / SDF resolution, mm.
    pub model_resolution_mm: f64,
}

impl Default for SinusConfig {
    fn default() -> Self {
        let p = SinusParams::default();
        SinusConfig {
            mesh: None,
            semi_axes_mm: [p.semi_axes.x / MM, p.semi_axes.y / MM, p.semi_axes.z / MM],
            center_mm: [p.center.x / MM, p.center.y / MM, p.center.z / MM],
            taper: p.taper,
            bumpiness: p.bumpiness,
            socket_xz_mm: [p.socket_xz.0 / MM, p.socket_xz.1 / MM],
            socket_radius_mm: p.socket_radius / MM,
            socket_depth_mm: p.socket_depth / MM,
            model_resolution_mm: p.model_dx / MM,
        }
    }
}

impl SinusConfig {
    /// Convert to SI parametric parameters.
    pub fn to_params(&self) -> SinusParams {
        SinusParams {
            semi_axes: Vec3::new(
                self.semi_axes_mm[0],
                self.semi_axes_mm[1],
                self.semi_axes_mm[2],
            ) * MM,
            center: Vec3::new(self.center_mm[0], self.center_mm[1], self.center_mm[2]) * MM,
            taper: self.taper,
            bumpiness: self.bumpiness,
            socket_xz: (self.socket_xz_mm[0] * MM, self.socket_xz_mm[1] * MM),
            socket_radius: self.socket_radius_mm * MM,
            socket_depth: self.socket_depth_mm * MM,
            model_dx: self.model_resolution_mm * MM,
        }
    }
}

/// Irrigation needle placement and flow.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct NeedleConfig {
    /// Tip position, mm. Omit (or set `auto = true`) to auto-place at the
    /// socket — only possible for a parametric sinus.
    pub tip_mm: Option<[f64; 3]>,
    /// Auto-place the tip at the parametric socket entrance.
    pub auto: bool,
    /// Jet direction (need not be normalised).
    pub axis: [f64; 3],
    /// Inner bore diameter, mm (sets the jet speed for a given flow rate).
    pub diameter_mm: f64,
    /// Volumetric flow rate, millilitres per second.
    pub flow_rate_ml_s: f64,
    /// Irrigate for this many seconds, then stop and let the dose settle into a
    /// pool. Omit to inject for the whole run. `flow_rate_ml_s * inject_s` is the
    /// delivered dose; keep it below the cavity volume for a partial fill.
    pub inject_s: Option<f64>,
}

impl Default for NeedleConfig {
    fn default() -> Self {
        NeedleConfig {
            tip_mm: None,
            auto: true,
            axis: [0.0, 1.0, 0.0],
            diameter_mm: 0.8,
            flow_rate_ml_s: 5.0,
            inject_s: None,
        }
    }
}

/// Fluid and transfer parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FluidConfig {
    pub density_kg_m3: f64,
    /// Gravity vector, m/s².
    pub gravity_m_s2: [f64; 3],
    /// FLIP/PIC blend (1 = lively, 0 = viscous).
    pub flip_ratio: f64,
    pub particles_per_cell: usize,
    /// Volume-control strength that relieves over-packed cells (0 = off, plain
    /// FLIP which clumps; 1 = default). See [`FluidParams::volume_stiffness`].
    pub volume_stiffness: f64,
    /// Redistribute over-packed cells' surplus particles into open cavity cells
    /// (`true` = default). The decisive anti-clumping step for a point jet in a
    /// closed cavity. See [`FluidParams::redistribute`].
    pub redistribute: bool,
}

impl Default for FluidConfig {
    fn default() -> Self {
        let f = FluidParams::default();
        FluidConfig {
            density_kg_m3: f.density,
            gravity_m_s2: [f.gravity.x, f.gravity.y, f.gravity.z],
            flip_ratio: f.flip_ratio,
            particles_per_cell: f.particles_per_cell,
            volume_stiffness: f.volume_stiffness,
            redistribute: f.redistribute,
        }
    }
}

impl FluidConfig {
    pub fn to_params(&self) -> FluidParams {
        FluidParams {
            density: self.density_kg_m3,
            gravity: Vec3::new(
                self.gravity_m_s2[0],
                self.gravity_m_s2[1],
                self.gravity_m_s2[2],
            ),
            flip_ratio: self.flip_ratio,
            particles_per_cell: self.particles_per_cell,
            volume_stiffness: self.volume_stiffness,
            redistribute: self.redistribute,
        }
    }
}

/// Simulation discretisation and output cadence.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SimConfig {
    /// Simulation grid spacing, mm (the cell size the solver runs on).
    pub resolution_mm: f64,
    /// Total simulated time, seconds.
    pub duration_s: f64,
    /// Number of output frames over the duration.
    pub frames: usize,
    /// Max conjugate-gradient iterations per projection.
    pub pcg_iters: usize,
    /// Convergence tolerance for the projection.
    pub pcg_tol: f64,
}

impl Default for SimConfig {
    fn default() -> Self {
        SimConfig {
            resolution_mm: 1.0,
            duration_s: 2.0,
            frames: 60,
            pcg_iters: 200,
            pcg_tol: 1e-6,
        }
    }
}

/// An axis-aligned box in millimetres.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoxMm {
    pub min_mm: [f64; 3],
    pub max_mm: [f64; 3],
}

impl BoxMm {
    fn to_aabb(self) -> Aabb {
        Aabb::new(
            Vec3::new(self.min_mm[0], self.min_mm[1], self.min_mm[2]) * MM,
            Vec3::new(self.max_mm[0], self.max_mm[1], self.max_mm[2]) * MM,
        )
    }
}

/// A fully-constructed, ready-to-run scene.
pub struct BuiltScene {
    pub solver: Solver,
    /// Closed mesh of the cavity (for surfacing / rendering reference).
    pub cavity_mesh: TriMesh,
    /// Interior (air) volume of the empty cavity, m³.
    pub cavity_volume: f64,
    /// Seconds between output frames.
    pub frame_dt: f64,
    /// Number of output frames.
    pub frames: usize,
}

impl SceneConfig {
    /// Parse a scene from a TOML string.
    pub fn from_toml_str(s: &str) -> Result<Self> {
        toml::from_str(s).context("parsing scene TOML")
    }

    /// Load a scene from a TOML file.
    pub fn from_toml_file(path: &str) -> Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading scene file {path}"))?;
        Self::from_toml_str(&text)
    }

    /// Serialise back to a TOML string (useful for writing a template).
    pub fn to_toml_string(&self) -> Result<String> {
        toml::to_string_pretty(self).context("serialising scene TOML")
    }

    /// Validate and convert the config into a runnable [`BuiltScene`].
    pub fn build(&self) -> Result<BuiltScene> {
        if self.sim.resolution_mm <= 0.0 {
            bail!("sim.resolution_mm must be positive");
        }
        if self.fluid.flip_ratio < 0.0 || self.fluid.flip_ratio > 1.0 {
            bail!("fluid.flip_ratio must be in [0, 1]");
        }

        let sim_dx = self.sim.resolution_mm * MM;
        let fluid = self.fluid.to_params();

        // Resolve geometry: parametric or loaded mesh.
        let params = self.sinus.to_params();
        let (cavity_mesh, sdf, auto_needle, auto_outlet) = if let Some(path) = &self.sinus.mesh {
            let mut mesh =
                TriMesh::load(path).with_context(|| format!("loading sinus mesh {path}"))?;
            // Mesh files are authored in millimetres; bring them to metres.
            mesh.scale(MM);
            let sdf_dx = (self.sinus.model_resolution_mm * MM).min(sim_dx);
            let sdf = Sdf::from_mesh(&mesh, sdf_dx, 3.0 * sim_dx);
            (mesh, sdf, None, None)
        } else {
            let mesh = params.generate();
            let bb = params.bounds();
            let dx = self.sinus.model_resolution_mm * MM;
            let size = bb.size();
            let dims = (
                (size.x / dx).ceil() as usize + 1,
                (size.y / dx).ceil() as usize + 1,
                (size.z / dx).ceil() as usize + 1,
            );
            let sdf = Sdf::from_fn(bb.min, dx, dims, |p| params.implicit(p));
            // Derived outlet at the socket opening.
            let (sx, sz) = params.socket_xz;
            let yb = params.socket_bottom_y();
            let r = params.socket_radius + 1.5 * sim_dx;
            let outlet = Aabb::new(
                Vec3::new(sx - r, yb - 2.0 * sim_dx, sz - r),
                Vec3::new(sx + r, yb + 1.5 * sim_dx, sz + r),
            );
            (mesh, sdf, Some(params.suggested_needle()), Some(outlet))
        };

        // Resolve the needle.
        let needle = self.resolve_needle(auto_needle)?;

        let cavity_volume = cavity_mesh.signed_volume().abs();

        let mut solver = Solver::new(sdf, sim_dx, fluid, needle);
        solver.solve = SolveParams {
            max_iters: self.sim.pcg_iters,
            tol: self.sim.pcg_tol,
        };
        solver.outlet = match &self.outlet {
            Some(b) => Some(b.to_aabb()),
            None => auto_outlet,
        };

        let frames = self.sim.frames.max(1);
        let frame_dt = self.sim.duration_s / frames as f64;

        Ok(BuiltScene {
            solver,
            cavity_mesh,
            cavity_volume,
            frame_dt,
            frames,
        })
    }

    fn resolve_needle(&self, auto: Option<(Vec3, Vec3)>) -> Result<Needle> {
        let radius = 0.5 * self.needle.diameter_mm * MM;
        if radius <= 0.0 {
            bail!("needle.diameter_mm must be positive");
        }
        let flow_rate = self.needle.flow_rate_ml_s * ML_PER_S;

        let (tip, axis) = if let Some(t) = self.needle.tip_mm {
            let tip = Vec3::new(t[0], t[1], t[2]) * MM;
            let axis = Vec3::new(
                self.needle.axis[0],
                self.needle.axis[1],
                self.needle.axis[2],
            );
            (tip, axis)
        } else if self.needle.auto {
            match auto {
                Some((tip, axis)) => (tip, axis),
                None => bail!(
                    "needle.auto is set but the sinus is a loaded mesh; specify needle.tip_mm"
                ),
            }
        } else {
            bail!("set either needle.tip_mm or needle.auto = true");
        };

        let axis = axis.normalize_or_zero();
        if axis.length() < 0.5 {
            bail!("needle.axis must be a non-zero direction");
        }
        Ok(Needle {
            tip,
            axis,
            radius,
            flow_rate,
            inject_until: self.needle.inject_s.unwrap_or(f64::INFINITY),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The committed real antrum mesh, resolved against the crate root so the
    /// test runs from any working directory.
    fn real_mesh_path() -> String {
        format!("{}/assets/maxillary_sinus.stl", env!("CARGO_MANIFEST_DIR"))
    }

    /// The real-mesh irrigation scene (same geometry and needle as
    /// `examples/maxillary_real.toml`) but at a deliberately coarse resolution
    /// so the whole mesh → SDF → solver path runs inside CI in a second or two.
    /// The production render uses 0.8 mm; here 3 mm is plenty to exercise the
    /// pipeline and guard reproducibility.
    fn real_mesh_scene() -> SceneConfig {
        let mut cfg = SceneConfig::default();
        cfg.sinus.mesh = Some(real_mesh_path());
        cfg.sinus.model_resolution_mm = 3.0;
        cfg.needle.auto = false;
        cfg.needle.tip_mm = Some([17.31, 7.66, 19.81]);
        cfg.needle.axis = [0.0, -0.98, 0.20];
        cfg.needle.diameter_mm = 0.8;
        cfg.needle.flow_rate_ml_s = 4.0;
        cfg.fluid.gravity_m_s2 = [0.0, 8.03, -5.63];
        cfg.sim.resolution_mm = 3.0;
        cfg.sim.duration_s = 0.05;
        cfg.sim.frames = 2;
        cfg
    }

    #[test]
    fn real_mesh_scene_builds_steps_and_accumulates() {
        let built = real_mesh_scene()
            .build()
            .expect("the committed real-mesh scene should build");
        // Cavity volume is taken straight from the mesh (~2.23 ml), so it is
        // independent of the coarse SDF resolution.
        assert!(
            (1.5e-6..3.0e-6).contains(&built.cavity_volume),
            "cavity volume {} m^3 is far from the ~2.23 ml antrum",
            built.cavity_volume
        );

        let mut solver = built.solver;
        for _ in 0..8 {
            solver.substep(2.0e-4);
        }
        let early = solver.particles.len();
        assert!(early > 0, "the needle never emitted any fluid");

        for _ in 0..8 {
            solver.substep(2.0e-4);
        }
        // This scene has no outlet, so particles can only accumulate — a drop
        // would mean mass was lost.
        assert!(
            solver.particles.len() >= early,
            "particle count fell ({} < {early}): mass was not conserved",
            solver.particles.len()
        );
        assert!(
            solver.particles.positions.iter().all(|p| p.is_finite()),
            "a particle position went non-finite"
        );
        // Bounds enforcement must keep every particle inside the cavity (one
        // cell of wall slack, as in the solver's own tests).
        assert!(
            solver
                .particles
                .positions
                .iter()
                .all(|&p| solver.solid.sample(p) < solver.solid.dx),
            "a particle escaped the cavity walls"
        );
    }

    #[test]
    fn real_mesh_scene_is_bit_reproducible() {
        let run = || {
            let mut s = real_mesh_scene().build().unwrap().solver;
            for _ in 0..16 {
                s.substep(2.0e-4);
            }
            s.particles
        };
        let a = run();
        let b = run();
        assert_eq!(a.len(), b.len(), "particle count is not reproducible");
        for (i, (&pa, &pb)) in a.positions.iter().zip(&b.positions).enumerate() {
            assert_eq!(
                pa.x.to_bits(),
                pb.x.to_bits(),
                "particle[{i}].x not reproducible"
            );
            assert_eq!(
                pa.y.to_bits(),
                pb.y.to_bits(),
                "particle[{i}].y not reproducible"
            );
            assert_eq!(
                pa.z.to_bits(),
                pb.z.to_bits(),
                "particle[{i}].z not reproducible"
            );
        }
    }

    #[test]
    fn default_roundtrips_through_toml() {
        let cfg = SceneConfig::default();
        let s = cfg.to_toml_string().unwrap();
        let back = SceneConfig::from_toml_str(&s).unwrap();
        // A couple of spot checks survive the round trip.
        assert_eq!(back.sim.frames, cfg.sim.frames);
        assert_eq!(back.needle.diameter_mm, cfg.needle.diameter_mm);
    }

    #[test]
    fn minimal_config_uses_defaults() {
        // An almost-empty document should still build (everything defaulted).
        let cfg = SceneConfig::from_toml_str("[sim]\nframes = 4\n").unwrap();
        assert_eq!(cfg.sim.frames, 4);
        let built = cfg.build().unwrap();
        assert!(built.cavity_volume > 0.0);
        assert!(built.frame_dt > 0.0);
        // Auto needle should have been placed inside the cavity.
        assert!(built.solver.solid.sample(built.solver.needle.tip) < 0.0);
    }

    #[test]
    fn unknown_keys_are_rejected() {
        // Typos in a doctor's config should fail loudly, not silently default.
        let err = SceneConfig::from_toml_str("[needle]\nflowrate_ml_s = 3.0\n");
        assert!(err.is_err(), "unknown key should be rejected");
    }

    #[test]
    fn millimetre_conversion_is_correct() {
        let toml = r#"
[sinus]
semi_axes_mm = [20, 18, 15]
[needle]
auto = true
diameter_mm = 1.0
flow_rate_ml_s = 6.0
"#;
        let cfg = SceneConfig::from_toml_str(toml).unwrap();
        let params = cfg.sinus.to_params();
        assert!((params.semi_axes.x - 0.020).abs() < 1e-12);
        let needle = cfg
            .resolve_needle(Some((Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0))))
            .unwrap();
        assert!((needle.radius - 0.0005).abs() < 1e-12);
        assert!((needle.flow_rate - 6.0e-6).abs() < 1e-18);
    }
}
