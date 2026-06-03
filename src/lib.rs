//! `sdr` — a hydrodynamic simulator of maxillary-sinus irrigation.
//!
//! See the crate README for the clinical background. In short: given a 3D model
//! of a maxillary sinus (the air cavity above the upper teeth) and the position
//! of an irrigation needle inserted through an oroantral communication (the
//! socket of an extracted tooth), this crate simulates how irrigation fluid
//! fills, flushes, and drains the cavity, and reports clinically meaningful
//! metrics (fill fraction, wall-contact "wash" coverage, pressure).
//!
//! The fluid solver is a **FLIP/PIC** hybrid on a staggered MAC grid — the same
//! family of method (Zhu & Bridson, *Animating Sand as a Fluid*, SIGGRAPH 2005)
//! referenced by the thesis. Particle output is written in a format that the
//! [`splashsurf`](https://github.com/InteractiveComputerGraphics/splashsurf)
//! surface-reconstruction tool reads directly, so the result can be meshed and
//! rendered in Blender exactly as in the original pipeline.

pub mod grid;
pub mod math;
pub mod mesh;
pub mod metrics;
pub mod output;
pub mod particles;
pub mod pressure;
pub mod recon;
pub mod render;
pub mod scene;
pub mod sdf;
pub mod sinus;
pub mod solver;
pub mod surface_nets;

pub use grid::MacGrid;
pub use math::{Aabb, Vec3};
pub use mesh::TriMesh;
pub use metrics::{FrameMetrics, MetricsCollector, Summary};
pub use particles::ParticleSet;
pub use pressure::{project, project_capturing, Cell, SolveParams, SolveReport};
pub use scene::{BuiltScene, SceneConfig};
pub use sdf::Sdf;
pub use solver::{FluidParams, Needle, Solver};
