//! `sdr` command-line interface.
//!
//! Subcommands cover the whole pipeline a clinician or researcher needs:
//!
//! * `init` — write an annotated template scene configuration.
//! * `generate-sinus` — build the cavity mesh (parametric or from a scene) for inspection / Blender import.
//! * `simulate` — run the irrigation simulation: particle frames, clinical metrics, and preview images.
//! * `surface` — reconstruct a smooth surface from a particle frame via `splashsurf`.
//!
//! Everything is configured in clinical units (mm, ml/s) via a small TOML file;
//! see `sdr init`.

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use sdr::metrics::MetricsCollector;
use sdr::recon::{self, ReconParams};
use sdr::render::{self, SlicePlane};
use sdr::scene::SceneConfig;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "sdr", version, about = "Hydrodynamic simulator of maxillary-sinus irrigation")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Write a template scene configuration (clinical units: mm, ml/s).
    Init {
        /// Output path for the template.
        #[arg(short, long, default_value = "scene.toml")]
        output: String,
    },
    /// Generate the cavity mesh and save it (.obj/.stl).
    GenerateSinus {
        /// Scene file (uses its `[sinus]` section). Defaults to the built-in sinus.
        #[arg(short, long)]
        scene: Option<String>,
        /// Output mesh path.
        #[arg(short, long, default_value = "sinus.obj")]
        output: String,
    },
    /// Run an irrigation simulation and write frames, metrics and previews.
    Simulate {
        /// Scene file. Defaults to the built-in parametric sinus.
        #[arg(short, long)]
        scene: Option<String>,
        /// Output directory.
        #[arg(short, long, default_value = "out")]
        out_dir: String,
        /// Particle file format for splashsurf.
        #[arg(long, value_enum, default_value_t = ParticleFormat::Vtk)]
        format: ParticleFormat,
        /// Longest edge of preview images, pixels (0 disables previews).
        #[arg(long, default_value_t = 600)]
        preview_px: u32,
        /// Reconstruct the final frame's surface via splashsurf.
        #[arg(long)]
        reconstruct: bool,
        /// Fail if peak wall coverage is below this fraction (0–1).
        #[arg(long)]
        min_coverage: Option<f64>,
        /// Fail if peak fill fraction is below this fraction.
        #[arg(long)]
        min_fill: Option<f64>,
        /// Reduce logging.
        #[arg(short, long)]
        quiet: bool,
    },
    /// Reconstruct a surface mesh from a particle file via splashsurf.
    Surface {
        /// Input particle file (.vtk/.xyz/.ply/.bgeo).
        #[arg(short, long)]
        input: String,
        /// Output mesh path (default: `<input>_surface.obj`).
        #[arg(short, long)]
        output: Option<String>,
        /// Particle radius, millimetres.
        #[arg(short, long)]
        radius_mm: f64,
        /// Smoothing length (× radius).
        #[arg(long, default_value_t = 2.0)]
        smoothing_length: f64,
        /// Marching-cubes cell size (× radius).
        #[arg(long, default_value_t = 0.5)]
        cube_size: f64,
        /// Iso-surface threshold.
        #[arg(long, default_value_t = 0.6)]
        surface_threshold: f64,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum ParticleFormat {
    Vtk,
    Xyz,
}

impl ParticleFormat {
    fn ext(self) -> &'static str {
        match self {
            ParticleFormat::Vtk => "vtk",
            ParticleFormat::Xyz => "xyz",
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Init { output } => cmd_init(&output),
        Cmd::GenerateSinus { scene, output } => cmd_generate_sinus(scene.as_deref(), &output),
        Cmd::Simulate { scene, out_dir, format, preview_px, reconstruct, min_coverage, min_fill, quiet } => {
            cmd_simulate(SimulateArgs {
                scene: scene.as_deref(),
                out_dir: &out_dir,
                format,
                preview_px,
                reconstruct,
                min_coverage,
                min_fill,
                quiet,
            })
        }
        Cmd::Surface { input, output, radius_mm, smoothing_length, cube_size, surface_threshold } => cmd_surface(
            &input,
            output.as_deref(),
            radius_mm,
            smoothing_length,
            cube_size,
            surface_threshold,
        ),
    }
}

/// Load a scene file, or the built-in default if none is given.
fn load_scene(scene: Option<&str>) -> Result<SceneConfig> {
    match scene {
        Some(path) => SceneConfig::from_toml_file(path),
        None => Ok(SceneConfig::default()),
    }
}

fn cmd_init(output: &str) -> Result<()> {
    let cfg = SceneConfig::default();
    let text = cfg.to_toml_string()?;
    std::fs::write(output, &text).with_context(|| format!("writing {output}"))?;
    println!("Wrote template scene to {output}");
    Ok(())
}

fn cmd_generate_sinus(scene: Option<&str>, output: &str) -> Result<()> {
    let cfg = load_scene(scene)?;
    let built = cfg.build()?;
    built.cavity_mesh.save(output).with_context(|| format!("saving mesh to {output}"))?;
    println!(
        "Wrote cavity mesh to {output}  ({} triangles, interior volume {:.2} ml)",
        built.cavity_mesh.triangle_count(),
        built.cavity_volume * 1.0e6
    );
    Ok(())
}

struct SimulateArgs<'a> {
    scene: Option<&'a str>,
    out_dir: &'a str,
    format: ParticleFormat,
    preview_px: u32,
    reconstruct: bool,
    min_coverage: Option<f64>,
    min_fill: Option<f64>,
    quiet: bool,
}

fn cmd_simulate(args: SimulateArgs) -> Result<()> {
    let cfg = load_scene(args.scene)?;
    let mut built = cfg.build()?;
    let out = Path::new(args.out_dir);
    let frames_dir = out.join("frames");
    std::fs::create_dir_all(&frames_dir).with_context(|| format!("creating {}", frames_dir.display()))?;
    let preview_dir = out.join("preview");
    if args.preview_px > 0 {
        std::fs::create_dir_all(&preview_dir)?;
    }

    if !args.quiet {
        eprintln!(
            "Cavity volume {:.2} ml | grid {}x{}x{} @ {:.2} mm | {} frames over {:.2} s",
            built.cavity_volume * 1.0e6,
            built.solver.grid.nx,
            built.solver.grid.ny,
            built.solver.grid.nz,
            built.solver.grid.dx * 1e3,
            built.frames,
            built.frame_dt * built.frames as f64,
        );
    }

    let mut metrics = MetricsCollector::new(&built.solver, built.cavity_volume);
    // Keep the colour scale constant across frames for a comparable animation.
    let vmax = built.solver.needle.jet_speed();
    let slice = SlicePlane::xy(built.solver.needle.tip.z, built.solver.grid.dx);
    let mut last_frame_file: Option<PathBuf> = None;

    for f in 0..built.frames {
        built.solver.step(built.frame_dt);
        let m = metrics.record(&built.solver, f);

        // Particle cloud for this frame.
        let pf = sdr::output::frame_path(&frames_dir, "particles", f, args.format.ext());
        match args.format {
            ParticleFormat::Vtk => sdr::output::write_points_vtk(&pf, &built.solver.particles.positions)?,
            ParticleFormat::Xyz => sdr::output::write_points_xyz(&pf, &built.solver.particles.positions)?,
        }
        last_frame_file = Some(pf);

        // Preview slice.
        if args.preview_px > 0 {
            let img = render::render_slice(&built.solver, &slice, args.preview_px, vmax);
            let pp = sdr::output::frame_path(&preview_dir, "slice", f, "png");
            render::save_png(&img, &pp)?;
        }

        if !args.quiet {
            eprintln!(
                "frame {f:>4}/{}: t={:.3}s  particles={:>6}  fill={:>5.1}%  wash={:>5.1}%  p_wall={:>6.1} mmHg  pcg={}",
                built.frames,
                m.time_s,
                m.particles,
                m.fill_fraction * 100.0,
                m.wall_coverage * 100.0,
                m.peak_wall_pressure_mmhg,
                m.pcg_iters,
            );
        }
    }

    // Metrics outputs.
    metrics.write_json(out.join("metrics.json"))?;
    metrics.write_csv(out.join("metrics.csv"))?;
    let summary = metrics.summary();
    std::fs::write(out.join("summary.json"), serde_json::to_string_pretty(&summary)?)?;

    // Final time-series plot.
    if args.preview_px > 0 {
        let plot = render::render_timeseries(metrics.frames(), 720, 360);
        render::save_png(&plot, out.join("metrics.png"))?;
    }

    println!("\n=== irrigation summary ===");
    println!("peak fill fraction : {:>6.1} %", summary.peak_fill_fraction * 100.0);
    println!("peak wall coverage : {:>6.1} %  (flushing effectiveness)", summary.peak_wall_coverage * 100.0);
    println!(
        "peak membrane load : {:>6.1} mmHg ({:.0} Pa)  (focal jet impingement — over-pressure risk)",
        summary.peak_membrane_pressure_mmhg, summary.peak_membrane_pressure_pa
    );
    println!(
        "mean membrane load : {:>6.1} mmHg ({:.0} Pa)  (typical broad load on the lining)",
        summary.mean_membrane_pressure_mmhg, summary.mean_membrane_pressure_pa
    );
    println!("drained volume     : {:>6.2} ml", summary.total_drained_ml);
    println!("mass balance       : {:>6.3}  (1.0 = conserved)", summary.mass_balance);
    println!("outputs written to : {}", out.display());

    // Optional surface reconstruction of the final frame.
    if args.reconstruct {
        if let Some(input) = &last_frame_file {
            if !recon::available() {
                bail!("--reconstruct requested but `splashsurf` is not on PATH (install: cargo install splashsurf)");
            }
            let surf_dir = out.join("surface");
            std::fs::create_dir_all(&surf_dir)?;
            let output = surf_dir.join("surface_final.obj");
            let params = ReconParams {
                particle_radius: recon::radius_for_volume(built.solver.particle_volume()),
                ..ReconParams::default()
            };
            if !args.quiet {
                eprintln!("Reconstructing final surface (r = {:.4} mm)...", params.particle_radius * 1e3);
            }
            recon::reconstruct(input, &output, params)?;
            println!("surface mesh       : {}", output.display());
        }
    }

    // Self-validation gates for CI.
    if let Some(min) = args.min_fill {
        if summary.peak_fill_fraction < min {
            bail!("peak fill fraction {:.3} < required {:.3}", summary.peak_fill_fraction, min);
        }
    }
    if let Some(min) = args.min_coverage {
        if summary.peak_wall_coverage < min {
            bail!("peak wall coverage {:.3} < required {:.3}", summary.peak_wall_coverage, min);
        }
    }
    Ok(())
}

fn cmd_surface(
    input: &str,
    output: Option<&str>,
    radius_mm: f64,
    smoothing_length: f64,
    cube_size: f64,
    surface_threshold: f64,
) -> Result<()> {
    if !recon::available() {
        bail!("`splashsurf` is not on PATH (install: cargo install splashsurf)");
    }
    let out: PathBuf = match output {
        Some(o) => PathBuf::from(o),
        None => {
            let stem = Path::new(input).file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "particles".into());
            Path::new(input).with_file_name(format!("{stem}_surface.obj"))
        }
    };
    let params = ReconParams {
        particle_radius: radius_mm * 1e-3,
        smoothing_length,
        cube_size,
        surface_threshold,
    };
    recon::reconstruct(input, &out, params)?;
    println!("Wrote surface mesh to {}", out.display());
    Ok(())
}
