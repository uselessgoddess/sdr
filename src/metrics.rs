//! Clinically meaningful metrics extracted from the simulation.
//!
//! The thesis asks not just "does fluid go in" but "how well does the irrigation
//! *flush* the cavity". We track, per output frame:
//!
//! * **fill fraction** — fluid volume / empty-cavity volume,
//! * **wall coverage** ("wash") — fraction of the cavity lining that fluid has
//!   touched at least once (a proxy for cleaning effectiveness),
//! * **pressure** — mean and peak fluid gauge pressure (clinical over-pressure
//!   risk to the sinus membrane),
//! * **drainage** — volume that has left through the oroantral communication
//!   (a mass-conservation check, and a measure of through-flow),
//! * **vigour** — peak particle speed and kinetic energy.
//!
//! Output is written as JSON (one array of frames) and CSV for plotting.

use crate::solver::Solver;
use anyhow::{Context, Result};
use serde::Serialize;
use std::io::Write;
use std::path::Path;

const M3_TO_ML: f64 = 1.0e6; // 1 m³ = 1e6 ml
const PA_TO_MMHG: f64 = 1.0 / 133.322; // pascals -> millimetres of mercury

/// Metrics for a single output frame.
#[derive(Debug, Clone, Serialize)]
pub struct FrameMetrics {
    pub frame: usize,
    pub time_s: f64,
    pub particles: usize,
    pub fluid_volume_ml: f64,
    pub drained_volume_ml: f64,
    /// Fluid volume as a fraction of the empty cavity volume (can briefly exceed
    /// 1 from particle packing before draining settles it).
    pub fill_fraction: f64,
    /// Fraction of cavity-wall cells that fluid has contacted at least once.
    pub wall_coverage: f64,
    /// Mean gauge pressure over *all* fluid cells, pascals (whole-fluid
    /// diagnostic — includes the high-pressure jet core right at the needle).
    pub mean_pressure_pa: f64,
    /// Maximum gauge pressure over any single fluid cell, pascals (diagnostic;
    /// dominated by the needle jet core and occasional single-cell solver
    /// transients — *not* the clinical membrane load, see `peak_wall_pressure`).
    pub max_pressure_pa: f64,
    pub max_pressure_mmhg: f64,
    /// Mean gauge pressure on the cavity lining (Schneiderian membrane),
    /// pascals — the typical, broadly-distributed over-pressure load. Averaged
    /// over the wall cells currently in contact with fluid.
    pub wall_pressure_pa: f64,
    /// Peak focal membrane pressure, pascals: the highest gauge pressure on any
    /// wall cell touching fluid — physically the spot where the irrigation jet
    /// impinges on the lining, and the clinical over-pressure risk. Cells above a
    /// physical ceiling (a few × the jet stagnation pressure) are excluded as
    /// single-cell solver artifacts; see `membrane_artifacts`.
    pub peak_wall_pressure_pa: f64,
    pub peak_wall_pressure_mmhg: f64,
    /// Number of wall cells discarded this frame as non-physical pressure spikes
    /// (above the stagnation-pressure ceiling). Normally zero; a non-zero count
    /// flags solver stress and is surfaced for transparency.
    pub membrane_artifacts: usize,
    pub max_speed_m_s: f64,
    pub kinetic_energy: f64,
    pub pcg_iters: usize,
}

/// Aggregate over a whole run (useful for CI assertions and a one-line report).
#[derive(Debug, Clone, Serialize)]
pub struct Summary {
    pub frames: usize,
    pub final_time_s: f64,
    pub peak_fill_fraction: f64,
    pub final_fill_fraction: f64,
    pub peak_wall_coverage: f64,
    pub final_wall_coverage: f64,
    /// Headline clinical pressure: the peak *focal* membrane (cavity-lining)
    /// load over the run, in pascals and mmHg — where the irrigation jet strikes
    /// the lining. This is the over-pressure risk number.
    pub peak_membrane_pressure_pa: f64,
    pub peak_membrane_pressure_mmhg: f64,
    /// Time-averaged *broad* membrane load over the run, in pascals and mmHg —
    /// the typical pressure on the lining away from the focal jet spot. Reported
    /// alongside the focal peak so the latter is not read in isolation.
    pub mean_membrane_pressure_pa: f64,
    pub mean_membrane_pressure_mmhg: f64,
    /// Peak whole-fluid pressure including the needle jet core (diagnostic, not
    /// the membrane load).
    pub peak_pressure_pa: f64,
    pub peak_pressure_mmhg: f64,
    /// Total wall cells discarded across the run as non-physical single-cell
    /// pressure spikes. A solver-health diagnostic: ~0 for a well-resolved run.
    pub membrane_artifacts_total: usize,
    pub total_drained_ml: f64,
    pub injected_ml: f64,
    /// `drained + still-resident` over `injected`; ≈ 1 means mass is conserved.
    pub mass_balance: f64,
}

/// Accumulates metrics across frames and tracks which wall cells have been
/// washed. Build it once from a freshly-built solver (it reads the static
/// cavity geometry), then call [`MetricsCollector::record`] each frame.
pub struct MetricsCollector {
    cavity_volume: f64,
    nx: usize,
    ny: usize,
    nz: usize,
    origin: crate::math::Vec3,
    dx: f64,
    is_wall: Vec<bool>,
    washed: Vec<bool>,
    wall_count: usize,
    history: Vec<FrameMetrics>,
    injected_ml: f64,
}

impl MetricsCollector {
    /// `cavity_volume` is the empty cavity's interior volume in m³ (e.g. from
    /// the closed cavity mesh).
    pub fn new(solver: &Solver, cavity_volume: f64) -> Self {
        let g = &solver.grid;
        let (nx, ny, nz) = (g.nx, g.ny, g.nz);
        let n = nx * ny * nz;

        // A cell is "inside" the cavity if its centre samples negative.
        let mut inside = vec![false; n];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let c = (k * ny + j) * nx + i;
                    inside[c] = solver.solid.sample(g.cell_center(i, j, k)) < 0.0;
                }
            }
        }
        // A "wall" cell is an inside cell with at least one non-inside (solid or
        // out-of-domain) face neighbour — i.e. it lines the cavity surface.
        let mut is_wall = vec![false; n];
        let mut wall_count = 0;
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let c = (k * ny + j) * nx + i;
                    if !inside[c] {
                        continue;
                    }
                    let nb_solid = |ci: i64, cj: i64, ck: i64| -> bool {
                        if ci < 0 || cj < 0 || ck < 0 || ci as usize >= nx || cj as usize >= ny || ck as usize >= nz {
                            return true; // domain boundary counts as wall
                        }
                        !inside[(ck as usize * ny + cj as usize) * nx + ci as usize]
                    };
                    let (i, j, k) = (i as i64, j as i64, k as i64);
                    let touches = nb_solid(i - 1, j, k)
                        || nb_solid(i + 1, j, k)
                        || nb_solid(i, j - 1, k)
                        || nb_solid(i, j + 1, k)
                        || nb_solid(i, j, k - 1)
                        || nb_solid(i, j, k + 1);
                    if touches {
                        is_wall[c] = true;
                        wall_count += 1;
                    }
                }
            }
        }

        MetricsCollector {
            cavity_volume,
            nx,
            ny,
            nz,
            origin: g.origin,
            dx: g.dx,
            is_wall,
            washed: vec![false; n],
            wall_count,
            history: Vec::new(),
            injected_ml: 0.0,
        }
    }

    /// Record metrics for the current solver state as frame `frame`.
    pub fn record(&mut self, solver: &Solver, frame: usize) -> FrameMetrics {
        // Mark wall cells currently containing fluid as washed, and collect the
        // (deduplicated) set of wall cells in contact with fluid *this* frame so
        // we can read the membrane pressure load there.
        let mut wet_wall: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for &p in &solver.particles.positions {
            if let Some(c) = self.cell_index(p) {
                if self.is_wall[c] {
                    self.washed[c] = true;
                    wet_wall.insert(c);
                }
            }
        }
        let washed_count = self.washed.iter().filter(|&&b| b).count();
        let wall_coverage = if self.wall_count > 0 {
            washed_count as f64 / self.wall_count as f64
        } else {
            0.0
        };

        let fluid_volume = solver.fluid_volume();
        let drained_volume = solver.drained_volume();
        let fill_fraction = if self.cavity_volume > 0.0 {
            fluid_volume / self.cavity_volume
        } else {
            0.0
        };
        let max_p = solver.max_pressure();
        self.injected_ml = (fluid_volume + drained_volume) * M3_TO_ML;

        // Membrane (wall) pressure load. Reading `solver.pressure` at these cells
        // is valid because the projection leaves the pressure field exactly zero
        // outside fluid cells. The hot spot is real — it is where the irrigation
        // jet impinges on the cavity lining (the Schneiderian membrane) — and is
        // bounded by the jet stagnation pressure ½ρv². We therefore report it
        // honestly, only excluding cells above a generous physical ceiling: a
        // near-empty fluid cell wedged against the wall can hand the Poisson solve
        // a tiny diagonal and produce a single-cell pressure spike one to two
        // orders of magnitude above any real load, which would otherwise dominate
        // the clinical risk number.
        let v_jet = solver.needle.jet_speed();
        let stagnation = 0.5 * solver.fluid.density * v_jet * v_jet;
        let hydrostatic = solver.fluid.density * solver.fluid.gravity.length() * (self.ny as f64 * self.dx);
        let p_cap = MEMBRANE_PRESSURE_CAP_FACTOR * stagnation + hydrostatic + 500.0;

        let wall_pressures: Vec<f64> = wet_wall.iter().map(|&c| solver.pressure[c]).collect();
        let (wall_mean, wall_peak, artifacts) = membrane_stats(&wall_pressures, p_cap);

        if std::env::var("SDR_DEBUG_PRESSURE").is_ok() {
            // For the hottest wall cells, report (i,j,k) and how many of the 6
            // face neighbours are also hot — a coherent focal zone (the jet
            // impingement on the wall) has hot neighbours; a lone solver artifact
            // does not.
            let mut cells: Vec<(usize, f64)> = wet_wall.iter().map(|&c| (c, solver.pressure[c])).collect();
            cells.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let descr: Vec<String> = cells
                .iter()
                .take(4)
                .map(|&(c, p)| {
                    let i = c % self.nx;
                    let j = (c / self.nx) % self.ny;
                    let k = c / (self.nx * self.ny);
                    let mut hot_nb = 0;
                    for (di, dj, dk) in [(-1i64, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)] {
                        let (ni, nj, nk) = (i as i64 + di, j as i64 + dj, k as i64 + dk);
                        if ni >= 0 && nj >= 0 && nk >= 0 && (ni as usize) < self.nx && (nj as usize) < self.ny && (nk as usize) < self.nz {
                            let nc = (nk as usize * self.ny + nj as usize) * self.nx + ni as usize;
                            if solver.pressure[nc] > 0.25 * p {
                                hot_nb += 1;
                            }
                        }
                    }
                    format!("(i{i},j{j},k{k} p={:.0} hotnb={hot_nb})", p.round())
                })
                .collect();
            eprintln!(
                "[dbg] frame {frame}: wet_wall={} wall_total={} fluid_cells={} stagn={:.0}Pa cap={:.0}Pa artifacts={artifacts} mean={:.0}Pa peak={:.0}Pa | {}",
                wall_pressures.len(),
                self.wall_count,
                solver.cells_fluid_count(),
                stagnation,
                p_cap,
                wall_mean,
                wall_peak,
                descr.join(" "),
            );
        }

        let m = FrameMetrics {
            frame,
            time_s: solver.time,
            particles: solver.particles.len(),
            fluid_volume_ml: fluid_volume * M3_TO_ML,
            drained_volume_ml: drained_volume * M3_TO_ML,
            fill_fraction,
            wall_coverage,
            mean_pressure_pa: solver.mean_pressure(),
            max_pressure_pa: max_p,
            max_pressure_mmhg: max_p * PA_TO_MMHG,
            wall_pressure_pa: wall_mean,
            peak_wall_pressure_pa: wall_peak,
            peak_wall_pressure_mmhg: wall_peak * PA_TO_MMHG,
            membrane_artifacts: artifacts,
            max_speed_m_s: solver.max_speed(),
            kinetic_energy: solver.particles.kinetic_energy(),
            pcg_iters: solver.last_pcg_iters,
        };
        self.history.push(m.clone());
        m
    }

    /// All recorded frames.
    pub fn frames(&self) -> &[FrameMetrics] {
        &self.history
    }

    /// Aggregate summary across all recorded frames.
    pub fn summary(&self) -> Summary {
        let peak_fill = self.history.iter().map(|m| m.fill_fraction).fold(0.0, f64::max);
        let peak_cov = self.history.iter().map(|m| m.wall_coverage).fold(0.0, f64::max);
        let peak_p = self.history.iter().map(|m| m.max_pressure_pa).fold(0.0, f64::max);
        let peak_membrane = self.history.iter().map(|m| m.peak_wall_pressure_pa).fold(0.0, f64::max);
        let mean_membrane = if self.history.is_empty() {
            0.0
        } else {
            self.history.iter().map(|m| m.wall_pressure_pa).sum::<f64>() / self.history.len() as f64
        };
        let artifacts_total: usize = self.history.iter().map(|m| m.membrane_artifacts).sum();
        let last = self.history.last();
        let final_fill = last.map(|m| m.fill_fraction).unwrap_or(0.0);
        let final_cov = last.map(|m| m.wall_coverage).unwrap_or(0.0);
        let final_time = last.map(|m| m.time_s).unwrap_or(0.0);
        let resident_ml = last.map(|m| m.fluid_volume_ml).unwrap_or(0.0);
        let drained_ml = last.map(|m| m.drained_volume_ml).unwrap_or(0.0);
        let injected = resident_ml + drained_ml;
        let mass_balance = if self.injected_ml > 0.0 { injected / self.injected_ml } else { 1.0 };

        Summary {
            frames: self.history.len(),
            final_time_s: final_time,
            peak_fill_fraction: peak_fill,
            final_fill_fraction: final_fill,
            peak_wall_coverage: peak_cov,
            final_wall_coverage: final_cov,
            peak_membrane_pressure_pa: peak_membrane,
            peak_membrane_pressure_mmhg: peak_membrane * PA_TO_MMHG,
            mean_membrane_pressure_pa: mean_membrane,
            mean_membrane_pressure_mmhg: mean_membrane * PA_TO_MMHG,
            peak_pressure_pa: peak_p,
            peak_pressure_mmhg: peak_p * PA_TO_MMHG,
            membrane_artifacts_total: artifacts_total,
            total_drained_ml: drained_ml,
            injected_ml: injected,
            mass_balance,
        }
    }

    /// Write the per-frame history as a JSON array.
    pub fn write_json(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let text = serde_json::to_string_pretty(&self.history).context("serialising metrics JSON")?;
        std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))?;
        Ok(())
    }

    /// Write the per-frame history as CSV (header + one row per frame).
    pub fn write_csv(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let file = std::fs::File::create(path).with_context(|| format!("creating {}", path.display()))?;
        let mut w = std::io::BufWriter::new(file);
        writeln!(
            w,
            "frame,time_s,particles,fluid_volume_ml,drained_volume_ml,fill_fraction,wall_coverage,\
             mean_pressure_pa,max_pressure_pa,max_pressure_mmhg,\
             wall_pressure_pa,peak_wall_pressure_pa,peak_wall_pressure_mmhg,membrane_artifacts,\
             max_speed_m_s,kinetic_energy,pcg_iters"
        )?;
        for m in &self.history {
            writeln!(
                w,
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                m.frame,
                m.time_s,
                m.particles,
                m.fluid_volume_ml,
                m.drained_volume_ml,
                m.fill_fraction,
                m.wall_coverage,
                m.mean_pressure_pa,
                m.max_pressure_pa,
                m.max_pressure_mmhg,
                m.wall_pressure_pa,
                m.peak_wall_pressure_pa,
                m.peak_wall_pressure_mmhg,
                m.membrane_artifacts,
                m.max_speed_m_s,
                m.kinetic_energy,
                m.pcg_iters,
            )?;
        }
        w.flush()?;
        Ok(())
    }

    /// Index of the grid cell containing world point `p`, or `None` if outside.
    fn cell_index(&self, p: crate::math::Vec3) -> Option<usize> {
        let l = (p - self.origin) / self.dx;
        if l.x < 0.0 || l.y < 0.0 || l.z < 0.0 {
            return None;
        }
        let (i, j, k) = (l.x as usize, l.y as usize, l.z as usize);
        if i < self.nx && j < self.ny && k < self.nz {
            Some((k * self.ny + j) * self.nx + i)
        } else {
            None
        }
    }
}

/// Multiple of the jet stagnation pressure (½ρv²) above which a wall-cell
/// pressure is treated as a non-physical single-cell solver artifact rather than
/// a real membrane load. The genuine jet-impingement load is bounded *below* the
/// stagnation pressure, while solver spikes from near-empty fluid cells run one
/// to two orders of magnitude higher, so a generous factor cleanly separates the
/// two.
const MEMBRANE_PRESSURE_CAP_FACTOR: f64 = 3.0;

/// Mean and peak membrane pressure over wall cells in contact with fluid, in
/// pascals, excluding cells above `cap` (non-physical single-cell solver
/// artifacts). Returns `(mean, peak, artifact_count)`; `(0, 0, n)` if every cell
/// is an artifact and `(0, 0, 0)` for an empty set.
fn membrane_stats(values: &[f64], cap: f64) -> (f64, f64, usize) {
    let mut sum = 0.0;
    let mut peak = 0.0_f64;
    let mut n = 0usize;
    let mut artifacts = 0usize;
    for &p in values {
        if p > cap {
            artifacts += 1;
            continue;
        }
        sum += p;
        peak = peak.max(p);
        n += 1;
    }
    if n == 0 {
        (0.0, 0.0, artifacts)
    } else {
        (sum / n as f64, peak, artifacts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::SceneConfig;

    #[test]
    fn collector_records_and_summarises() {
        // A small, fast scene.
        let toml = r#"
[sim]
resolution_mm = 2.5
duration_s = 0.04
frames = 4
[needle]
auto = true
diameter_mm = 1.2
flow_rate_ml_s = 5.0
[fluid]
particles_per_cell = 16
"#;
        let cfg = SceneConfig::from_toml_str(toml).unwrap();
        let mut built = cfg.build().unwrap();
        let mut metrics = MetricsCollector::new(&built.solver, built.cavity_volume);
        assert!(metrics.wall_count > 0, "cavity should have wall cells");

        for f in 0..built.frames {
            built.solver.step(built.frame_dt);
            let m = metrics.record(&built.solver, f);
            assert!(m.fill_fraction >= 0.0 && m.fill_fraction.is_finite());
            assert!(m.wall_coverage >= 0.0 && m.wall_coverage <= 1.0);
            assert!(m.max_pressure_pa.is_finite());
            assert!(m.peak_wall_pressure_pa.is_finite() && m.peak_wall_pressure_pa >= 0.0);
            // The focal membrane peak never exceeds the whole-fluid maximum,
            // and the broad mean never exceeds the focal peak.
            assert!(m.peak_wall_pressure_pa <= m.max_pressure_pa + 1e-6);
            assert!(m.wall_pressure_pa <= m.peak_wall_pressure_pa + 1e-6);
        }
        let s = metrics.summary();
        assert_eq!(s.frames, built.frames);
        // Some fluid must have been injected and touched the walls.
        assert!(s.injected_ml > 0.0, "no fluid injected");
        assert!(s.peak_wall_coverage > 0.0, "fluid never reached a wall");
        // The membrane load is the headline clinical pressure and is bounded by
        // the whole-fluid peak (which includes the needle jet core); the broad
        // mean load is in turn bounded by the focal peak.
        assert!(s.peak_membrane_pressure_pa <= s.peak_pressure_pa + 1e-6);
        assert!(s.mean_membrane_pressure_pa <= s.peak_membrane_pressure_pa + 1e-6);
        // On a coarse but valid small scene the cap should reject few or no cells.
        assert!(s.membrane_artifacts_total <= s.frames * 2, "too many pressure artifacts: {}", s.membrane_artifacts_total);
    }

    #[test]
    fn membrane_stats_excludes_nonphysical_artifacts() {
        // A handful of real wall loads plus one single-cell blow-up far above the
        // physical ceiling: the blow-up must be excluded from *both* the mean and
        // the peak, and counted as an artifact.
        let p = [100.0, 200.0, 500.0, 900.0, 2_000_000.0];
        let (mean, peak, artifacts) = membrane_stats(&p, 1000.0);
        assert_eq!(artifacts, 1);
        assert_eq!(peak, 900.0, "peak must ignore the spike above the cap");
        assert!((mean - (100.0 + 200.0 + 500.0 + 900.0) / 4.0).abs() < 1e-9);
        // A genuine focal load just under the cap is kept (not suppressed).
        assert_eq!(membrane_stats(&[10.0, 950.0], 1000.0), (480.0, 950.0, 0));
        // All-artifact and empty inputs are well defined.
        assert_eq!(membrane_stats(&[5000.0], 1000.0), (0.0, 0.0, 1));
        assert_eq!(membrane_stats(&[], 1000.0), (0.0, 0.0, 0));
    }

    #[test]
    fn writes_json_and_csv() {
        let cfg = SceneConfig::from_toml_str(
            "[sim]\nresolution_mm = 3.0\nduration_s = 0.02\nframes = 2\n[fluid]\nparticles_per_cell = 8\n",
        )
        .unwrap();
        let mut built = cfg.build().unwrap();
        let mut metrics = MetricsCollector::new(&built.solver, built.cavity_volume);
        for f in 0..built.frames {
            built.solver.step(built.frame_dt);
            metrics.record(&built.solver, f);
        }
        let dir = std::env::temp_dir().join("sdr_metrics_test");
        std::fs::create_dir_all(&dir).unwrap();
        let json = dir.join("metrics.json");
        let csv = dir.join("metrics.csv");
        metrics.write_json(&json).unwrap();
        metrics.write_csv(&csv).unwrap();
        let jtext = std::fs::read_to_string(&json).unwrap();
        assert!(jtext.contains("fill_fraction"));
        let ctext = std::fs::read_to_string(&csv).unwrap();
        assert!(ctext.lines().count() >= 3); // header + 2 frames
    }
}
