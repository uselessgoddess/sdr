//! The FLIP/PIC fluid solver.
//!
//! This is the engine described by the thesis: a hybrid **FLIP/PIC** method
//! (Zhu & Bridson, *Animating Sand as a Fluid*, SIGGRAPH 2005) on a staggered
//! MAC grid. Each step:
//!
//! 1. **emit** fresh fluid from the irrigation needle,
//! 2. **P2G** — splat particle velocities onto the grid faces,
//! 3. **forces** — add gravity,
//! 4. **project** — make the field divergence-free (see [`crate::pressure`]),
//! 5. **extrapolate** grid velocities into the air for stable interpolation,
//! 6. **G2P** — pull velocities back to particles (FLIP/PIC blend),
//! 7. **advect** — move particles through the grid velocity, keeping them
//!    inside the cavity and draining them at the outlet.
//!
//! Solids come from a signed-distance field (`φ < 0` inside the cavity), so the
//! same solver runs on the analytic parametric sinus or on a patient mesh.

use crate::grid::{trilinear_sample, trilinear_scatter, MacGrid};
use crate::math::{Aabb, Vec3};
use crate::particles::ParticleSet;
use crate::pressure::{self, Cell, SolveParams};
use crate::sdf::Sdf;

/// Per-step ceiling on the volume-control expansion, as a fraction of one cell
/// per step (`dx/dt`). Relieving an over-packed cell gradually — at most a
/// quarter of a cell of expansion per substep — keeps wall pressures physical
/// and the induced particle displacement comfortably inside the one-cell
/// advection clamp, so the fluid spreads to fill the cavity without the
/// projection ever shoving a particle through a wall.
const VC_MAX_FRACTION: f64 = 0.25;

/// Number of redistribution diffusion sweeps per sub-step (see
/// [`Solver::redistribute`]). A handful keeps the injection point from ever
/// piling up while staying local enough not to teleport fluid across the cavity.
/// The over-fill ceiling itself is the per-fluid [`FluidParams::redist_cap_factor`].
const REDIST_SWEEPS: usize = 4;

/// Physical properties of the irrigation fluid and the simulation.
#[derive(Debug, Clone, Copy)]
pub struct FluidParams {
    /// Density, kg/m³ (water ≈ 1000, saline slightly higher).
    pub density: f64,
    /// Body acceleration, m/s² (gravity, default `-9.81 ŷ`).
    pub gravity: Vec3,
    /// FLIP/PIC blend in `[0, 1]`. `1` = pure FLIP (lively, low dissipation),
    /// `0` = pure PIC (smooth, viscous). `0.95` is a good default.
    pub flip_ratio: f64,
    /// Target number of particles seeded per fluid cell when emitting.
    pub particles_per_cell: usize,
    /// Strength of the Zhu-Bridson volume control that relieves over-packed
    /// cells (see [`Solver::volume_source`]). `0` disables it (plain FLIP, which
    /// clumps); `1` is a good default. Has no effect on a cell at rest density.
    pub volume_stiffness: f64,
    /// Enable deterministic particle redistribution (see [`Solver::redistribute`]).
    /// Volume control biases the *grid* outward, but a motionless over-packed
    /// clump has no divergence for the projection to act on, so a point jet can
    /// still pile thousands of particles into one cell. Redistribution moves that
    /// surplus into adjacent open cavity cells directly, which is what lets the
    /// fluid spread and pool. `true` is the default for real-mesh scenes.
    pub redistribute: bool,
    /// Over-fill ceiling for redistribution, as a multiple of the seeding
    /// density (see [`Solver::redistribute`]). A cell sheds surplus only once it
    /// exceeds `redist_cap_factor × particles_per_cell`. Keep this near `1`: at
    /// rest density the settled pool is *at* the cap, so redistribution leaves it
    /// alone and gravity is free to shape a free surface. A larger value lets the
    /// fluid over-compress before relief; a much larger one was the original bug,
    /// where redistribution diffused the pool toward a *uniform* fill (lofting
    /// fluid back up against gravity) instead of letting it settle.
    pub redist_cap_factor: f64,
}

impl Default for FluidParams {
    fn default() -> Self {
        FluidParams {
            density: 1000.0,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            flip_ratio: 0.95,
            particles_per_cell: 8,
            volume_stiffness: 1.0,
            redistribute: true,
            redist_cap_factor: 1.0,
        }
    }
}

/// The irrigation needle: a thin jet entering the cavity.
#[derive(Debug, Clone, Copy)]
pub struct Needle {
    /// Tip position (where fluid is injected), metres.
    pub tip: Vec3,
    /// Unit jet direction.
    pub axis: Vec3,
    /// Inner radius of the cannula, metres.
    pub radius: f64,
    /// Volumetric flow rate, m³/s (1 ml/s = 1e-6 m³/s).
    pub flow_rate: f64,
    /// Stop injecting once the simulation clock reaches this time (seconds).
    /// Models a *fixed-volume* irrigation: the clinician pushes a set dose, then
    /// the fluid settles into a pool under gravity. `f64::INFINITY` injects for
    /// the whole run (the historical behaviour).
    pub inject_until: f64,
}

impl Needle {
    /// Mean jet speed from the flow rate and bore (continuity).
    pub fn jet_speed(&self) -> f64 {
        let area = std::f64::consts::PI * self.radius * self.radius;
        if area > 0.0 {
            self.flow_rate / area
        } else {
            0.0
        }
    }
}

/// Deterministic xorshift RNG so simulations are byte-for-byte reproducible
/// (essential for CI assertions).
#[derive(Debug, Clone)]
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// Uniform in `[0, 1)`.
    #[inline]
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }
}

/// The fluid solver state.
pub struct Solver {
    pub grid: MacGrid,
    pub particles: ParticleSet,
    /// Cavity SDF: `φ < 0` inside the cavity (where fluid lives), `φ ≥ 0` in the
    /// surrounding bone/solid.
    pub solid: Sdf,
    pub fluid: FluidParams,
    pub needle: Needle,
    /// Optional drainage region (e.g. the socket opening). Cells here act as a
    /// zero-pressure outlet and particles entering it are removed.
    pub outlet: Option<Aabb>,
    pub solve: SolveParams,
    pub time: f64,
    pub steps: u64,
    /// Iterations taken by the last pressure solve (diagnostic).
    pub last_pcg_iters: usize,
    /// Sub-steps taken by the last [`Solver::step`] (diagnostic). Climbs toward
    /// the CFL count and saturates at the per-frame cap when a velocity spike
    /// would otherwise collapse the timestep.
    pub last_substeps: u32,
    /// Per-cell pressure (Pa, gauge) from the last projection; `0` off-fluid.
    pub pressure: Vec<f64>,
    /// Cumulative number of particles removed at the outlet (drainage).
    pub drained: u64,

    cells: Vec<Cell>,
    /// Static mask: `true` where a cell centre lies inside the cavity (`φ < 0`).
    /// The solid never moves, so this is computed once and reused by both
    /// [`Solver::classify`] and [`Solver::redistribute`] instead of re-sampling
    /// the SDF for every cell on every substep.
    cavity_mask: Vec<bool>,
    // FLIP needs the post-P2G grid to form the velocity delta.
    saved_u: Vec<f64>,
    saved_v: Vec<f64>,
    saved_w: Vec<f64>,
    rng: Rng,
    emit_accum: f64,
}

impl Solver {
    /// Build a solver whose grid covers the SDF's bounds at spacing `sim_dx`.
    pub fn new(solid: Sdf, sim_dx: f64, fluid: FluidParams, needle: Needle) -> Self {
        let bb = solid.bounds();
        let size = bb.size();
        let nx = (size.x / sim_dx).ceil().max(1.0) as usize;
        let ny = (size.y / sim_dx).ceil().max(1.0) as usize;
        let nz = (size.z / sim_dx).ceil().max(1.0) as usize;
        let grid = MacGrid::new(nx, ny, nz, sim_dx, bb.min);
        let n = nx * ny * nz;
        // Precompute the static cavity mask once: the solid SDF never changes,
        // so there is no reason to re-sample it for every cell on every substep.
        let cavity_mask: Vec<bool> = (0..n)
            .map(|c| {
                let (i, j, k) = (c % nx, (c / nx) % ny, c / (nx * ny));
                solid.sample(grid.cell_center(i, j, k)) < 0.0
            })
            .collect();
        Solver {
            grid,
            particles: ParticleSet::new(),
            solid,
            fluid,
            needle,
            outlet: None,
            solve: SolveParams::default(),
            time: 0.0,
            steps: 0,
            last_pcg_iters: 0,
            last_substeps: 0,
            pressure: vec![0.0; n],
            drained: 0,
            cells: vec![Cell::Air; n],
            cavity_mask,
            saved_u: Vec::new(),
            saved_v: Vec::new(),
            saved_w: Vec::new(),
            rng: Rng::new(0x5deece66d),
            emit_accum: 0.0,
        }
    }

    /// Volume represented by one particle (cell volume / seeding density).
    pub fn particle_volume(&self) -> f64 {
        let cell = self.grid.dx.powi(3);
        cell / self.fluid.particles_per_cell.max(1) as f64
    }

    /// Total fluid volume currently in the cavity, m³.
    pub fn fluid_volume(&self) -> f64 {
        self.particles.len() as f64 * self.particle_volume()
    }

    /// Advance the simulation by `dt` seconds, internally sub-stepping so the
    /// CFL condition (particles move < 1 cell/step) always holds.
    pub fn step(&mut self, dt: f64) {
        // Bound the sub-step count per frame. `cfl_dt` shrinks the timestep from
        // the *maximum* grid speed, so a single pathological cell — e.g. a
        // thin-septum SDF artefact where volume control and the wall projection
        // briefly fight — can drive `max_speed` to numerical garbage and collapse
        // the global timestep, stalling one frame in tens of thousands of
        // sub-steps. Advection already clamps displacement to one cell
        // (`advect`), so the floor timestep stays stable; the floor only adds a
        // little numerical damping, which a settling pool tolerates well. The cap
        // sits above the legitimate jet/gravity CFL count, so well-behaved frames
        // are unaffected and only the spike is clipped.
        const MAX_SUBSTEPS: u32 = 512;
        let floor = dt / MAX_SUBSTEPS as f64;
        let mut remaining = dt;
        let mut taken = 0;
        while remaining > 1e-12 {
            let sub = self.cfl_dt().max(floor).min(remaining);
            self.substep(sub);
            remaining -= sub;
            taken += 1;
        }
        self.last_substeps = taken;
    }

    /// A stable sub-step from the current maximum speed.
    fn cfl_dt(&self) -> f64 {
        let grid_speed = self.grid.max_speed();
        // The jet only bounds the timestep while it is actually firing. It must
        // be included *then* because on the very first emit the fast inflow is
        // not yet on the grid (`grid_speed` would miss it). Once the fixed dose
        // is delivered (`time >= inject_until`, matching `emit`) the needle adds
        // nothing, and the still-moving jet fluid is already captured by
        // `grid_speed` — so dropping the constant jet term here lets the
        // sub-step count fall away as the pool comes to rest instead of pinning
        // at the jet-CFL count for the whole settle phase.
        let jet = if self.time < self.needle.inject_until {
            self.needle.jet_speed()
        } else {
            0.0
        };
        let g = (self.fluid.gravity.length() * self.grid.dx).sqrt();
        let speed = grid_speed.max(jet).max(g).max(1e-6);
        // Allow ~1 cell of travel; cap so an empty sim still progresses.
        (self.grid.dx / speed).min(0.01)
    }

    /// One full FLIP/PIC sub-step.
    pub fn substep(&mut self, dt: f64) {
        self.emit(dt);
        self.classify();
        self.p2g();
        self.save_grid();
        self.add_forces(dt);
        let source = self.volume_source(dt);
        let (report, pressure) = pressure::project_capturing_with_source(
            &mut self.grid,
            &self.cells,
            self.fluid.density,
            dt,
            self.solve,
            Some(&source),
        );
        self.last_pcg_iters = report.iters;
        self.pressure = pressure;
        self.extrapolate_velocities();
        self.g2p();
        self.advect(dt);
        self.enforce_particle_bounds();
        if self.fluid.redistribute {
            self.redistribute();
        }
        self.time += dt;
        self.steps += 1;
    }

    /// Inject particles from the needle to match the volumetric flow rate.
    fn emit(&mut self, dt: f64) {
        let pv = self.particle_volume();
        if pv <= 0.0 || self.needle.flow_rate <= 0.0 || self.time >= self.needle.inject_until {
            return;
        }
        self.emit_accum += self.needle.flow_rate * dt / pv;
        let count = self.emit_accum.floor();
        let n = count as usize;
        self.emit_accum -= count;
        if n == 0 {
            return;
        }
        let axis = self.needle.axis.normalize_or_zero();
        let (e1, e2) = orthonormal_basis(axis);
        let speed = self.needle.jet_speed();
        let vel = axis * speed;
        for _ in 0..n {
            // Uniform sample in the bore disk, plus a little axial jitter so
            // particles don't all land on one plane.
            let r = self.needle.radius * self.rng.unit().sqrt();
            let theta = 2.0 * std::f64::consts::PI * self.rng.unit();
            let along = (self.rng.unit() - 0.5) * self.grid.dx;
            let pos =
                self.needle.tip + e1 * (r * theta.cos()) + e2 * (r * theta.sin()) + axis * along;
            self.particles.push(pos, vel);
        }
    }

    /// Classify every cell as solid / fluid / air for this step.
    fn classify(&mut self) {
        let (nx, ny, nz) = (self.grid.nx, self.grid.ny, self.grid.nz);
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let c = (k * ny + j) * nx + i;
                    let outlet = self
                        .outlet
                        .map(|o| o.contains(self.grid.cell_center(i, j, k)))
                        .unwrap_or(false);
                    self.cells[c] = if outlet {
                        Cell::Air // zero-pressure drain
                    } else if !self.cavity_mask[c] {
                        Cell::Solid
                    } else {
                        Cell::Air
                    };
                }
            }
        }
        // Mark cells holding fluid particles (unless they are solid/outlet).
        for &p in &self.particles.positions {
            if let Some((i, j, k)) = self.cell_of(p) {
                let c = (k * ny + j) * nx + i;
                if self.cells[c] == Cell::Air && !self.in_outlet(p) {
                    self.cells[c] = Cell::Fluid;
                }
            }
        }
    }

    /// Per-cell **volume-control source** for the pressure solve (the technique
    /// of Zhu & Bridson 2005, §5). Plain FLIP/PIC keeps the *grid* velocity
    /// divergence-free but is blind to how many particles share a cell, so a
    /// fast jet can pack hundreds of particles into one cell and the projection
    /// never pushes them apart — the fluid clumps into a streak instead of
    /// filling the cavity. Here we measure each fluid cell's particle density,
    /// and return a positive (outward) target divergence for over-full cells so
    /// the solve raises their pressure and spreads the fluid into open space.
    ///
    /// The expansion is relieved *gradually*: the source ramps with the cell's
    /// over-fill (scaled by `volume_stiffness`) but is capped at
    /// [`VC_MAX_FRACTION`] of one cell-per-step (`dx/dt`). A gentle, persistent
    /// outward bias spread over several substeps keeps wall pressures physical
    /// and the resulting particle displacement well inside the one-cell
    /// advection clamp, so the projection never shoves a particle through a wall
    /// (which, in a closed cavity, would cull it and lose mass). Counting and
    /// the index-ordered loop are deterministic, preserving the crate's
    /// byte-for-byte reproducibility guarantee. Returns all-zeros when
    /// `volume_stiffness == 0` (recovering plain FLIP).
    fn volume_source(&self, dt: f64) -> Vec<f64> {
        let (nx, ny, nz) = (self.grid.nx, self.grid.ny, self.grid.nz);
        let n = nx * ny * nz;
        let k_vol = self.fluid.volume_stiffness;
        if k_vol <= 0.0 || dt <= 0.0 {
            return vec![0.0; n];
        }
        // Particles per cell (only where the cell is classified fluid).
        let mut count = vec![0u32; n];
        for &p in &self.particles.positions {
            if let Some((i, j, k)) = self.cell_of(p) {
                let c = (k * ny + j) * nx + i;
                if self.cells[c] == Cell::Fluid {
                    count[c] += 1;
                }
            }
        }
        let rest = self.fluid.particles_per_cell.max(1) as f64;
        let unit = self.grid.dx / dt; // one cell of expansion per step
        let cap = VC_MAX_FRACTION * unit; // gentle per-step ceiling
        let mut source = vec![0.0f64; n];
        for c in 0..n {
            if self.cells[c] != Cell::Fluid {
                continue;
            }
            let over = count[c] as f64 / rest - 1.0;
            if over > 0.0 {
                source[c] = (k_vol * over * unit).min(cap);
            }
        }
        source
    }

    /// Relieve FLIP particle clumping by **redistributing** the surplus of
    /// over-packed cells into adjacent open cavity cells.
    ///
    /// Volume control ([`Solver::volume_source`]) biases the *grid* outward, but
    /// the pressure projection only ever acts on velocity *divergence* — and a
    /// clump of particles at rest against a wall has none. A point jet firing
    /// into a closed cavity therefore packs thousands of particles into the few
    /// cells around the impact point, and no amount of projection or FLIP/PIC
    /// tuning spreads them (verified: pure PIC clumps just as badly). This is the
    /// classic limitation of marker-particle methods, and the classic cure is to
    /// act on the particles directly.
    ///
    /// Each sweep histograms particles per cell, then — visiting particles in
    /// index order — drains any cell holding more than `REDIST_CAP_FACTOR ×` the
    /// seeding density, moving its surplus into the least-occupied face-adjacent
    /// neighbour that is inside the cavity and below the cap. A moved particle is
    /// repositioned near the destination cell's centre (with a deterministic
    /// sub-cell jitter so particles don't stack on the centre) and takes the
    /// local projected grid velocity, so it rejoins the flow without injecting
    /// spurious momentum. Because particles are only ever *moved* (never created
    /// or destroyed) fluid mass is conserved exactly, and because every choice
    /// reads index-ordered data with hash-based jitter the result stays
    /// byte-for-byte reproducible.
    ///
    /// Surplus only ever flows "downhill" in occupancy, so this is pure density
    /// diffusion: it strictly lowers the peak per-cell count and cannot
    /// oscillate. Gravity (added every step on the grid) supplies the downward
    /// drift, so the two together settle the fluid into a pool with a free
    /// surface instead of a stalled streak.
    fn redistribute(&mut self) {
        let (nx, ny, nz) = (self.grid.nx, self.grid.ny, self.grid.nz);
        let ncells = nx * ny * nz;
        let target = self.fluid.particles_per_cell.max(1);
        let cap = (self.fluid.redist_cap_factor.max(1.0) * target as f64).ceil() as u32;
        // Unit "down" axis (gravity direction). Surplus from an over-full cell
        // prefers to fall into the lowest neighbour that still has room, so the
        // fluid seeks the lowest open space and stacks upward — a pool with a
        // rising free surface — instead of diffusing isotropically toward a
        // uniform fill (which lofts fluid back up against gravity).
        let gdir = self.fluid.gravity.normalize_or_zero();

        // Valid destinations are cells whose centre lies inside the cavity; this
        // is the precomputed `self.cavity_mask` (the solid never moves).
        let mut count = vec![0u32; ncells];
        for _ in 0..REDIST_SWEEPS {
            count.iter_mut().for_each(|c| *c = 0);
            for &p in &self.particles.positions {
                if let Some((i, j, k)) = self.cell_of(p) {
                    count[(k * ny + j) * nx + i] += 1;
                }
            }
            for idx in 0..self.particles.len() {
                let c = match self.cell_of(self.particles.positions[idx]) {
                    Some((i, j, k)) => (k * ny + j) * nx + i,
                    None => continue,
                };
                if count[c] <= cap {
                    continue;
                }
                let (i, j, k) = (c % nx, (c / nx) % ny, c / (nx * ny));
                // Gather the in-bounds face neighbours together with their unit
                // direction from this cell, so we can rank them by how far "down"
                // (along gravity) they lie.
                let mut neigh = [(0usize, 0.0f64); 6];
                let mut nn = 0;
                let mut add = |nc: usize, dir: Vec3| {
                    neigh[nn] = (nc, dir.dot(gdir));
                    nn += 1;
                };
                if i > 0 {
                    add(c - 1, Vec3::new(-1.0, 0.0, 0.0));
                }
                if i + 1 < nx {
                    add(c + 1, Vec3::new(1.0, 0.0, 0.0));
                }
                if j > 0 {
                    add(c - nx, Vec3::new(0.0, -1.0, 0.0));
                }
                if j + 1 < ny {
                    add(c + nx, Vec3::new(0.0, 1.0, 0.0));
                }
                if k > 0 {
                    add(c - nx * ny, Vec3::new(0.0, 0.0, -1.0));
                }
                if k + 1 < nz {
                    add(c + nx * ny, Vec3::new(0.0, 0.0, 1.0));
                }
                // Send the surplus to the **lowest in-cavity neighbour that still
                // has room** (below the cap). Ranking by gravity (most-downhill
                // first), then by emptiness, makes the fluid seek the lowest open
                // space and stack upward — a settling pool with a rising free
                // surface. Crucially the receiver must be *below the cap*: an
                // unconditional "least-occupied neighbour" test instead diffuses
                // occupancy toward a uniform fill, which lofts fluid back up
                // against gravity (the pool never forms). Restricting receivers to
                // those with room means a pool already at rest density is left
                // untouched, so gravity alone shapes it; only genuine over-packing
                // (the point-jet clump) is relieved, draining cell-by-cell into the
                // open space below. Ranking over a fixed neighbour-visit order with
                // plain comparisons keeps the choice deterministic.
                let mut best: Option<usize> = None;
                let mut best_down = f64::NEG_INFINITY;
                let mut best_room = 0u32; // larger = emptier
                for &(nc, down) in &neigh[..nn] {
                    if !self.cavity_mask[nc] || count[nc] >= cap {
                        continue;
                    }
                    let room = cap - count[nc];
                    // Most-downhill first; ties broken toward the emptier cell, and
                    // finally toward the first neighbour visited (fixed index order)
                    // — every comparison is over a fixed deterministic ordering.
                    let better = best.is_none()
                        || down > best_down
                        || (down == best_down && room > best_room);
                    if better {
                        best = Some(nc);
                        best_down = down;
                        best_room = room;
                    }
                }
                if let Some(nc) = best {
                    count[c] -= 1;
                    count[nc] += 1;
                    let (ci, cj, ck) = (nc % nx, (nc / nx) % ny, nc / (nx * ny));
                    let jitter = hash_jitter(idx as u64);
                    let pos = self.grid.cell_center(ci, cj, ck) + jitter * self.grid.dx;
                    let pos = self.project_inside(pos);
                    self.particles.positions[idx] = pos;
                    self.particles.velocities[idx] = self.grid.velocity_at(pos);
                }
            }
        }
    }

    /// Splat particle velocities onto the staggered faces (mass-weighted).
    fn p2g(&mut self) {
        self.grid.zero();
        let ud = self.grid.u_dims();
        let vd = self.grid.v_dims();
        let wd = self.grid.w_dims();
        let mut wu = vec![0.0f64; self.grid.u.len()];
        let mut wv = vec![0.0f64; self.grid.v.len()];
        let mut ww = vec![0.0f64; self.grid.w.len()];

        for (p, vel) in self
            .particles
            .positions
            .iter()
            .zip(&self.particles.velocities)
        {
            let (gx, gy, gz) = self.grid.u_coords(*p);
            trilinear_scatter(&mut self.grid.u, &mut wu, ud, gx, gy, gz, vel.x);
            let (gx, gy, gz) = self.grid.v_coords(*p);
            trilinear_scatter(&mut self.grid.v, &mut wv, vd, gx, gy, gz, vel.y);
            let (gx, gy, gz) = self.grid.w_coords(*p);
            trilinear_scatter(&mut self.grid.w, &mut ww, wd, gx, gy, gz, vel.z);
        }
        normalize(&mut self.grid.u, &wu);
        normalize(&mut self.grid.v, &wv);
        normalize(&mut self.grid.w, &ww);
    }

    fn save_grid(&mut self) {
        self.saved_u.clone_from(&self.grid.u);
        self.saved_v.clone_from(&self.grid.v);
        self.saved_w.clone_from(&self.grid.w);
    }

    fn add_forces(&mut self, dt: f64) {
        let g = self.fluid.gravity;
        if g.x != 0.0 {
            self.grid.u.iter_mut().for_each(|u| *u += g.x * dt);
        }
        if g.y != 0.0 {
            self.grid.v.iter_mut().for_each(|v| *v += g.y * dt);
        }
        if g.z != 0.0 {
            self.grid.w.iter_mut().for_each(|w| *w += g.z * dt);
        }
    }

    /// Extrapolate face velocities from fluid into neighbouring air faces so
    /// that interpolation for particles near the free surface stays sensible.
    fn extrapolate_velocities(&mut self) {
        let (nx, ny, nz) = (self.grid.nx, self.grid.ny, self.grid.nz);
        let fluid = |i: usize, j: usize, k: usize| self.cells[(k * ny + j) * nx + i] == Cell::Fluid;

        // Validity = face borders a fluid cell.
        let mut valid_u = vec![false; self.grid.u.len()];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..=nx {
                    let l = i > 0 && fluid(i - 1, j, k);
                    let r = i < nx && fluid(i, j, k);
                    valid_u[self.grid.u_idx(i, j, k)] = l || r;
                }
            }
        }
        let mut valid_v = vec![false; self.grid.v.len()];
        for k in 0..nz {
            for j in 0..=ny {
                for i in 0..nx {
                    let d = j > 0 && fluid(i, j - 1, k);
                    let u = j < ny && fluid(i, j, k);
                    valid_v[self.grid.v_idx(i, j, k)] = d || u;
                }
            }
        }
        let mut valid_w = vec![false; self.grid.w.len()];
        for k in 0..=nz {
            for j in 0..ny {
                for i in 0..nx {
                    let b = k > 0 && fluid(i, j, k - 1);
                    let f = k < nz && fluid(i, j, k);
                    valid_w[self.grid.w_idx(i, j, k)] = b || f;
                }
            }
        }
        let (ud, vd, wd) = (self.grid.u_dims(), self.grid.v_dims(), self.grid.w_dims());
        extrapolate(&mut self.grid.u, &valid_u, ud, 4);
        extrapolate(&mut self.grid.v, &valid_v, vd, 4);
        extrapolate(&mut self.grid.w, &valid_w, wd, 4);
    }

    /// Pull grid velocities back to particles, blending FLIP and PIC.
    fn g2p(&mut self) {
        let flip = self.fluid.flip_ratio.clamp(0.0, 1.0);
        let du = diff(&self.grid.u, &self.saved_u);
        let dv = diff(&self.grid.v, &self.saved_v);
        let dw = diff(&self.grid.w, &self.saved_w);
        let ud = self.grid.u_dims();
        let vd = self.grid.v_dims();
        let wd = self.grid.w_dims();

        for (p, vel) in self
            .particles
            .positions
            .iter()
            .zip(self.particles.velocities.iter_mut())
        {
            // PIC: fresh interpolation of the new field.
            let pic = self.grid.velocity_at(*p);
            // FLIP: old particle velocity plus the interpolated change.
            let (gx, gy, gz) = self.grid.u_coords(*p);
            let dvx = trilinear_sample(&du, ud, gx, gy, gz);
            let (gx, gy, gz) = self.grid.v_coords(*p);
            let dvy = trilinear_sample(&dv, vd, gx, gy, gz);
            let (gx, gy, gz) = self.grid.w_coords(*p);
            let dvz = trilinear_sample(&dw, wd, gx, gy, gz);
            let flip_v = *vel + Vec3::new(dvx, dvy, dvz);
            *vel = flip_v * flip + pic * (1.0 - flip);
        }
    }

    /// Move particles through the grid velocity (RK2 midpoint) and keep them
    /// inside the cavity.
    ///
    /// Two safeguards make the closed cavity leak-proof — important now that
    /// volume control actively pushes fluid against the walls:
    ///
    /// * the per-substep displacement is clamped to one cell, so a velocity
    ///   spike (the projection runs *after* the CFL `dt` was chosen, so it can
    ///   briefly outrun it) can never shoot a particle clean through a wall;
    /// * if the wall projection still can't return a particle to the interior
    ///   (e.g. a degenerate SDF gradient in a thin septum), the particle stays
    ///   at its last valid interior position rather than escaping and being
    ///   culled. With no outlet every culled particle is a tunnelling artefact,
    ///   so this keeps mass conserved.
    fn advect(&mut self, dt: f64) {
        let n = self.particles.len();
        let max_step = self.grid.dx;
        for idx in 0..n {
            let p0 = self.particles.positions[idx];
            let v0 = self.grid.velocity_at(p0);
            let mid = p0 + v0 * (0.5 * dt);
            let vmid = self.grid.velocity_at(mid);
            let mut step = vmid * dt;
            let len = step.length();
            if len > max_step {
                step = step * (max_step / len);
            }
            let p = self.project_inside(p0 + step);
            // Refuse to leave the cavity: if projection couldn't recover an
            // interior point, hold the previous (valid) position.
            self.particles.positions[idx] = if self.solid.sample(p) <= 0.0 { p } else { p0 };
        }
    }

    /// Push a point back inside the cavity if it has entered the wall, using the
    /// SDF gradient. Keeps a half-cell margin off the wall. Iterates a few times
    /// because one Newton step under-corrects where the SDF is not a perfect
    /// distance field (its gradient is only approximately unit length).
    fn project_inside(&self, p: Vec3) -> Vec3 {
        let margin = 0.5 * self.grid.dx;
        let mut p = p;
        for _ in 0..3 {
            let phi = self.solid.sample(p);
            if phi <= -margin {
                break;
            }
            let g = self.solid.gradient(p).normalize_or_zero();
            if g.length() == 0.0 {
                break;
            }
            p -= g * (phi + margin);
        }
        p
    }

    /// Remove particles that drained out, escaped the domain, or got stuck deep
    /// in a wall.
    fn enforce_particle_bounds(&mut self) {
        let bb = self.solid.bounds();
        let outlet = self.outlet;
        let solid = &self.solid;
        let mut drained = 0u64;
        self.particles.retain(|p, _| {
            if !bb.contains(p) {
                return false;
            }
            if let Some(o) = outlet {
                if o.contains(p) {
                    drained += 1; // left through the outlet
                    return false;
                }
            }
            // Deep inside the wall and unrecoverable -> drop.
            solid.sample(p) < 0.5 * solid.dx
        });
        self.drained += drained;
    }

    /// Volume that has drained out of the outlet so far, m³.
    pub fn drained_volume(&self) -> f64 {
        self.drained as f64 * self.particle_volume()
    }

    /// Number of cells currently classified as fluid (diagnostic).
    pub fn cells_fluid_count(&self) -> usize {
        self.cells.iter().filter(|c| **c == Cell::Fluid).count()
    }

    /// Largest number of particles sharing a single grid cell (diagnostic for
    /// FLIP clumping; with volume control on it should stay near
    /// `particles_per_cell`).
    pub fn max_particles_per_cell(&self) -> usize {
        let (nx, ny, nz) = (self.grid.nx, self.grid.ny, self.grid.nz);
        let mut count = vec![0u32; nx * ny * nz];
        let mut max = 0u32;
        for &p in &self.particles.positions {
            if let Some((i, j, k)) = self.cell_of(p) {
                let c = (k * ny + j) * nx + i;
                count[c] += 1;
                max = max.max(count[c]);
            }
        }
        max as usize
    }

    /// Depth profile of the cavity itself (diagnostic): how many grid cells have
    /// their centre at each signed-distance band. If the cavity is a thin shell
    /// every interior cell is within ~1 cell of a wall, so `deep` stays ~0 and no
    /// physics can produce a thick fill — the geometry, not the solver, is the
    /// limit. Run once at start-up.
    pub fn debug_cavity_geometry(&self) -> String {
        let (nx, ny, nz) = (self.grid.nx, self.grid.ny, self.grid.nz);
        let dx = self.grid.dx;
        let (mut inside, mut deep2, mut deep1, mut mid, mut near) = (0u64, 0u64, 0u64, 0u64, 0u64);
        let mut min_phi = 0.0f64;
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let phi = self.solid.sample(self.grid.cell_center(i, j, k)) / dx;
                    if phi >= 0.0 {
                        continue;
                    }
                    inside += 1;
                    min_phi = min_phi.min(phi);
                    if phi < -2.0 {
                        deep2 += 1;
                    } else if phi < -1.5 {
                        deep1 += 1;
                    } else if phi < -1.0 {
                        mid += 1;
                    } else {
                        near += 1;
                    }
                }
            }
        }
        format!(
            "cavity cells={inside} | phi/dx: <-2={deep2} [-2,-1.5)={deep1} [-1.5,-1)={mid} [-1,0)={near} | deepest={min_phi:.2} dx"
        )
    }

    /// Where the particles actually sit, by the classification of the cell that
    /// holds them and by signed-distance band (diagnostic). A large share of
    /// particles in `Solid`-classified cells is the pathology where the jet rams
    /// fluid against a wall into cells whose *centre* is in bone: such cells are
    /// excluded from the pressure solve and the volume-control source, so the
    /// fluid there is invisible to the physics and piles up unrelieved.
    pub fn debug_distribution(&self) -> String {
        let (nx, ny, _nz) = (self.grid.nx, self.grid.ny, self.grid.nz);
        let (mut in_solid, mut in_fluid, mut in_air, mut outside) = (0u64, 0u64, 0u64, 0u64);
        let dx = self.solid.dx;
        // phi bands, in units of dx: deep (<−1), mid [−1,−0.5), near [−0.5,0), wall [0,∞)
        let (mut deep, mut mid, mut near, mut wall) = (0u64, 0u64, 0u64, 0u64);
        for &p in &self.particles.positions {
            match self.cell_of(p) {
                Some((i, j, k)) => match self.cells[(k * ny + j) * nx + i] {
                    Cell::Solid => in_solid += 1,
                    Cell::Fluid => in_fluid += 1,
                    Cell::Air => in_air += 1,
                },
                None => outside += 1,
            }
            let phi = self.solid.sample(p) / dx;
            if phi < -1.0 {
                deep += 1;
            } else if phi < -0.5 {
                mid += 1;
            } else if phi < 0.0 {
                near += 1;
            } else {
                wall += 1;
            }
        }
        format!(
            "        cell-class: solid={in_solid} fluid={in_fluid} air={in_air} outside={outside} | \
             phi/dx: deep<-1={deep} [-1,-.5)={mid} [-.5,0)={near} >=0={wall}"
        )
    }

    /// Peak fluid gauge pressure in the cavity, pascals.
    pub fn max_pressure(&self) -> f64 {
        self.cells
            .iter()
            .zip(&self.pressure)
            .filter(|(c, _)| **c == Cell::Fluid)
            .map(|(_, &p)| p)
            .fold(0.0_f64, f64::max)
    }

    /// Mean fluid gauge pressure over fluid cells, pascals.
    pub fn mean_pressure(&self) -> f64 {
        let (sum, n) = self
            .cells
            .iter()
            .zip(&self.pressure)
            .filter(|(c, _)| **c == Cell::Fluid)
            .fold((0.0_f64, 0u64), |(s, n), (_, &p)| (s + p, n + 1));
        if n > 0 {
            sum / n as f64
        } else {
            0.0
        }
    }

    /// Maximum particle speed, m/s.
    pub fn max_speed(&self) -> f64 {
        self.particles
            .velocities
            .iter()
            .map(|v| v.length())
            .fold(0.0_f64, f64::max)
    }

    fn in_outlet(&self, p: Vec3) -> bool {
        self.outlet.map(|o| o.contains(p)).unwrap_or(false)
    }

    /// Integer cell containing world point `p`, or `None` if outside the grid.
    fn cell_of(&self, p: Vec3) -> Option<(usize, usize, usize)> {
        let l = (p - self.grid.origin) / self.grid.dx;
        if l.x < 0.0 || l.y < 0.0 || l.z < 0.0 {
            return None;
        }
        let (i, j, k) = (l.x as usize, l.y as usize, l.z as usize);
        if i < self.grid.nx && j < self.grid.ny && k < self.grid.nz {
            Some((i, j, k))
        } else {
            None
        }
    }
}

/// Deterministic sub-cell jitter in `[-0.3, 0.3]³` from a particle key, so
/// redistributed particles spread within the destination cell instead of
/// stacking on its centre. Uses a splitmix64 finaliser (no global RNG state),
/// so it never perturbs the emission stream and stays byte-for-byte
/// reproducible.
fn hash_jitter(key: u64) -> Vec3 {
    let mix = |k: u64| -> f64 {
        let mut z = k.wrapping_add(0x9E3779B97F4A7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        // Map the top 53 bits to [-0.3, 0.3).
        ((z >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.6
    };
    Vec3::new(
        mix(key.wrapping_mul(3)),
        mix(key.wrapping_mul(3).wrapping_add(1)),
        mix(key.wrapping_mul(3).wrapping_add(2)),
    )
}

/// Build two unit vectors orthogonal to `n` (and to each other).
fn orthonormal_basis(n: Vec3) -> (Vec3, Vec3) {
    let a = if n.x.abs() < 0.9 {
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    };
    let e1 = n.cross(a).normalize_or_zero();
    let e2 = n.cross(e1).normalize_or_zero();
    (e1, e2)
}

fn normalize(field: &mut [f64], weights: &[f64]) {
    for (f, &w) in field.iter_mut().zip(weights) {
        if w > 1e-12 {
            *f /= w;
        } else {
            *f = 0.0;
        }
    }
}

fn diff(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

/// Flood-fill extrapolation: repeatedly set each invalid cell to the average of
/// its already-valid neighbours.
fn extrapolate(field: &mut [f64], valid: &[bool], dims: (usize, usize, usize), iters: usize) {
    let (nx, ny, nz) = dims;
    let idx = |i: usize, j: usize, k: usize| (k * ny + j) * nx + i;
    let mut known = valid.to_vec();
    for _ in 0..iters {
        let snap = field.to_vec();
        let ksnap = known.clone();
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let c = idx(i, j, k);
                    if ksnap[c] {
                        continue;
                    }
                    let mut sum = 0.0;
                    let mut cnt = 0;
                    let mut acc = |ci: usize, cj: usize, ck: usize| {
                        let nc = idx(ci, cj, ck);
                        if ksnap[nc] {
                            sum += snap[nc];
                            cnt += 1;
                        }
                    };
                    if i > 0 {
                        acc(i - 1, j, k);
                    }
                    if i + 1 < nx {
                        acc(i + 1, j, k);
                    }
                    if j > 0 {
                        acc(i, j - 1, k);
                    }
                    if j + 1 < ny {
                        acc(i, j + 1, k);
                    }
                    if k > 0 {
                        acc(i, j, k - 1);
                    }
                    if k + 1 < nz {
                        acc(i, j, k + 1);
                    }
                    if cnt > 0 {
                        field[c] = sum / cnt as f64;
                        known[c] = true;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sinus::SinusParams;

    fn small_solver() -> Solver {
        let params = SinusParams::default();
        let mesh = params.generate();
        // A coarse SDF / sim grid keeps the test fast; high seeding density
        // gives plenty of particles without driving up the sub-step count.
        let sdf = Sdf::from_mesh(&mesh, 0.0025, 0.006);
        let (tip, axis) = params.suggested_needle();
        let needle = Needle {
            tip,
            axis,
            radius: 0.0006,
            flow_rate: 5.0e-6,
            inject_until: f64::INFINITY,
        };
        let fluid = FluidParams {
            particles_per_cell: 16,
            ..FluidParams::default()
        };
        let mut s = Solver::new(sdf, 0.0025, fluid, needle);
        // Drain at the very bottom of the socket.
        let (sx, sz) = params.socket_xz;
        let yb = params.socket_bottom_y();
        s.outlet = Some(Aabb::new(
            Vec3::new(sx - 0.004, yb - 0.002, sz - 0.004),
            Vec3::new(sx + 0.004, yb + 0.002, sz + 0.004),
        ));
        s
    }

    #[test]
    fn jet_speed_from_flow_rate() {
        let n = Needle {
            tip: Vec3::ZERO,
            axis: Vec3::new(0.0, 1.0, 0.0),
            radius: 0.001,
            flow_rate: 1e-6,
            inject_until: f64::INFINITY,
        };
        // 1 ml/s through r=1mm: v = Q/(pi r^2) = 1e-6 / (pi*1e-6) = 1/pi.
        assert!((n.jet_speed() - 1.0 / std::f64::consts::PI).abs() < 1e-9);
    }

    #[test]
    fn emits_and_fills_over_time() {
        let mut s = small_solver();
        assert_eq!(s.particles.len(), 0);
        // Simulate a short burst of irrigation.
        for _ in 0..8 {
            s.step(0.005);
        }
        assert!(!s.particles.is_empty(), "needle should have injected fluid");
        // Fluid volume should be positive and finite.
        let vol = s.fluid_volume();
        assert!(vol > 0.0 && vol.is_finite(), "fluid volume {vol}");
        // Particles must stay finite and (mostly) inside the padded cavity.
        assert!(s.particles.positions.iter().all(|p| p.is_finite()));
    }

    #[test]
    fn injection_stops_at_inject_until() {
        // A closed box has no outlet, so the particle count can only ever change
        // by emission — the clean way to prove the irrigation cutoff fires.
        let mut s = box_solver();
        s.needle.inject_until = 0.02;
        // Step well past the cutoff (0.05 s > 0.02 s).
        for _ in 0..10 {
            s.step(0.005);
        }
        let after_cutoff = s.particles.len();
        assert!(
            after_cutoff > 0,
            "fluid should be injected before the cutoff"
        );
        assert!(s.time > s.needle.inject_until);
        // Once the clock is past inject_until, no further fluid may appear.
        for _ in 0..10 {
            s.step(0.005);
        }
        assert_eq!(
            s.particles.len(),
            after_cutoff,
            "no fluid may be injected after inject_until"
        );
    }

    #[test]
    fn particles_stay_out_of_walls() {
        let mut s = small_solver();
        for _ in 0..12 {
            s.step(0.005);
        }
        // After settling, no particle should be deep inside the bone wall.
        let max_phi = s
            .particles
            .positions
            .iter()
            .map(|&p| s.solid.sample(p))
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            max_phi < s.solid.dx,
            "a particle sits inside the wall (phi = {max_phi})"
        );
    }

    /// Two independent runs of the same scene must agree bit-for-bit. This is
    /// the crate's reproducibility guarantee, and it is fragile: any reduction
    /// whose order depends on thread scheduling (e.g. `par_iter().sum()` in the
    /// CG dot product) would make the recovered pressure — and every quantity
    /// derived from it — drift between runs. The solve runs in parallel, so this
    /// exercises the very code paths where such non-determinism would appear.
    #[test]
    fn simulation_is_bit_reproducible() {
        let run = || {
            let mut s = small_solver();
            for _ in 0..16 {
                s.step(0.005);
            }
            s
        };
        let a = run();
        let b = run();

        assert_eq!(
            a.particles.len(),
            b.particles.len(),
            "particle count diverged"
        );
        // Pressure field: compare raw bit patterns, not approximate equality.
        assert_eq!(a.pressure.len(), b.pressure.len());
        for (idx, (&pa, &pb)) in a.pressure.iter().zip(&b.pressure).enumerate() {
            assert_eq!(
                pa.to_bits(),
                pb.to_bits(),
                "pressure[{idx}] differs: {pa} vs {pb}"
            );
        }
        // Particle positions: every component identical down to the last bit.
        for (idx, (&qa, &qb)) in a
            .particles
            .positions
            .iter()
            .zip(&b.particles.positions)
            .enumerate()
        {
            assert_eq!(qa.x.to_bits(), qb.x.to_bits(), "particle[{idx}].x differs");
            assert_eq!(qa.y.to_bits(), qb.y.to_bits(), "particle[{idx}].y differs");
            assert_eq!(qa.z.to_bits(), qb.z.to_bits(), "particle[{idx}].z differs");
        }
    }

    /// A roomy closed box cavity (24 mm) carved from a solid block: `φ < 0`
    /// inside, `φ > 0` in the surrounding wall margin. Its large, simply
    /// connected interior gives every central cell six in-cavity neighbours,
    /// the clean setting needed to exercise particle bookkeeping directly.
    fn box_solver() -> Solver {
        let dx = 0.002;
        let origin = Vec3::new(-0.016, -0.016, -0.016);
        let a = 0.012;
        let sdf = Sdf::from_fn(origin, dx, (16, 16, 16), move |p| {
            (p.x.abs() - a).max(p.y.abs() - a).max(p.z.abs() - a)
        });
        let needle = Needle {
            tip: Vec3::new(0.0, 0.010, 0.0),
            axis: Vec3::new(0.0, -1.0, 0.0),
            radius: 0.0006,
            flow_rate: 5.0e-6,
            inject_until: f64::INFINITY,
        };
        let fluid = FluidParams {
            particles_per_cell: 8,
            ..FluidParams::default()
        };
        Solver::new(sdf, dx, fluid, needle)
    }

    /// Cram many particles into a single interior cell, far past the cap.
    fn pack_one_cell(s: &mut Solver, cell: (usize, usize, usize), n: usize) -> Vec3 {
        let center = s.grid.cell_center(cell.0, cell.1, cell.2);
        s.particles = ParticleSet::new();
        for idx in 0..n {
            // A deterministic sub-cell offset (|component| < 0.3·dx) keeps every
            // particle inside this one cell while giving them distinct positions.
            let pos = center + hash_jitter(idx as u64) * s.grid.dx;
            s.particles.push(pos, Vec3::ZERO);
        }
        center
    }

    /// The gravity-aware heart of the de-clumping fix: surplus from an
    /// over-packed cell must fall **downhill** — into the lowest neighbour that
    /// still has room — so the fluid stacks into a settling pool rather than
    /// diffusing isotropically toward a uniform fill (which would loft it back
    /// up against gravity). With a surplus small enough to fit in a single
    /// neighbour, every surplus particle should land in the one cell directly
    /// "below" along gravity, and none should be pushed uphill.
    #[test]
    fn redistribution_drains_surplus_downhill() {
        let mut s = box_solver();
        // Default gravity points along -y, so "down" is the j-1 neighbour.
        assert_eq!(s.fluid.gravity, Vec3::new(0.0, -9.81, 0.0));

        let cap = s.fluid.particles_per_cell; // redist_cap_factor defaults to 1.0
        let surplus = 4; // fits inside a single neighbour (surplus < cap)
        let n = cap + surplus;

        let cell = (8usize, 8usize, 8usize);
        pack_one_cell(&mut s, cell, n);
        s.redistribute();

        let count_at = |s: &Solver, i: usize, j: usize, k: usize| {
            s.particles
                .positions
                .iter()
                .filter(|&&p| s.cell_of(p) == Some((i, j, k)))
                .count()
        };
        let here = count_at(&s, cell.0, cell.1, cell.2);
        let below = count_at(&s, cell.0, cell.1 - 1, cell.2); // -y, downhill
        let above = count_at(&s, cell.0, cell.1 + 1, cell.2); // +y, uphill

        assert_eq!(s.particles.len(), n, "mass not conserved");
        assert_eq!(here, cap, "source cell should drain down to the cap");
        assert_eq!(
            below, surplus,
            "all surplus should fall into the cell directly below"
        );
        assert_eq!(above, 0, "nothing should be lofted uphill against gravity");
    }

    /// An over-packed cell must be relieved by *moving* its surplus into open
    /// neighbours — never creating or destroying particles (mass conservation),
    /// never pushing a *receiver* past the rest-density cap (so a settled pool
    /// is left alone, not lofted), never flinging any into the walls — and the
    /// whole operation must stay byte-for-byte reproducible.
    #[test]
    fn redistribution_relieves_overpacked_cell_and_conserves_mass() {
        const N: usize = 100;

        let mut s = box_solver();
        let cap = s.fluid.particles_per_cell; // redist_cap_factor defaults to 1.0
        let center = pack_one_cell(&mut s, (8, 8, 8), N);
        assert!(
            s.solid.sample(center) < 0.0,
            "the test cell must lie inside the cavity"
        );
        assert_eq!(
            s.max_particles_per_cell(),
            N,
            "setup should pile all {N} particles into one cell"
        );

        s.redistribute();

        // Mass conserved exactly — redistribution only ever *moves* particles.
        assert_eq!(
            s.particles.len(),
            N,
            "redistribution changed the particle count (mass not conserved)"
        );
        // The clump is relieved (peak strictly dropped from N). It need not reach
        // the cap in one call: each cell sheds only into neighbours that have
        // room, so the source drains gradually as gravity clears the space below.
        let peak = s.max_particles_per_cell();
        assert!(
            peak < N,
            "redistribution failed to relieve the clump (peak={peak}, N={N})"
        );
        // The cap binds on *receivers*: at most one cell (the still-draining
        // source) may sit above it — every cell that received surplus is held at
        // or below the cap, so a pool already at rest density is never overfilled.
        let (nx, ny, nz) = (s.grid.nx, s.grid.ny, s.grid.nz);
        let mut count = vec![0u32; nx * ny * nz];
        for &p in &s.particles.positions {
            if let Some((i, j, k)) = s.cell_of(p) {
                count[(k * ny + j) * nx + i] += 1;
            }
        }
        let over_cap = count.iter().filter(|&&c| c as usize > cap).count();
        assert!(
            over_cap <= 1,
            "a receiving cell was pushed past the cap (cells over cap: {over_cap})"
        );
        // Nothing was flung into the walls; every particle stays finite.
        assert!(s.particles.positions.iter().all(|p| p.is_finite()));
        assert!(
            s.particles
                .positions
                .iter()
                .all(|&p| s.solid.sample(p) < s.solid.dx),
            "a redistributed particle left the cavity"
        );

        // Determinism: an identical clump must redistribute to identical bits.
        let mut s2 = box_solver();
        pack_one_cell(&mut s2, (8, 8, 8), N);
        s2.redistribute();
        for (idx, (&a, &b)) in s
            .particles
            .positions
            .iter()
            .zip(&s2.particles.positions)
            .enumerate()
        {
            assert_eq!(
                a.x.to_bits(),
                b.x.to_bits(),
                "particle[{idx}].x not reproducible"
            );
            assert_eq!(
                a.y.to_bits(),
                b.y.to_bits(),
                "particle[{idx}].y not reproducible"
            );
            assert_eq!(
                a.z.to_bits(),
                b.z.to_bits(),
                "particle[{idx}].z not reproducible"
            );
        }
    }
}
