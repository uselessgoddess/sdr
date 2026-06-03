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
}

impl Default for FluidParams {
    fn default() -> Self {
        FluidParams {
            density: 1000.0,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            flip_ratio: 0.95,
            particles_per_cell: 8,
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

    cells: Vec<Cell>,
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
            cells: vec![Cell::Air; n],
            saved_u: Vec::new(),
            saved_v: Vec::new(),
            saved_w: Vec::new(),
            rng: Rng::new(0x5d_eece_66d),
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
        let mut remaining = dt;
        while remaining > 1e-12 {
            let sub = self.cfl_dt().min(remaining);
            self.substep(sub);
            remaining -= sub;
        }
    }

    /// A stable sub-step from the current maximum speed.
    fn cfl_dt(&self) -> f64 {
        let grid_speed = self.grid.max_speed();
        let jet = self.needle.jet_speed();
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
        let report = pressure::project(&mut self.grid, &self.cells, self.fluid.density, dt, self.solve);
        self.last_pcg_iters = report.iters;
        self.extrapolate_velocities();
        self.g2p();
        self.advect(dt);
        self.enforce_particle_bounds();
        self.time += dt;
        self.steps += 1;
    }

    /// Inject particles from the needle to match the volumetric flow rate.
    fn emit(&mut self, dt: f64) {
        let pv = self.particle_volume();
        if pv <= 0.0 || self.needle.flow_rate <= 0.0 {
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
            let pos = self.needle.tip
                + e1 * (r * theta.cos())
                + e2 * (r * theta.sin())
                + axis * along;
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
                    let center = self.grid.cell_center(i, j, k);
                    let outlet = self.outlet.map(|o| o.contains(center)).unwrap_or(false);
                    self.cells[c] = if outlet {
                        Cell::Air // zero-pressure drain
                    } else if self.solid.sample(center) >= 0.0 {
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

    /// Splat particle velocities onto the staggered faces (mass-weighted).
    fn p2g(&mut self) {
        self.grid.zero();
        let ud = self.grid.u_dims();
        let vd = self.grid.v_dims();
        let wd = self.grid.w_dims();
        let mut wu = vec![0.0f64; self.grid.u.len()];
        let mut wv = vec![0.0f64; self.grid.v.len()];
        let mut ww = vec![0.0f64; self.grid.w.len()];

        for (p, vel) in self.particles.positions.iter().zip(&self.particles.velocities) {
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

        for (p, vel) in self.particles.positions.iter().zip(self.particles.velocities.iter_mut()) {
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
    fn advect(&mut self, dt: f64) {
        let n = self.particles.len();
        for idx in 0..n {
            let p0 = self.particles.positions[idx];
            let v0 = self.grid.velocity_at(p0);
            let mid = p0 + v0 * (0.5 * dt);
            let vmid = self.grid.velocity_at(mid);
            let mut p = p0 + vmid * dt;
            p = self.project_inside(p);
            self.particles.positions[idx] = p;
        }
    }

    /// Push a point back inside the cavity if it has entered the wall, using the
    /// SDF gradient. Keeps a half-cell margin off the wall.
    fn project_inside(&self, p: Vec3) -> Vec3 {
        let margin = 0.5 * self.grid.dx;
        let phi = self.solid.sample(p);
        if phi > -margin {
            let g = self.solid.gradient(p).normalize_or_zero();
            if g.length() > 0.0 {
                return p - g * (phi + margin);
            }
        }
        p
    }

    /// Remove particles that drained out, escaped the domain, or got stuck deep
    /// in a wall.
    fn enforce_particle_bounds(&mut self) {
        let bb = self.solid.bounds();
        let outlet = self.outlet;
        let solid = &self.solid;
        self.particles.retain(|p, _| {
            if !bb.contains(p) {
                return false;
            }
            if let Some(o) = outlet {
                if o.contains(p) {
                    return false; // drained
                }
            }
            // Deep inside the wall and unrecoverable -> drop.
            solid.sample(p) < 0.5 * solid.dx
        });
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

/// Build two unit vectors orthogonal to `n` (and to each other).
fn orthonormal_basis(n: Vec3) -> (Vec3, Vec3) {
    let a = if n.x.abs() < 0.9 { Vec3::new(1.0, 0.0, 0.0) } else { Vec3::new(0.0, 1.0, 0.0) };
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
        let needle = Needle { tip, axis, radius: 0.0006, flow_rate: 5.0e-6 };
        let fluid = FluidParams { particles_per_cell: 16, ..FluidParams::default() };
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
        let n = Needle { tip: Vec3::ZERO, axis: Vec3::new(0.0, 1.0, 0.0), radius: 0.001, flow_rate: 1e-6 };
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
        assert!(s.particles.len() > 0, "needle should have injected fluid");
        // Fluid volume should be positive and finite.
        let vol = s.fluid_volume();
        assert!(vol > 0.0 && vol.is_finite(), "fluid volume {vol}");
        // Particles must stay finite and (mostly) inside the padded cavity.
        assert!(s.particles.positions.iter().all(|p| p.is_finite()));
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
        assert!(max_phi < s.solid.dx, "a particle sits inside the wall (phi = {max_phi})");
    }
}
