//! Pressure projection: the incompressibility solve at the core of the fluid
//! simulator.
//!
//! After advection and body forces we hold an intermediate velocity field `u*`
//! that is generally *not* divergence-free. We find a pressure `p` such that
//!
//! ```text
//!     u^{n+1} = u* − (dt/ρ) ∇p ,      ∇·u^{n+1} = 0
//! ```
//!
//! Taking the divergence turns this into a Poisson equation `∇²p = (ρ/dt) ∇·u*`
//! with Neumann conditions at solid walls and a Dirichlet `p = 0` condition at
//! the free surface (air). Discretised on the MAC grid this is a large, sparse,
//! symmetric positive-definite linear system which we solve with a
//! **conjugate-gradient** iteration preconditioned by a **Modified Incomplete
//! Cholesky** factorisation, `MIC(0)` — the method recommended by Bridson,
//! *Fluid Simulation for Computer Graphics*.

use crate::grid::MacGrid;
use rayon::prelude::*;

/// Classification of every grid cell for the pressure solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cell {
    /// Inside a wall / outside the domain. Neumann (∂p/∂n = 0); excluded from
    /// the linear system.
    Solid,
    /// Contains fluid. An unknown in the linear system.
    Fluid,
    /// Empty space inside the cavity (free surface). Dirichlet `p = 0`.
    Air,
}

/// Tunables for the conjugate-gradient solve.
#[derive(Debug, Clone, Copy)]
pub struct SolveParams {
    pub max_iters: usize,
    /// Convergence tolerance on the infinity-norm of the residual (units of the
    /// scaled divergence right-hand side).
    pub tol: f64,
}

impl Default for SolveParams {
    fn default() -> Self {
        SolveParams { max_iters: 200, tol: 1e-6 }
    }
}

/// Result of a projection: how hard the solve had to work.
#[derive(Debug, Clone, Copy)]
pub struct SolveReport {
    pub iters: usize,
    pub residual: f64,
}

/// The symmetric 7-point Laplacian, stored as a diagonal plus the three
/// "positive" off-diagonals (to the +x, +y, +z neighbours). Symmetry gives the
/// negative off-diagonals for free.
struct Laplacian {
    diag: Vec<f64>,
    plus_x: Vec<f64>,
    plus_y: Vec<f64>,
    plus_z: Vec<f64>,
    nx: usize,
    ny: usize,
    nz: usize,
}

impl Laplacian {
    #[inline]
    fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        (k * self.ny + j) * self.nx + i
    }
}

/// Project the velocity field in `grid` onto its divergence-free part.
///
/// * `cells` — per-cell classification (length `nx*ny*nz`).
/// * `rho` — fluid density (kg/m³).
/// * `dt` — timestep (s).
///
/// On return the face velocities of `grid` are (approximately) divergence-free
/// inside the fluid, faces touching solids carry the (zero) wall velocity, and
/// the report gives the iteration count and final residual.
pub fn project(grid: &mut MacGrid, cells: &[Cell], rho: f64, dt: f64, params: SolveParams) -> SolveReport {
    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
    let n = nx * ny * nz;
    assert_eq!(cells.len(), n, "cell classification must cover the whole grid");

    let cell = |i: usize, j: usize, k: usize| cells[(k * ny + j) * nx + i];

    // 1. Enforce solid (no-through-flow) velocities on every face that touches
    //    a solid cell or the domain boundary. Walls are static => zero.
    enforce_solid_faces(grid, cells);

    // 2. Build the right-hand side b = −(ρ·dx/dt)·D, where D is the raw face
    //    divergence of each fluid cell.
    let inv_scale = rho * grid.dx / dt;
    let mut b = vec![0.0f64; n];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let c = (k * ny + j) * nx + i;
                if cells[c] != Cell::Fluid {
                    continue;
                }
                let div = (grid.u[grid.u_idx(i + 1, j, k)] - grid.u[grid.u_idx(i, j, k)])
                    + (grid.v[grid.v_idx(i, j + 1, k)] - grid.v[grid.v_idx(i, j, k)])
                    + (grid.w[grid.w_idx(i, j, k + 1)] - grid.w[grid.w_idx(i, j, k)]);
                b[c] = -inv_scale * div;
            }
        }
    }

    // 3. Assemble the Laplacian matrix.
    let mut a = Laplacian {
        diag: vec![0.0; n],
        plus_x: vec![0.0; n],
        plus_y: vec![0.0; n],
        plus_z: vec![0.0; n],
        nx,
        ny,
        nz,
    };
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let c = (k * ny + j) * nx + i;
                if cells[c] != Cell::Fluid {
                    continue;
                }
                // Each non-solid neighbour adds 1 to the diagonal; each *fluid*
                // neighbour additionally couples via a −1 off-diagonal.
                let mut diag = 0.0;
                let mut consider = |ni: i64, nj: i64, nk: i64, plus: Option<&mut f64>| {
                    let inside = ni >= 0
                        && nj >= 0
                        && nk >= 0
                        && (ni as usize) < nx
                        && (nj as usize) < ny
                        && (nk as usize) < nz;
                    let kind = if inside {
                        cell(ni as usize, nj as usize, nk as usize)
                    } else {
                        Cell::Solid
                    };
                    match kind {
                        Cell::Solid => {}
                        Cell::Air => diag += 1.0,
                        Cell::Fluid => {
                            diag += 1.0;
                            if let Some(p) = plus {
                                *p = -1.0;
                            }
                        }
                    }
                };
                let (i, j, k) = (i as i64, j as i64, k as i64);
                consider(i - 1, j, k, None);
                consider(i + 1, j, k, Some(&mut a.plus_x[c]));
                consider(i, j - 1, k, None);
                consider(i, j + 1, k, Some(&mut a.plus_y[c]));
                consider(i, j, k - 1, None);
                consider(i, j, k + 1, Some(&mut a.plus_z[c]));
                a.diag[c] = diag;
            }
        }
    }

    // 4. Preconditioned conjugate gradient.
    let report = pcg(&a, cells, &b, params);

    // 5. Subtract the pressure gradient from the velocity field.
    let coef = dt / (rho * grid.dx);
    apply_pressure_gradient(grid, cells, &report.pressure, coef);

    SolveReport { iters: report.iters, residual: report.residual }
}

/// Set every face touching a solid (or the domain boundary) to the static wall
/// velocity of zero.
fn enforce_solid_faces(grid: &mut MacGrid, cells: &[Cell]) {
    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
    let solid = |i: i64, j: i64, k: i64| -> bool {
        if i < 0 || j < 0 || k < 0 || i as usize >= nx || j as usize >= ny || k as usize >= nz {
            return true;
        }
        cells[(k as usize * ny + j as usize) * nx + i as usize] == Cell::Solid
    };
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..=nx {
                if solid(i as i64 - 1, j as i64, k as i64) || solid(i as i64, j as i64, k as i64) {
                    let idx = grid.u_idx(i, j, k);
                    grid.u[idx] = 0.0;
                }
            }
        }
    }
    for k in 0..nz {
        for j in 0..=ny {
            for i in 0..nx {
                if solid(i as i64, j as i64 - 1, k as i64) || solid(i as i64, j as i64, k as i64) {
                    let idx = grid.v_idx(i, j, k);
                    grid.v[idx] = 0.0;
                }
            }
        }
    }
    for k in 0..=nz {
        for j in 0..ny {
            for i in 0..nx {
                if solid(i as i64, j as i64, k as i64 - 1) || solid(i as i64, j as i64, k as i64) {
                    let idx = grid.w_idx(i, j, k);
                    grid.w[idx] = 0.0;
                }
            }
        }
    }
}

/// Subtract `coef·∇p` from the velocity on every face between two non-solid
/// cells where at least one side is fluid. Air pressure is zero.
fn apply_pressure_gradient(grid: &mut MacGrid, cells: &[Cell], p: &[f64], coef: f64) {
    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
    let at = |i: usize, j: usize, k: usize| cells[(k * ny + j) * nx + i];
    let pr = |i: usize, j: usize, k: usize| p[(k * ny + j) * nx + i];

    for k in 0..nz {
        for j in 0..ny {
            for i in 1..nx {
                let (l, r) = (at(i - 1, j, k), at(i, j, k));
                if l == Cell::Solid || r == Cell::Solid {
                    continue;
                }
                if l == Cell::Fluid || r == Cell::Fluid {
                    let idx = grid.u_idx(i, j, k);
                    grid.u[idx] -= coef * (pr(i, j, k) - pr(i - 1, j, k));
                }
            }
        }
    }
    for k in 0..nz {
        for j in 1..ny {
            for i in 0..nx {
                let (l, r) = (at(i, j - 1, k), at(i, j, k));
                if l == Cell::Solid || r == Cell::Solid {
                    continue;
                }
                if l == Cell::Fluid || r == Cell::Fluid {
                    let idx = grid.v_idx(i, j, k);
                    grid.v[idx] -= coef * (pr(i, j, k) - pr(i, j - 1, k));
                }
            }
        }
    }
    for k in 1..nz {
        for j in 0..ny {
            for i in 0..nx {
                let (l, r) = (at(i, j, k - 1), at(i, j, k));
                if l == Cell::Solid || r == Cell::Solid {
                    continue;
                }
                if l == Cell::Fluid || r == Cell::Fluid {
                    let idx = grid.w_idx(i, j, k);
                    grid.w[idx] -= coef * (pr(i, j, k) - pr(i, j, k - 1));
                }
            }
        }
    }
}

struct PcgResult {
    pressure: Vec<f64>,
    iters: usize,
    residual: f64,
}

/// Conjugate gradient with an MIC(0) preconditioner.
fn pcg(a: &Laplacian, cells: &[Cell], b: &[f64], params: SolveParams) -> PcgResult {
    let n = b.len();
    let mut p = vec![0.0f64; n]; // solution (pressure)

    let mut residual = inf_norm(b);
    if residual <= params.tol {
        return PcgResult { pressure: p, iters: 0, residual };
    }

    let precon = build_mic0(a, cells);
    let mut r = b.to_vec();
    let mut z = vec![0.0f64; n];
    apply_mic0(a, &precon, cells, &r, &mut z);
    let mut s = z.clone();
    let mut sigma = dot(&z, &r);
    let mut as_ = vec![0.0f64; n];

    let mut iters = 0;
    for it in 0..params.max_iters {
        iters = it + 1;
        apply_a(a, cells, &s, &mut as_);
        let denom = dot(&s, &as_);
        if denom.abs() < 1e-30 {
            break;
        }
        let alpha = sigma / denom;
        // p += alpha s ; r -= alpha As
        p.par_iter_mut().zip(&s).for_each(|(p, &s)| *p += alpha * s);
        r.par_iter_mut().zip(&as_).for_each(|(r, &a)| *r -= alpha * a);

        residual = inf_norm(&r);
        if residual <= params.tol {
            break;
        }

        apply_mic0(a, &precon, cells, &r, &mut z);
        let sigma_new = dot(&z, &r);
        let beta = sigma_new / sigma;
        // s = z + beta s
        s.par_iter_mut().zip(&z).for_each(|(s, &z)| *s = z + beta * *s);
        sigma = sigma_new;
    }

    PcgResult { pressure: p, iters, residual }
}

/// Sparse matrix-vector product `out = A·x` over fluid cells.
fn apply_a(a: &Laplacian, cells: &[Cell], x: &[f64], out: &mut [f64]) {
    let (nx, ny, nz) = (a.nx, a.ny, a.nz);
    out.par_chunks_mut(nx * ny).enumerate().for_each(|(k, slice)| {
        for j in 0..ny {
            for i in 0..nx {
                let c = (k * ny + j) * nx + i;
                let local = j * nx + i;
                if cells[c] != Cell::Fluid {
                    slice[local] = 0.0;
                    continue;
                }
                let mut v = a.diag[c] * x[c];
                // +x / −x
                if i + 1 < nx {
                    v += a.plus_x[c] * x[a.idx(i + 1, j, k)];
                }
                if i >= 1 {
                    v += a.plus_x[a.idx(i - 1, j, k)] * x[a.idx(i - 1, j, k)];
                }
                // +y / −y
                if j + 1 < ny {
                    v += a.plus_y[c] * x[a.idx(i, j + 1, k)];
                }
                if j >= 1 {
                    v += a.plus_y[a.idx(i, j - 1, k)] * x[a.idx(i, j - 1, k)];
                }
                // +z / −z
                if k + 1 < nz {
                    v += a.plus_z[c] * x[a.idx(i, j, k + 1)];
                }
                if k >= 1 {
                    v += a.plus_z[a.idx(i, j, k - 1)] * x[a.idx(i, j, k - 1)];
                }
                slice[local] = v;
            }
        }
    });
}

/// Build the MIC(0) preconditioner factor (one reciprocal-sqrt per fluid cell).
fn build_mic0(a: &Laplacian, cells: &[Cell]) -> Vec<f64> {
    const TAU: f64 = 0.97; // modification constant
    const SIGMA: f64 = 0.25; // safety floor
    let (nx, ny, nz) = (a.nx, a.ny, a.nz);
    let mut precon = vec![0.0f64; nx * ny * nz];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let c = (k * ny + j) * nx + i;
                if cells[c] != Cell::Fluid {
                    continue;
                }
                let (axi, pxi) = if i >= 1 {
                    let cm = a.idx(i - 1, j, k);
                    (a.plus_x[cm], precon[cm])
                } else {
                    (0.0, 0.0)
                };
                let (ayj, pyj) = if j >= 1 {
                    let cm = a.idx(i, j - 1, k);
                    (a.plus_y[cm], precon[cm])
                } else {
                    (0.0, 0.0)
                };
                let (azk, pzk) = if k >= 1 {
                    let cm = a.idx(i, j, k - 1);
                    (a.plus_z[cm], precon[cm])
                } else {
                    (0.0, 0.0)
                };

                let term_x = axi * pxi;
                let term_y = ayj * pyj;
                let term_z = azk * pzk;

                // Off-diagonals of the −x neighbour into the +y/+z directions, etc.
                let (axi_y, axi_z) = if i >= 1 {
                    let cm = a.idx(i - 1, j, k);
                    (a.plus_y[cm], a.plus_z[cm])
                } else {
                    (0.0, 0.0)
                };
                let (ayj_x, ayj_z) = if j >= 1 {
                    let cm = a.idx(i, j - 1, k);
                    (a.plus_x[cm], a.plus_z[cm])
                } else {
                    (0.0, 0.0)
                };
                let (azk_x, azk_y) = if k >= 1 {
                    let cm = a.idx(i, j, k - 1);
                    (a.plus_x[cm], a.plus_y[cm])
                } else {
                    (0.0, 0.0)
                };

                let mut e = a.diag[c]
                    - term_x * term_x
                    - term_y * term_y
                    - term_z * term_z
                    - TAU
                        * (axi * (axi_y + axi_z) * pxi * pxi
                            + ayj * (ayj_x + ayj_z) * pyj * pyj
                            + azk * (azk_x + azk_y) * pzk * pzk);
                if e < SIGMA * a.diag[c] {
                    e = a.diag[c];
                }
                precon[c] = if e > 0.0 { 1.0 / e.sqrt() } else { 0.0 };
            }
        }
    }
    precon
}

/// Apply the MIC(0) preconditioner: solve `M z = r` via forward then backward
/// substitution. Inherently sequential.
fn apply_mic0(a: &Laplacian, precon: &[f64], cells: &[Cell], r: &[f64], z: &mut [f64]) {
    let (nx, ny, nz) = (a.nx, a.ny, a.nz);
    // q holds the intermediate forward-substitution result; reuse z's storage
    // is unsafe here because we need both, so use a scratch vector.
    let mut q = vec![0.0f64; r.len()];
    // Forward solve L q = r.
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let c = (k * ny + j) * nx + i;
                if cells[c] != Cell::Fluid {
                    continue;
                }
                let mut t = r[c];
                if i >= 1 {
                    let cm = a.idx(i - 1, j, k);
                    t -= a.plus_x[cm] * precon[cm] * q[cm];
                }
                if j >= 1 {
                    let cm = a.idx(i, j - 1, k);
                    t -= a.plus_y[cm] * precon[cm] * q[cm];
                }
                if k >= 1 {
                    let cm = a.idx(i, j, k - 1);
                    t -= a.plus_z[cm] * precon[cm] * q[cm];
                }
                q[c] = t * precon[c];
            }
        }
    }
    // Backward solve Lᵀ z = q.
    for k in (0..nz).rev() {
        for j in (0..ny).rev() {
            for i in (0..nx).rev() {
                let c = (k * ny + j) * nx + i;
                if cells[c] != Cell::Fluid {
                    z[c] = 0.0;
                    continue;
                }
                let mut t = q[c];
                if i + 1 < nx {
                    t -= a.plus_x[c] * precon[c] * z[a.idx(i + 1, j, k)];
                }
                if j + 1 < ny {
                    t -= a.plus_y[c] * precon[c] * z[a.idx(i, j + 1, k)];
                }
                if k + 1 < nz {
                    t -= a.plus_z[c] * precon[c] * z[a.idx(i, j, k + 1)];
                }
                z[c] = t * precon[c];
            }
        }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.par_iter().zip(b).map(|(&x, &y)| x * y).sum()
}

fn inf_norm(a: &[f64]) -> f64 {
    a.par_iter().fold(|| 0.0f64, |m, &x| m.max(x.abs())).reduce(|| 0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec3;

    /// Projection is a true projection operator: applying it to an already
    /// divergence-free (wall-respecting) field leaves it unchanged (`P² = P`).
    #[test]
    fn projection_is_idempotent() {
        let mut g = MacGrid::new(12, 12, 12, 0.01, Vec3::ZERO);
        fill_divergent(&mut g);
        let cells = classify_box(&g);
        let solve = SolveParams { max_iters: 400, tol: 1e-12 };
        // First projection makes the field divergence-free.
        project(&mut g, &cells, 1000.0, 1e-3, solve);
        let (u1, v1, w1) = (g.u.clone(), g.v.clone(), g.w.clone());
        // Second projection should be a no-op.
        project(&mut g, &cells, 1000.0, 1e-3, solve);
        let max_change = g
            .u
            .iter()
            .zip(&u1)
            .chain(g.v.iter().zip(&v1))
            .chain(g.w.iter().zip(&w1))
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        assert!(max_change < 1e-7, "projection not idempotent: changed by {max_change}");
    }

    /// Fill a grid with an arbitrary, strongly divergent velocity field.
    fn fill_divergent(g: &mut MacGrid) {
        for k in 0..g.nz {
            for j in 0..g.ny {
                for i in 0..=g.nx {
                    let idx = g.u_idx(i, j, k);
                    g.u[idx] = (i as f64 - 6.0) * 0.05 + (j as f64 * 0.013).sin();
                }
            }
        }
        for k in 0..g.nz {
            for j in 0..=g.ny {
                for i in 0..g.nx {
                    let idx = g.v_idx(i, j, k);
                    g.v[idx] = (j as f64 - 6.0) * 0.05 + (k as f64 * 0.017).cos();
                }
            }
        }
        for k in 0..=g.nz {
            for j in 0..g.ny {
                for i in 0..g.nx {
                    let idx = g.w_idx(i, j, k);
                    g.w[idx] = (k as f64 - 6.0) * 0.03;
                }
            }
        }
    }

    /// An arbitrary (divergent) field must come out essentially divergence-free
    /// inside the fluid region.
    #[test]
    fn removes_divergence() {
        let mut g = MacGrid::new(12, 12, 12, 0.01, Vec3::ZERO);
        // A radially expanding field has strong positive divergence.
        for k in 0..g.nz {
            for j in 0..g.ny {
                for i in 0..=g.nx {
                    let idx = g.u_idx(i, j, k);
                    g.u[idx] = (i as f64 - 6.0) * 0.05;
                }
            }
        }
        for k in 0..g.nz {
            for j in 0..=g.ny {
                for i in 0..g.nx {
                    let idx = g.v_idx(i, j, k);
                    g.v[idx] = (j as f64 - 6.0) * 0.05;
                }
            }
        }
        let cells = classify_box(&g);
        let rep = project(&mut g, &cells, 1000.0, 1e-3, SolveParams { max_iters: 400, tol: 1e-9 });

        // Check max divergence over interior fluid cells.
        let mut max_div = 0.0f64;
        for k in 1..g.nz - 1 {
            for j in 1..g.ny - 1 {
                for i in 1..g.nx - 1 {
                    let div = (g.u[g.u_idx(i + 1, j, k)] - g.u[g.u_idx(i, j, k)])
                        + (g.v[g.v_idx(i, j + 1, k)] - g.v[g.v_idx(i, j, k)])
                        + (g.w[g.w_idx(i, j, k + 1)] - g.w[g.w_idx(i, j, k)]);
                    max_div = max_div.max(div.abs());
                }
            }
        }
        assert!(max_div < 1e-5, "max interior divergence after projection: {max_div} (iters {})", rep.iters);
    }

    /// Build a classification with a 1-cell solid wall around an all-fluid core.
    fn classify_box(g: &MacGrid) -> Vec<Cell> {
        let (nx, ny, nz) = (g.nx, g.ny, g.nz);
        let mut cells = vec![Cell::Fluid; nx * ny * nz];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let edge = i == 0 || j == 0 || k == 0 || i == nx - 1 || j == ny - 1 || k == nz - 1;
                    if edge {
                        cells[(k * ny + j) * nx + i] = Cell::Solid;
                    }
                }
            }
        }
        cells
    }
}
