//! Staggered **MAC** (Marker-And-Cell) grid for incompressible flow.
//!
//! Scalars (pressure, divergence, cell type) live at cell centres; velocity
//! components live on the cell faces they are normal to:
//!
//! * `u` on x-faces, dims `(nx+1, ny, nz)`
//! * `v` on y-faces, dims `(nx, ny+1, nz)`
//! * `w` on z-faces, dims `(nx, ny, nz+1)`
//!
//! Staggering places each pressure-gradient and divergence stencil exactly
//! where it is needed, which is what makes the pressure projection clean and
//! free of the checkerboard artefacts a collocated grid would suffer.

use crate::math::Vec3;

/// Clamped trilinear interpolation of a scalar array with the given node dims.
/// `(gx, gy, gz)` are fractional node coordinates.
pub fn trilinear_sample(
    arr: &[f64],
    dims: (usize, usize, usize),
    gx: f64,
    gy: f64,
    gz: f64,
) -> f64 {
    let (nx, ny, nz) = dims;
    let clamp_base = |g: f64, n: usize| -> (usize, f64) {
        if g <= 0.0 {
            (0, 0.0)
        } else if g >= (n - 1) as f64 {
            (n - 2, 1.0)
        } else {
            let i = g.floor();
            (i as usize, g - i)
        }
    };
    let (i, fx) = clamp_base(gx, nx);
    let (j, fy) = clamp_base(gy, ny);
    let (k, fz) = clamp_base(gz, nz);
    let at = |i: usize, j: usize, k: usize| arr[(k * ny + j) * nx + i];
    let c00 = at(i, j, k) * (1.0 - fx) + at(i + 1, j, k) * fx;
    let c10 = at(i, j + 1, k) * (1.0 - fx) + at(i + 1, j + 1, k) * fx;
    let c01 = at(i, j, k + 1) * (1.0 - fx) + at(i + 1, j, k + 1) * fx;
    let c11 = at(i, j + 1, k + 1) * (1.0 - fx) + at(i + 1, j + 1, k + 1) * fx;
    let c0 = c00 * (1.0 - fy) + c10 * fy;
    let c1 = c01 * (1.0 - fy) + c11 * fy;
    c0 * (1.0 - fz) + c1 * fz
}

/// Scatter `val` into `arr` (and the matching weight into `wsum`) using the
/// same trilinear stencil as [`trilinear_sample`]. This pairing makes
/// particle→grid then grid→particle transfer of a constant field exact.
pub fn trilinear_scatter(
    arr: &mut [f64],
    wsum: &mut [f64],
    dims: (usize, usize, usize),
    gx: f64,
    gy: f64,
    gz: f64,
    val: f64,
) {
    let (nx, ny, nz) = dims;
    let clamp_base = |g: f64, n: usize| -> (usize, f64) {
        if g <= 0.0 {
            (0, 0.0)
        } else if g >= (n - 1) as f64 {
            (n - 2, 1.0)
        } else {
            let i = g.floor();
            (i as usize, g - i)
        }
    };
    let (i, fx) = clamp_base(gx, nx);
    let (j, fy) = clamp_base(gy, ny);
    let (k, fz) = clamp_base(gz, nz);
    let wx = [1.0 - fx, fx];
    let wy = [1.0 - fy, fy];
    let wz = [1.0 - fz, fz];
    for (dk, &wzk) in wz.iter().enumerate() {
        for (dj, &wyj) in wy.iter().enumerate() {
            for (di, &wxi) in wx.iter().enumerate() {
                let w = wxi * wyj * wzk;
                let idx = ((k + dk) * ny + (j + dj)) * nx + (i + di);
                arr[idx] += w * val;
                wsum[idx] += w;
            }
        }
    }
}

/// A staggered velocity field on a uniform grid of `nx*ny*nz` cells.
#[derive(Debug, Clone)]
pub struct MacGrid {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f64,
    pub origin: Vec3,
    pub u: Vec<f64>,
    pub v: Vec<f64>,
    pub w: Vec<f64>,
}

impl MacGrid {
    pub fn new(nx: usize, ny: usize, nz: usize, dx: f64, origin: Vec3) -> Self {
        MacGrid {
            nx,
            ny,
            nz,
            dx,
            origin,
            u: vec![0.0; (nx + 1) * ny * nz],
            v: vec![0.0; nx * (ny + 1) * nz],
            w: vec![0.0; nx * ny * (nz + 1)],
        }
    }

    pub fn u_dims(&self) -> (usize, usize, usize) {
        (self.nx + 1, self.ny, self.nz)
    }
    pub fn v_dims(&self) -> (usize, usize, usize) {
        (self.nx, self.ny + 1, self.nz)
    }
    pub fn w_dims(&self) -> (usize, usize, usize) {
        (self.nx, self.ny, self.nz + 1)
    }

    #[inline]
    pub fn u_idx(&self, i: usize, j: usize, k: usize) -> usize {
        (k * self.ny + j) * (self.nx + 1) + i
    }
    #[inline]
    pub fn v_idx(&self, i: usize, j: usize, k: usize) -> usize {
        (k * (self.ny + 1) + j) * self.nx + i
    }
    #[inline]
    pub fn w_idx(&self, i: usize, j: usize, k: usize) -> usize {
        (k * self.ny + j) * self.nx + i
    }
    #[inline]
    pub fn cell_idx(&self, i: usize, j: usize, k: usize) -> usize {
        (k * self.ny + j) * self.nx + i
    }
    #[inline]
    pub fn cell_count(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    /// World position of the centre of cell `(i,j,k)`.
    #[inline]
    pub fn cell_center(&self, i: usize, j: usize, k: usize) -> Vec3 {
        self.origin
            + Vec3::new(
                (i as f64 + 0.5) * self.dx,
                (j as f64 + 0.5) * self.dx,
                (k as f64 + 0.5) * self.dx,
            )
    }

    // Fractional grid coordinates for each staggered component. The 0.5 offsets
    // encode where each component physically sits.
    #[inline]
    pub fn u_coords(&self, p: Vec3) -> (f64, f64, f64) {
        let l = (p - self.origin) / self.dx;
        (l.x, l.y - 0.5, l.z - 0.5)
    }
    #[inline]
    pub fn v_coords(&self, p: Vec3) -> (f64, f64, f64) {
        let l = (p - self.origin) / self.dx;
        (l.x - 0.5, l.y, l.z - 0.5)
    }
    #[inline]
    pub fn w_coords(&self, p: Vec3) -> (f64, f64, f64) {
        let l = (p - self.origin) / self.dx;
        (l.x - 0.5, l.y - 0.5, l.z)
    }

    pub fn sample_u(&self, p: Vec3) -> f64 {
        let (gx, gy, gz) = self.u_coords(p);
        trilinear_sample(&self.u, self.u_dims(), gx, gy, gz)
    }
    pub fn sample_v(&self, p: Vec3) -> f64 {
        let (gx, gy, gz) = self.v_coords(p);
        trilinear_sample(&self.v, self.v_dims(), gx, gy, gz)
    }
    pub fn sample_w(&self, p: Vec3) -> f64 {
        let (gx, gy, gz) = self.w_coords(p);
        trilinear_sample(&self.w, self.w_dims(), gx, gy, gz)
    }

    /// Trilinearly interpolated velocity at world point `p`.
    pub fn velocity_at(&self, p: Vec3) -> Vec3 {
        Vec3::new(self.sample_u(p), self.sample_v(p), self.sample_w(p))
    }

    pub fn zero(&mut self) {
        self.u.iter_mut().for_each(|x| *x = 0.0);
        self.v.iter_mut().for_each(|x| *x = 0.0);
        self.w.iter_mut().for_each(|x| *x = 0.0);
    }

    /// Largest velocity magnitude stored on any face (for CFL).
    pub fn max_speed(&self) -> f64 {
        let m = |a: &[f64]| a.iter().fold(0.0f64, |m, &x| m.max(x.abs()));
        // Per-component bound; the true vector magnitude is at most sqrt(3)x.
        (m(&self.u).powi(2) + m(&self.v).powi(2) + m(&self.w).powi(2)).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn constant_field_samples_exactly() {
        let mut g = MacGrid::new(8, 8, 8, 0.1, Vec3::ZERO);
        g.u.iter_mut().for_each(|x| *x = 3.0);
        g.v.iter_mut().for_each(|x| *x = -2.0);
        g.w.iter_mut().for_each(|x| *x = 1.0);
        let vel = g.velocity_at(Vec3::new(0.41, 0.37, 0.55));
        assert_relative_eq!(vel.x, 3.0, epsilon = 1e-9);
        assert_relative_eq!(vel.y, -2.0, epsilon = 1e-9);
        assert_relative_eq!(vel.z, 1.0, epsilon = 1e-9);
    }

    #[test]
    fn scatter_then_sample_recovers_constant() {
        // Scatter a constant value from many particles, normalise, then sample.
        let g = MacGrid::new(6, 6, 6, 0.1, Vec3::ZERO);
        let dims = g.u_dims();
        let mut arr = vec![0.0; (g.nx + 1) * g.ny * g.nz];
        let mut wsum = vec![0.0; arr.len()];
        // Fill the domain densely with particles carrying u = 5.0.
        let n = 12;
        for a in 0..n {
            for b in 0..n {
                for c in 0..n {
                    let p = Vec3::new(
                        0.05 + a as f64 / n as f64 * 0.5,
                        0.05 + b as f64 / n as f64 * 0.5,
                        0.05 + c as f64 / n as f64 * 0.5,
                    );
                    let (gx, gy, gz) = g.u_coords(p);
                    trilinear_scatter(&mut arr, &mut wsum, dims, gx, gy, gz, 5.0);
                }
            }
        }
        for (a, w) in arr.iter_mut().zip(&wsum) {
            if *w > 1e-12 {
                *a /= *w;
            }
        }
        // Interior value should be ~5.0.
        let mid = trilinear_sample(&arr, dims, 3.0, 2.5, 2.5);
        assert_relative_eq!(mid, 5.0, epsilon = 1e-6);
    }
}
