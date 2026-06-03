//! Signed distance field (level set) sampled on a regular grid.
//!
//! The solver needs to know, at any point in space, (a) whether it is inside
//! the sinus cavity (where fluid and air live) or inside the bony wall (solid),
//! and (b) how far it is from the wall and in which direction. A signed
//! distance field answers both.
//!
//! **Sign convention:** `phi(x) < 0` *inside the cavity* (the fluid/air
//! region), `phi(x) > 0` *inside the solid* wall. The wall surface is the
//! `phi = 0` isosurface.
//!
//! The field is built from a closed triangle mesh of the cavity. Distance
//! magnitude comes from the closest point on the mesh; the sign comes from the
//! *generalized winding number*, which is robust for watertight meshes and
//! degrades gracefully near edges and vertices.

use crate::math::{Aabb, Vec3};
use crate::mesh::TriMesh;
use rayon::prelude::*;

/// A scalar signed-distance field on a uniform grid of node values.
#[derive(Debug, Clone)]
pub struct Sdf {
    /// World position of node (0,0,0).
    pub origin: Vec3,
    /// Node spacing (uniform, metres).
    pub dx: f64,
    /// Node counts along each axis.
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    /// `nx*ny*nz` signed-distance samples, x-fastest.
    pub data: Vec<f64>,
}

impl Sdf {
    #[inline]
    fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        (k * self.ny + j) * self.nx + i
    }

    /// Build a signed distance field from a closed triangle mesh.
    ///
    /// * `dx` — node spacing in metres.
    /// * `padding` — extra margin (metres) added around the mesh bounds so the
    ///   field has solid values surrounding the cavity.
    pub fn from_mesh(mesh: &TriMesh, dx: f64, padding: f64) -> Sdf {
        let bounds = mesh.bounds().padded(padding);
        Self::from_mesh_in_bounds(mesh, dx, bounds)
    }

    /// Build a signed distance field over an explicit world-space box.
    pub fn from_mesh_in_bounds(mesh: &TriMesh, dx: f64, bounds: Aabb) -> Sdf {
        let size = bounds.size();
        let nx = (size.x / dx).ceil() as usize + 1;
        let ny = (size.y / dx).ceil() as usize + 1;
        let nz = (size.z / dx).ceil() as usize + 1;
        let origin = bounds.min;

        // Pre-extract triangle vertices for cache-friendly iteration.
        let tris: Vec<[Vec3; 3]> = (0..mesh.triangle_count()).map(|t| mesh.triangle(t)).collect();

        let mut data = vec![0.0f64; nx * ny * nz];
        // Parallelise over z-slices: each slice is independent.
        data.par_chunks_mut(nx * ny)
            .enumerate()
            .for_each(|(k, slice)| {
                for j in 0..ny {
                    for i in 0..nx {
                        let p = origin
                            + Vec3::new(i as f64 * dx, j as f64 * dx, k as f64 * dx);
                        slice[j * nx + i] = signed_distance_to_mesh(p, &tris);
                    }
                }
            });

        Sdf { origin, dx, nx, ny, nz, data }
    }

    /// Construct from an analytic signed-distance function (used in tests and
    /// for simple primitive domains).
    pub fn from_fn(origin: Vec3, dx: f64, dims: (usize, usize, usize), f: impl Fn(Vec3) -> f64 + Sync) -> Sdf {
        let (nx, ny, nz) = dims;
        let mut data = vec![0.0; nx * ny * nz];
        data.par_chunks_mut(nx * ny).enumerate().for_each(|(k, slice)| {
            for j in 0..ny {
                for i in 0..nx {
                    let p = origin + Vec3::new(i as f64 * dx, j as f64 * dx, k as f64 * dx);
                    slice[j * nx + i] = f(p);
                }
            }
        });
        Sdf { origin, dx, nx, ny, nz, data }
    }

    /// World-space bounds covered by the field's nodes.
    pub fn bounds(&self) -> Aabb {
        Aabb::new(
            self.origin,
            self.origin
                + Vec3::new(
                    (self.nx - 1) as f64 * self.dx,
                    (self.ny - 1) as f64 * self.dx,
                    (self.nz - 1) as f64 * self.dx,
                ),
        )
    }

    /// Trilinearly interpolated signed distance at world point `p`.
    /// Points outside the grid clamp to the boundary nodes (which are solid,
    /// i.e. positive, by construction with padding).
    pub fn sample(&self, p: Vec3) -> f64 {
        let l = (p - self.origin) / self.dx;
        let clampi = |v: f64, n: usize| -> (usize, f64) {
            if v <= 0.0 {
                (0, 0.0)
            } else if v >= (n - 1) as f64 {
                (n - 2, 1.0)
            } else {
                let i = v.floor();
                (i as usize, v - i)
            }
        };
        let (i, fx) = clampi(l.x, self.nx);
        let (j, fy) = clampi(l.y, self.ny);
        let (k, fz) = clampi(l.z, self.nz);

        let c000 = self.data[self.idx(i, j, k)];
        let c100 = self.data[self.idx(i + 1, j, k)];
        let c010 = self.data[self.idx(i, j + 1, k)];
        let c110 = self.data[self.idx(i + 1, j + 1, k)];
        let c001 = self.data[self.idx(i, j, k + 1)];
        let c101 = self.data[self.idx(i + 1, j, k + 1)];
        let c011 = self.data[self.idx(i, j + 1, k + 1)];
        let c111 = self.data[self.idx(i + 1, j + 1, k + 1)];

        let c00 = c000 * (1.0 - fx) + c100 * fx;
        let c10 = c010 * (1.0 - fx) + c110 * fx;
        let c01 = c001 * (1.0 - fx) + c101 * fx;
        let c11 = c011 * (1.0 - fx) + c111 * fx;
        let c0 = c00 * (1.0 - fy) + c10 * fy;
        let c1 = c01 * (1.0 - fy) + c11 * fy;
        c0 * (1.0 - fz) + c1 * fz
    }

    /// Outward-pointing (towards solid) gradient at `p`, via central
    /// differences. Roughly unit length away from the surface.
    pub fn gradient(&self, p: Vec3) -> Vec3 {
        let h = self.dx;
        let dx = self.sample(p + Vec3::new(h, 0.0, 0.0)) - self.sample(p - Vec3::new(h, 0.0, 0.0));
        let dy = self.sample(p + Vec3::new(0.0, h, 0.0)) - self.sample(p - Vec3::new(0.0, h, 0.0));
        let dz = self.sample(p + Vec3::new(0.0, 0.0, h)) - self.sample(p - Vec3::new(0.0, 0.0, h));
        Vec3::new(dx, dy, dz) / (2.0 * h)
    }

    /// True if `p` is inside the cavity (fluid/air can be here).
    #[inline]
    pub fn is_inside_cavity(&self, p: Vec3) -> bool {
        self.sample(p) < 0.0
    }
}

/// Signed distance from `p` to the closed mesh given as triangles.
/// Negative inside, positive outside (assuming outward CCW winding).
fn signed_distance_to_mesh(p: Vec3, tris: &[[Vec3; 3]]) -> f64 {
    let mut min_d2 = f64::INFINITY;
    let mut winding = 0.0; // accumulated solid angle / (4*pi)
    for tri in tris {
        let d2 = point_triangle_distance_sq(p, tri[0], tri[1], tri[2]);
        if d2 < min_d2 {
            min_d2 = d2;
        }
        winding += solid_angle(p, tri[0], tri[1], tri[2]);
    }
    winding /= 4.0 * std::f64::consts::PI;
    let dist = min_d2.sqrt();
    // winding ~ 1 inside, ~ 0 outside; inside the cavity => negative.
    if winding > 0.5 {
        -dist
    } else {
        dist
    }
}

/// Squared distance from point `p` to triangle `(a,b,c)`.
/// Christer Ericson, *Real-Time Collision Detection*, closest-point routine.
pub fn point_triangle_distance_sq(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> f64 {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return ap.length_squared(); // vertex region A
    }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return bp.length_squared(); // vertex region B
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        let closest = a + ab * v;
        return (p - closest).length_squared(); // edge AB
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return cp.length_squared(); // vertex region C
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        let closest = a + ac * w;
        return (p - closest).length_squared(); // edge AC
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        let closest = b + (c - b) * w;
        return (p - closest).length_squared(); // edge BC
    }
    // Inside the face: project onto the plane.
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    let closest = a + ab * v + ac * w;
    (p - closest).length_squared()
}

/// Signed solid angle subtended by triangle `(a,b,c)` at point `p`
/// (van Oosterom & Strackee). Summed over a closed mesh this gives `4*pi`
/// times the generalized winding number.
fn solid_angle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> f64 {
    let av = a - p;
    let bv = b - p;
    let cv = c - p;
    let la = av.length();
    let lb = bv.length();
    let lc = cv.length();
    if la < 1e-12 || lb < 1e-12 || lc < 1e-12 {
        return 0.0;
    }
    let numerator = av.dot(bv.cross(cv));
    let denominator =
        la * lb * lc + av.dot(bv) * lc + bv.dot(cv) * la + cv.dot(av) * lb;
    2.0 * numerator.atan2(denominator)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn sphere_sdf_matches_analytic() {
        // Analytic sphere of radius 1 centred at origin; inside negative.
        let r = 1.0;
        let n = 21;
        let dx = 0.2;
        let origin = Vec3::splat(-2.0);
        let sdf = Sdf::from_fn(origin, dx, (n, n, n), |p| p.length() - r);
        // Sample near a few points.
        assert_relative_eq!(sdf.sample(Vec3::ZERO), -1.0, epsilon = 1e-9);
        assert_relative_eq!(sdf.sample(Vec3::new(1.5, 0.0, 0.0)), 0.5, epsilon = 1e-9);
        // Gradient points radially outward.
        let g = sdf.gradient(Vec3::new(1.5, 0.0, 0.0));
        assert!(g.x > 0.9 && g.x < 1.1);
        assert!(sdf.is_inside_cavity(Vec3::ZERO));
        assert!(!sdf.is_inside_cavity(Vec3::new(2.0, 0.0, 0.0)));
    }

    #[test]
    fn winding_sign_for_box() {
        // Build SDF from a box mesh; a point inside should be negative.
        let v = vec![
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(-1.0, -1.0, 1.0),
            Vec3::new(1.0, -1.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(-1.0, 1.0, 1.0),
        ];
        let f = vec![
            [0u32, 3, 2],
            [0, 2, 1],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ];
        let mesh = TriMesh::new(v, f);
        let tris: Vec<[Vec3; 3]> =
            (0..mesh.triangle_count()).map(|t| mesh.triangle(t)).collect();
        // Centre is inside: distance to nearest wall is 1, sign negative.
        let d = signed_distance_to_mesh(Vec3::ZERO, &tris);
        assert_relative_eq!(d, -1.0, epsilon = 1e-9);
        // A point well outside is positive.
        let d_out = signed_distance_to_mesh(Vec3::new(3.0, 0.0, 0.0), &tris);
        assert!(d_out > 0.0);
        assert_relative_eq!(d_out, 2.0, epsilon = 1e-9);
    }

    #[test]
    fn point_triangle_distance_basic() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(1.0, 0.0, 0.0);
        let c = Vec3::new(0.0, 1.0, 0.0);
        // Point directly above the interior.
        let d2 = point_triangle_distance_sq(Vec3::new(0.25, 0.25, 2.0), a, b, c);
        assert_relative_eq!(d2, 4.0, epsilon = 1e-12);
        // Point beyond vertex A.
        let d2v = point_triangle_distance_sq(Vec3::new(-1.0, -1.0, 0.0), a, b, c);
        assert_relative_eq!(d2v, 2.0, epsilon = 1e-12);
    }
}
