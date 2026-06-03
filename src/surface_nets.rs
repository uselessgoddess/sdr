//! Polygonisation of implicit scalar fields via **Naive Surface Nets**.
//!
//! Given a function `f: Vec3 -> f64` whose zero level set is the surface
//! (`f < 0` inside the object, `f > 0` outside), this produces a watertight
//! triangle mesh with outward-facing winding. Surface Nets places one vertex
//! per straddling cell at the average of its edge crossings and connects
//! neighbouring cell vertices into quads. It is far smaller than a full
//! Marching Cubes table and yields smooth, organic surfaces — well suited to
//! the soft anatomy of a sinus cavity.

use crate::math::Vec3;
use crate::mesh::TriMesh;
use rayon::prelude::*;

/// Corner offsets of a unit cube, indexed so that bit0=x, bit1=y, bit2=z.
const CORNERS: [(usize, usize, usize); 8] = [
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (1, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (0, 1, 1),
    (1, 1, 1),
];

/// The 12 edges of the cube as pairs of corner indices.
const EDGES: [(usize, usize); 12] = [
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7), // along x
    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7), // along y
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7), // along z
];

/// Polygonise the zero level set of `field` over a regular grid.
///
/// * `origin` — world position of node (0,0,0).
/// * `dx` — node spacing (metres).
/// * `dims` — number of *sample nodes* along each axis (cells = nodes − 1).
pub fn surface_nets(
    origin: Vec3,
    dx: f64,
    dims: (usize, usize, usize),
    field: impl Fn(Vec3) -> f64 + Sync,
) -> TriMesh {
    let (nx, ny, nz) = dims;
    assert!(nx >= 2 && ny >= 2 && nz >= 2, "surface_nets needs at least 2 nodes per axis");

    // 1. Sample the field at every node (parallel over z-slices).
    let mut values = vec![0.0f64; nx * ny * nz];
    values.par_chunks_mut(nx * ny).enumerate().for_each(|(k, slice)| {
        for j in 0..ny {
            for i in 0..nx {
                let p = origin + Vec3::new(i as f64 * dx, j as f64 * dx, k as f64 * dx);
                slice[j * nx + i] = field(p);
            }
        }
    });
    let node = |i: usize, j: usize, k: usize| values[(k * ny + j) * nx + i];

    // 2. One vertex per straddling cell.
    let (cx, cy, cz) = (nx - 1, ny - 1, nz - 1);
    let mut cell_vertex = vec![u32::MAX; cx * cy * cz];
    let cell_idx = |i: usize, j: usize, k: usize| (k * cy + j) * cx + i;
    let mut vertices: Vec<Vec3> = Vec::new();

    for k in 0..cz {
        for j in 0..cy {
            for i in 0..cx {
                let mut vals = [0.0f64; 8];
                for (c, &(oi, oj, ok)) in CORNERS.iter().enumerate() {
                    vals[c] = node(i + oi, j + oj, k + ok);
                }
                // Skip cells entirely inside or outside.
                let mut crossings = Vec::new();
                for &(a, b) in &EDGES {
                    let (va, vb) = (vals[a], vals[b]);
                    if (va < 0.0) != (vb < 0.0) {
                        let t = va / (va - vb); // crossing parameter in [0,1]
                        let (ai, aj, ak) = CORNERS[a];
                        let (bi, bj, bk) = CORNERS[b];
                        let pa = Vec3::new(ai as f64, aj as f64, ak as f64);
                        let pb = Vec3::new(bi as f64, bj as f64, bk as f64);
                        crossings.push(pa + (pb - pa) * t);
                    }
                }
                if crossings.is_empty() {
                    continue;
                }
                let mut sum = Vec3::ZERO;
                for c in &crossings {
                    sum += *c;
                }
                let local = sum / crossings.len() as f64;
                let world = origin
                    + Vec3::new((i as f64 + local.x) * dx, (j as f64 + local.y) * dx, (k as f64 + local.z) * dx);
                cell_vertex[cell_idx(i, j, k)] = vertices.len() as u32;
                vertices.push(world);
            }
        }
    }

    // 3. Connect cell vertices into quads across every straddling grid edge.
    let mut indices: Vec<[u32; 3]> = Vec::new();
    let get = |i: usize, j: usize, k: usize| -> Option<u32> {
        let v = cell_vertex[cell_idx(i, j, k)];
        if v == u32::MAX {
            None
        } else {
            Some(v)
        }
    };

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let v0 = node(i, j, k);
                // x-edge to (i+1,j,k); shared by cells with x-min=i, y in {j-1,j}, z in {k-1,k}.
                if i + 1 < nx && j >= 1 && k >= 1 {
                    let v1 = node(i + 1, j, k);
                    if (v0 < 0.0) != (v1 < 0.0) {
                        let ring = [
                            get(i, j - 1, k - 1),
                            get(i, j, k - 1),
                            get(i, j, k),
                            get(i, j - 1, k),
                        ];
                        let desired = if v0 < v1 { Vec3::new(1.0, 0.0, 0.0) } else { Vec3::new(-1.0, 0.0, 0.0) };
                        emit_quad(&ring, desired, &vertices, &mut indices);
                    }
                }
                // y-edge to (i,j+1,k); shared by cells with y-min=j, x in {i-1,i}, z in {k-1,k}.
                if j + 1 < ny && i >= 1 && k >= 1 {
                    let v1 = node(i, j + 1, k);
                    if (v0 < 0.0) != (v1 < 0.0) {
                        let ring = [
                            get(i - 1, j, k - 1),
                            get(i, j, k - 1),
                            get(i, j, k),
                            get(i - 1, j, k),
                        ];
                        let desired = if v0 < v1 { Vec3::new(0.0, 1.0, 0.0) } else { Vec3::new(0.0, -1.0, 0.0) };
                        emit_quad(&ring, desired, &vertices, &mut indices);
                    }
                }
                // z-edge to (i,j,k+1); shared by cells with z-min=k, x in {i-1,i}, y in {j-1,j}.
                if k + 1 < nz && i >= 1 && j >= 1 {
                    let v1 = node(i, j, k + 1);
                    if (v0 < 0.0) != (v1 < 0.0) {
                        let ring = [
                            get(i - 1, j - 1, k),
                            get(i, j - 1, k),
                            get(i, j, k),
                            get(i - 1, j, k),
                        ];
                        let desired = if v0 < v1 { Vec3::new(0.0, 0.0, 1.0) } else { Vec3::new(0.0, 0.0, -1.0) };
                        emit_quad(&ring, desired, &vertices, &mut indices);
                    }
                }
            }
        }
    }

    TriMesh::new(vertices, indices)
}

/// Emit two triangles for a quad given as a 4-vertex ring, orienting them so
/// the geometric normal agrees with `desired`.
fn emit_quad(ring: &[Option<u32>; 4], desired: Vec3, vertices: &[Vec3], indices: &mut Vec<[u32; 3]>) {
    // All four cells must have produced a vertex; on a closed surface they do.
    let (a, b, c, d) = match (ring[0], ring[1], ring[2], ring[3]) {
        (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
        _ => return,
    };
    let pa = vertices[a as usize];
    let pb = vertices[b as usize];
    let pc = vertices[c as usize];
    let n = (pb - pa).cross(pc - pa);
    if n.dot(desired) >= 0.0 {
        indices.push([a, b, c]);
        indices.push([a, c, d]);
    } else {
        indices.push([a, c, b]);
        indices.push([a, d, c]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn meshes_a_sphere() {
        let r = 1.0;
        let dx = 0.1;
        let n = 41; // covers [-2, 2]
        let origin = Vec3::splat(-2.0);
        let mesh = surface_nets(origin, dx, (n, n, n), |p| p.length() - r);
        assert!(!mesh.is_empty());
        // Outward winding => positive signed volume close to (4/3)pi r^3.
        let vol = mesh.signed_volume();
        let exact = 4.0 / 3.0 * std::f64::consts::PI * r.powi(3);
        assert!(vol > 0.0, "winding should be outward (positive volume), got {vol}");
        assert_relative_eq!(vol, exact, epsilon = 0.15);
        // Surface area close to 4 pi r^2.
        let area = mesh.surface_area();
        let exact_area = 4.0 * std::f64::consts::PI * r * r;
        assert_relative_eq!(area, exact_area, epsilon = 1.2);
    }
}
