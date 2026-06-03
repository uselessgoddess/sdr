//! Triangle mesh representation and import/export.
//!
//! Supports the two formats that show up in the DICOM → Blender → solver
//! pipeline described in the thesis: Wavefront **OBJ** (text) and binary
//! **STL**. Both are read and written. The sinus cavity the doctor prepares in
//! Blender can be handed to the solver as either format.

use crate::math::{Aabb, Vec3};
use anyhow::{bail, Context, Result};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;

/// An indexed triangle mesh.
#[derive(Debug, Clone, Default)]
pub struct TriMesh {
    pub vertices: Vec<Vec3>,
    /// Triangle vertex indices, three per face.
    pub indices: Vec<[u32; 3]>,
}

impl TriMesh {
    pub fn new(vertices: Vec<Vec3>, indices: Vec<[u32; 3]>) -> Self {
        TriMesh { vertices, indices }
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// The three vertices of triangle `i`.
    #[inline]
    pub fn triangle(&self, i: usize) -> [Vec3; 3] {
        let [a, b, c] = self.indices[i];
        [
            self.vertices[a as usize],
            self.vertices[b as usize],
            self.vertices[c as usize],
        ]
    }

    /// Axis-aligned bounding box of all vertices.
    pub fn bounds(&self) -> Aabb {
        Aabb::from_points(&self.vertices)
    }

    /// Total surface area (sum of triangle areas).
    pub fn surface_area(&self) -> f64 {
        (0..self.triangle_count())
            .map(|i| {
                let [a, b, c] = self.triangle(i);
                0.5 * (b - a).cross(c - a).length()
            })
            .sum()
    }

    /// Signed volume via the divergence theorem (sum of signed tetrahedra to
    /// the origin). Positive for an outward-facing (CCW) closed mesh.
    pub fn signed_volume(&self) -> f64 {
        let mut v = 0.0;
        for i in 0..self.triangle_count() {
            let [a, b, c] = self.triangle(i);
            v += a.dot(b.cross(c)) / 6.0;
        }
        v
    }

    /// Translate every vertex by `t`.
    pub fn translate(&mut self, t: Vec3) {
        for v in &mut self.vertices {
            *v += t;
        }
    }

    /// Scale every vertex about the origin by `s`.
    pub fn scale(&mut self, s: f64) {
        for v in &mut self.vertices {
            *v = *v * s;
        }
    }

    /// Per-vertex normals, area-weighted from incident faces.
    pub fn vertex_normals(&self) -> Vec<Vec3> {
        let mut normals = vec![Vec3::ZERO; self.vertices.len()];
        for i in 0..self.triangle_count() {
            let [ia, ib, ic] = self.indices[i];
            let [a, b, c] = self.triangle(i);
            // Cross product magnitude is proportional to area => area weighting.
            let n = (b - a).cross(c - a);
            normals[ia as usize] += n;
            normals[ib as usize] += n;
            normals[ic as usize] += n;
        }
        for n in &mut normals {
            *n = n.normalize_or_zero();
        }
        normals
    }

    /// Load a mesh from a file, dispatching on extension (`.obj` / `.stl`).
    pub fn load(path: impl AsRef<Path>) -> Result<TriMesh> {
        let path = path.as_ref();
        match path.extension().and_then(|e| e.to_str()).map(str::to_lowercase) {
            Some(ext) if ext == "obj" => Self::load_obj(path),
            Some(ext) if ext == "stl" => Self::load_stl(path),
            other => bail!("unsupported mesh extension: {:?} (use .obj or .stl)", other),
        }
    }

    /// Save a mesh to a file, dispatching on extension (`.obj` / `.stl`).
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        match path.extension().and_then(|e| e.to_str()).map(str::to_lowercase) {
            Some(ext) if ext == "obj" => self.save_obj(path),
            Some(ext) if ext == "stl" => self.save_stl(path),
            other => bail!("unsupported mesh extension: {:?} (use .obj or .stl)", other),
        }
    }

    /// Parse a Wavefront OBJ. Handles `v`/`f`, the `v/vt/vn` slash syntax, and
    /// triangulates convex polygons (fans) with more than three vertices.
    pub fn load_obj(path: impl AsRef<Path>) -> Result<TriMesh> {
        let path = path.as_ref();
        let file = File::open(path).with_context(|| format!("opening OBJ {}", path.display()))?;
        let reader = BufReader::new(file);

        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        for line in reader.lines() {
            let line = line?;
            let line = line.trim();
            let mut tok = line.split_whitespace();
            match tok.next() {
                Some("v") => {
                    let coords: Vec<f64> = tok.filter_map(|t| t.parse().ok()).collect();
                    if coords.len() < 3 {
                        bail!("malformed vertex line: {line:?}");
                    }
                    vertices.push(Vec3::new(coords[0], coords[1], coords[2]));
                }
                Some("f") => {
                    // Each token may be "i", "i/j" or "i/j/k"; we only want i.
                    // OBJ indices are 1-based and may be negative (relative).
                    let face: Vec<i64> = tok
                        .map(|t| {
                            t.split('/')
                                .next()
                                .unwrap_or("")
                                .parse::<i64>()
                                .map_err(|_| anyhow::anyhow!("bad face index in {line:?}"))
                        })
                        .collect::<Result<_>>()?;
                    if face.len() < 3 {
                        bail!("face with fewer than 3 vertices: {line:?}");
                    }
                    let resolve = |idx: i64| -> u32 {
                        if idx > 0 {
                            (idx - 1) as u32
                        } else {
                            // Negative indices count back from the end.
                            (vertices.len() as i64 + idx) as u32
                        }
                    };
                    // Fan triangulation: (0, k, k+1).
                    for k in 1..face.len() - 1 {
                        indices.push([resolve(face[0]), resolve(face[k]), resolve(face[k + 1])]);
                    }
                }
                _ => {} // ignore comments, normals, texcoords, groups, etc.
            }
        }
        Ok(TriMesh::new(vertices, indices))
    }

    /// Write a Wavefront OBJ.
    pub fn save_obj(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let file = File::create(path).with_context(|| format!("creating OBJ {}", path.display()))?;
        let mut w = BufWriter::new(file);
        writeln!(w, "# sdr parametric sinus mesh")?;
        writeln!(w, "# vertices: {} faces: {}", self.vertices.len(), self.indices.len())?;
        for v in &self.vertices {
            writeln!(w, "v {} {} {}", v.x, v.y, v.z)?;
        }
        for f in &self.indices {
            // OBJ is 1-based.
            writeln!(w, "f {} {} {}", f[0] + 1, f[1] + 1, f[2] + 1)?;
        }
        Ok(())
    }

    /// Load a binary STL. (ASCII STL is uncommon from Blender exports; we
    /// detect and reject it with a clear message.)
    pub fn load_stl(path: impl AsRef<Path>) -> Result<TriMesh> {
        let path = path.as_ref();
        let mut file = File::open(path).with_context(|| format!("opening STL {}", path.display()))?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)?;
        if buf.len() < 84 {
            bail!("STL file too small to be valid binary STL");
        }
        if buf.starts_with(b"solid") && !looks_like_binary_stl(&buf) {
            bail!("ASCII STL is not supported; re-export as binary STL");
        }
        let tri_count = u32::from_le_bytes([buf[80], buf[81], buf[82], buf[83]]) as usize;
        let expected = 84 + tri_count * 50;
        if buf.len() < expected {
            bail!("binary STL truncated: expected {expected} bytes, got {}", buf.len());
        }
        let mut vertices = Vec::with_capacity(tri_count * 3);
        let mut indices = Vec::with_capacity(tri_count);
        let read_f32 = |b: &[u8], o: usize| {
            f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]) as f64
        };
        for t in 0..tri_count {
            let base = 84 + t * 50;
            // Skip the 12-byte normal; recompute from geometry when needed.
            let mut tri = [0u32; 3];
            for (j, slot) in tri.iter_mut().enumerate() {
                let o = base + 12 + j * 12;
                let v = Vec3::new(read_f32(&buf, o), read_f32(&buf, o + 4), read_f32(&buf, o + 8));
                *slot = vertices.len() as u32;
                vertices.push(v);
            }
            indices.push(tri);
        }
        Ok(TriMesh::new(vertices, indices))
    }

    /// Write a binary STL.
    pub fn save_stl(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let file = File::create(path).with_context(|| format!("creating STL {}", path.display()))?;
        let mut w = BufWriter::new(file);
        let header = [0u8; 80];
        w.write_all(&header)?;
        w.write_all(&(self.triangle_count() as u32).to_le_bytes())?;
        for i in 0..self.triangle_count() {
            let [a, b, c] = self.triangle(i);
            let n = (b - a).cross(c - a).normalize_or_zero();
            for comp in [n.x, n.y, n.z] {
                w.write_all(&(comp as f32).to_le_bytes())?;
            }
            for v in [a, b, c] {
                for comp in [v.x, v.y, v.z] {
                    w.write_all(&(comp as f32).to_le_bytes())?;
                }
            }
            w.write_all(&0u16.to_le_bytes())?; // attribute byte count
        }
        Ok(())
    }
}

/// Heuristic: a file that starts with "solid" but whose declared triangle
/// count matches its byte length is really binary (some exporters write
/// "solid" into the binary header).
fn looks_like_binary_stl(buf: &[u8]) -> bool {
    if buf.len() < 84 {
        return false;
    }
    let tri_count = u32::from_le_bytes([buf[80], buf[81], buf[82], buf[83]]) as usize;
    buf.len() == 84 + tri_count * 50
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// A unit cube centred at the origin, side 2 (from -1 to 1), CCW outward.
    fn unit_box() -> TriMesh {
        // 8 corners.
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
        // 12 triangles, outward winding.
        let f = vec![
            [0, 3, 2],
            [0, 2, 1], // -z
            [4, 5, 6],
            [4, 6, 7], // +z
            [0, 1, 5],
            [0, 5, 4], // -y
            [3, 7, 6],
            [3, 6, 2], // +y
            [0, 4, 7],
            [0, 7, 3], // -x
            [1, 2, 6],
            [1, 6, 5], // +x
        ];
        TriMesh::new(v, f)
    }

    #[test]
    fn box_area_and_volume() {
        let m = unit_box();
        // Side 2 cube: surface area = 6 * (2*2) = 24, volume = 2^3 = 8.
        assert_relative_eq!(m.surface_area(), 24.0, epsilon = 1e-9);
        assert_relative_eq!(m.signed_volume(), 8.0, epsilon = 1e-9);
    }

    #[test]
    fn obj_roundtrip() {
        let m = unit_box();
        let dir = std::env::temp_dir().join("sdr_obj_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("box.obj");
        m.save_obj(&path).unwrap();
        let m2 = TriMesh::load_obj(&path).unwrap();
        assert_eq!(m2.vertices.len(), m.vertices.len());
        assert_eq!(m2.indices.len(), m.indices.len());
        assert_relative_eq!(m2.signed_volume(), 8.0, epsilon = 1e-6);
    }

    #[test]
    fn stl_roundtrip() {
        let m = unit_box();
        let dir = std::env::temp_dir().join("sdr_stl_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("box.stl");
        m.save_stl(&path).unwrap();
        let m2 = TriMesh::load_stl(&path).unwrap();
        assert_eq!(m2.triangle_count(), m.triangle_count());
        assert_relative_eq!(m2.surface_area(), 24.0, epsilon = 1e-4);
    }
}
