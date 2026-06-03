//! Writing simulation output in formats the rest of the pipeline understands.
//!
//! The original workflow is *solver → [splashsurf] → Blender*. `splashsurf`
//! reconstructs a smooth surface mesh from a particle point cloud, so the solver
//! must emit particle positions in a format it reads. We support two:
//!
//! * **Legacy VTK** (`.vtk`, ASCII `UNSTRUCTURED_GRID` of vertices) — readable
//!   by `splashsurf`, ParaView and Blender's VTK importers, and easy to inspect
//!   by hand. This is the default.
//! * **XYZ** (`.xyz`, raw little-endian `f32` triples, no header) — the compact
//!   binary point format `splashsurf` documents; handy for large clouds.
//!
//! [splashsurf]: https://github.com/InteractiveComputerGraphics/splashsurf

use crate::math::Vec3;
use anyhow::{Context, Result};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Write particle positions as an ASCII legacy-VTK unstructured grid of
/// `VTK_VERTEX` cells. This is the most broadly compatible point-cloud format.
pub fn write_points_vtk(path: impl AsRef<Path>, positions: &[Vec3]) -> Result<()> {
    let path = path.as_ref();
    let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut w = BufWriter::new(file);
    let n = positions.len();

    writeln!(w, "# vtk DataFile Version 4.2")?;
    writeln!(w, "sdr irrigation particles")?;
    writeln!(w, "ASCII")?;
    writeln!(w, "DATASET UNSTRUCTURED_GRID")?;
    writeln!(w, "POINTS {n} double")?;
    for p in positions {
        writeln!(w, "{} {} {}", p.x, p.y, p.z)?;
    }
    // One VTK_VERTEX (type 1) cell per point: "1 <index>".
    writeln!(w, "CELLS {n} {}", n * 2)?;
    for i in 0..n {
        writeln!(w, "1 {i}")?;
    }
    writeln!(w, "CELL_TYPES {n}")?;
    for _ in 0..n {
        writeln!(w, "1")?;
    }
    w.flush()?;
    Ok(())
}

/// Write particle positions as raw little-endian `f32` triples (`.xyz`), the
/// compact binary point cloud `splashsurf` reads with `--particle-radius`.
pub fn write_points_xyz(path: impl AsRef<Path>, positions: &[Vec3]) -> Result<()> {
    let path = path.as_ref();
    let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut w = BufWriter::new(file);
    let mut buf = [0u8; 12];
    for p in positions {
        buf[0..4].copy_from_slice(&(p.x as f32).to_le_bytes());
        buf[4..8].copy_from_slice(&(p.y as f32).to_le_bytes());
        buf[8..12].copy_from_slice(&(p.z as f32).to_le_bytes());
        w.write_all(&buf)?;
    }
    w.flush()?;
    Ok(())
}

/// Format a zero-padded frame filename like `prefix_000123.ext`.
pub fn frame_path(dir: impl AsRef<Path>, prefix: &str, frame: usize, ext: &str) -> std::path::PathBuf {
    dir.as_ref().join(format!("{prefix}_{frame:06}.{ext}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip_dir() -> std::path::PathBuf {
        std::env::temp_dir().join("sdr_output_test")
    }

    #[test]
    fn vtk_has_expected_structure() {
        let dir = roundtrip_dir();
        std::fs::create_dir_all(&dir).unwrap();
        let pts = vec![Vec3::new(0.0, 1.0, 2.0), Vec3::new(-1.0, 0.5, 3.0)];
        let path = dir.join("pts.vtk");
        write_points_vtk(&path, &pts).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("DATASET UNSTRUCTURED_GRID"));
        assert!(text.contains("POINTS 2 double"));
        assert!(text.contains("0 1 2"));
        assert!(text.contains("CELL_TYPES 2"));
    }

    #[test]
    fn xyz_is_binary_f32_triples() {
        let dir = roundtrip_dir();
        std::fs::create_dir_all(&dir).unwrap();
        let pts = vec![Vec3::new(0.0, 1.0, 2.0), Vec3::new(3.0, 4.0, 5.0)];
        let path = dir.join("pts.xyz");
        write_points_xyz(&path, &pts).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes.len(), 2 * 12); // 2 points * 3 f32 * 4 bytes
        let x0 = f32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let y1 = f32::from_le_bytes(bytes[16..20].try_into().unwrap());
        assert_eq!(x0, 0.0);
        assert_eq!(y1, 4.0);
    }

    #[test]
    fn frame_path_is_zero_padded() {
        let p = frame_path("/tmp/out", "particles", 42, "vtk");
        assert!(p.to_string_lossy().ends_with("particles_000042.vtk"));
    }
}
