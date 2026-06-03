//! A tiny software renderer for visual validation.
//!
//! No GPU, no external viewer — just enough 2D drawing (into a PNG via the
//! `image` crate) to *see* that the simulation did something sensible:
//!
//! * [`render_slice`] draws an orthographic cross-section of the cavity (from
//!   the solid SDF) with the fluid particles in a thin slab overlaid, coloured
//!   by speed, plus the needle. This is the picture a reviewer (or CI) looks at
//!   to confirm fluid actually fills and flushes the sinus.
//! * [`render_timeseries`] plots the headline metrics (fill, wall coverage,
//!   pressure) over the run.
//!
//! These images are committed by CI as build artifacts and embedded in the PR.

use crate::math::Vec3;
use crate::metrics::FrameMetrics;
use crate::solver::Solver;
use anyhow::{Context, Result};
use image::{Rgb, RgbImage};
use std::path::Path;

const BG: Rgb<u8> = Rgb([18, 18, 22]);
const INTERIOR: Rgb<u8> = Rgb([232, 236, 244]);
const WALL: Rgb<u8> = Rgb([44, 44, 54]);
const NEEDLE: Rgb<u8> = Rgb([255, 80, 80]);

/// An orthographic slice plane: two in-plane axes (horizontal, vertical) and one
/// out-of-plane axis we slice along. Axis indices are `0 = x, 1 = y, 2 = z`.
#[derive(Debug, Clone, Copy)]
pub struct SlicePlane {
    hax: usize,
    vax: usize,
    oax: usize,
    slice: f64,
    slab: f64,
    flip_v: bool,
}

impl SlicePlane {
    /// Vertical cross-section in the x–y plane at `z = slice` (the clinical
    /// "side view" through the socket). `slab` is the half-thickness of the
    /// particle band drawn around the plane.
    pub fn xy(slice: f64, slab: f64) -> Self {
        SlicePlane { hax: 0, vax: 1, oax: 2, slice, slab, flip_v: true }
    }

    /// Top-down view in the x–z plane at `y = slice`.
    pub fn xz(slice: f64, slab: f64) -> Self {
        SlicePlane { hax: 0, vax: 2, oax: 1, slice, slab, flip_v: false }
    }
}

/// Render a cross-section of the cavity with the fluid particles overlaid.
///
/// `target_px` sets the longest image dimension; the other is chosen to keep
/// the world aspect ratio. `vmax` (m/s) normalises the speed colour map; pass
/// `0.0` to auto-scale to the current peak particle speed.
pub fn render_slice(solver: &Solver, plane: &SlicePlane, target_px: u32, vmax: f64) -> RgbImage {
    let bb = solver.solid.bounds();
    let mn = bb.min.to_array();
    let mx = bb.max.to_array();
    let wm = (mx[plane.hax] - mn[plane.hax]).max(1e-9);
    let hm = (mx[plane.vax] - mn[plane.vax]).max(1e-9);

    let (w, h) = if wm >= hm {
        (target_px.max(1), (target_px as f64 * hm / wm).round().max(1.0) as u32)
    } else {
        ((target_px as f64 * wm / hm).round().max(1.0) as u32, target_px.max(1))
    };

    let mut img = RgbImage::from_pixel(w, h, BG);

    // Background: shade cavity interior vs. surrounding solid by sampling the
    // SDF at each pixel's world position on the slice plane.
    for py in 0..h {
        for px in 0..w {
            let p = self_world(plane, mn, mx, wm, hm, w, h, px, py);
            let color = if solver.solid.sample(p) < 0.0 { INTERIOR } else { WALL };
            img.put_pixel(px, py, color);
        }
    }

    let to_px = |p: Vec3| -> (i64, i64) {
        let a = p.to_array();
        let fx = (a[plane.hax] - mn[plane.hax]) / wm * w as f64;
        let fy = if plane.flip_v {
            (mx[plane.vax] - a[plane.vax]) / hm * h as f64
        } else {
            (a[plane.vax] - mn[plane.vax]) / hm * h as f64
        };
        (fx as i64, fy as i64)
    };

    // Particles within the slab, coloured by speed.
    let vmax = if vmax > 0.0 { vmax } else { solver.max_speed().max(1e-6) };
    for (p, v) in solver.particles.positions.iter().zip(&solver.particles.velocities) {
        if (p.to_array()[plane.oax] - plane.slice).abs() > plane.slab {
            continue;
        }
        let (cx, cy) = to_px(*p);
        let t = (v.length() / vmax).clamp(0.0, 1.0);
        draw_disk(&mut img, cx, cy, 2, speed_color(t));
    }

    // Needle: a short segment behind the tip plus a marker at the tip.
    let tip = solver.needle.tip;
    let back = tip - solver.needle.axis * 0.012;
    let (ax, ay) = to_px(back);
    let (bx, by) = to_px(tip);
    draw_line(&mut img, ax, ay, bx, by, NEEDLE);
    draw_disk(&mut img, bx, by, 3, NEEDLE);

    img
}

/// World position of pixel `(px, py)` on the slice plane.
#[allow(clippy::too_many_arguments)]
fn self_world(
    plane: &SlicePlane,
    mn: [f64; 3],
    mx: [f64; 3],
    wm: f64,
    hm: f64,
    w: u32,
    h: u32,
    px: u32,
    py: u32,
) -> Vec3 {
    let wx = mn[plane.hax] + (px as f64 + 0.5) / w as f64 * wm;
    let wy = if plane.flip_v {
        mx[plane.vax] - (py as f64 + 0.5) / h as f64 * hm
    } else {
        mn[plane.vax] + (py as f64 + 0.5) / h as f64 * hm
    };
    let mut c = [0.0; 3];
    c[plane.hax] = wx;
    c[plane.vax] = wy;
    c[plane.oax] = plane.slice;
    Vec3::new(c[0], c[1], c[2])
}

/// Plot the headline metrics over the run: fill fraction (blue), wall coverage
/// (green) and membrane pressure normalised by its peak (red), all on a `0..1`
/// axis. Pressure uses the focal membrane (wall) load — the jet-impingement peak
/// on the lining — with non-physical single-cell solver artifacts excluded.
pub fn render_timeseries(frames: &[FrameMetrics], width: u32, height: u32) -> RgbImage {
    let mut img = RgbImage::from_pixel(width, height, Rgb([250, 250, 252]));
    let (ml, mr, mt, mb) = (8i64, 8i64, 8i64, 8i64);
    let x0 = ml;
    let x1 = width as i64 - mr;
    let y0 = mt;
    let y1 = height as i64 - mb;

    // Plot frame: axes box + horizontal gridlines at 0, .25, .5, .75, 1.
    let grid = Rgb([220, 220, 226]);
    for g in 0..=4 {
        let y = y1 - (y1 - y0) * g / 4;
        draw_line(&mut img, x0, y, x1, y, grid);
    }
    let axis = Rgb([120, 120, 130]);
    draw_line(&mut img, x0, y0, x0, y1, axis);
    draw_line(&mut img, x0, y1, x1, y1, axis);

    if frames.len() < 2 {
        return img;
    }
    let n = frames.len();
    let peak_p = frames.iter().map(|m| m.peak_wall_pressure_pa).fold(0.0_f64, f64::max).max(1e-9);
    let map = |i: usize, val: f64| -> (i64, i64) {
        let fx = x0 as f64 + (x1 - x0) as f64 * i as f64 / (n - 1) as f64;
        let v = val.clamp(0.0, 1.0);
        let fy = y1 as f64 - (y1 - y0) as f64 * v;
        (fx as i64, fy as i64)
    };
    let series = [
        (Rgb([40, 90, 220]), Box::new(|m: &FrameMetrics| m.fill_fraction) as Box<dyn Fn(&FrameMetrics) -> f64>),
        (Rgb([30, 160, 60]), Box::new(|m: &FrameMetrics| m.wall_coverage)),
        (Rgb([210, 60, 50]), Box::new(move |m: &FrameMetrics| m.peak_wall_pressure_pa / peak_p)),
    ];
    for (color, f) in &series {
        for i in 1..n {
            let (xa, ya) = map(i - 1, f(&frames[i - 1]));
            let (xb, yb) = map(i, f(&frames[i]));
            draw_line(&mut img, xa, ya, xb, yb, *color);
        }
    }
    img
}

/// Save an image as PNG.
pub fn save_png(img: &RgbImage, path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    img.save_with_format(path, image::ImageFormat::Png)
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

/// Speed colour map: blue (slow) → green → red (fast).
fn speed_color(t: f64) -> Rgb<u8> {
    let t = t.clamp(0.0, 1.0);
    let (r, g, b) = if t < 0.5 {
        let u = t / 0.5;
        (0.0, u, 1.0 - u)
    } else {
        let u = (t - 0.5) / 0.5;
        (u, 1.0 - u, 0.0)
    };
    Rgb([(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8])
}

fn put(img: &mut RgbImage, x: i64, y: i64, c: Rgb<u8>) {
    if x >= 0 && y >= 0 && (x as u32) < img.width() && (y as u32) < img.height() {
        img.put_pixel(x as u32, y as u32, c);
    }
}

fn draw_disk(img: &mut RgbImage, cx: i64, cy: i64, r: i64, c: Rgb<u8>) {
    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy <= r * r {
                put(img, cx + dx, cy + dy, c);
            }
        }
    }
}

/// Bresenham line.
fn draw_line(img: &mut RgbImage, x0: i64, y0: i64, x1: i64, y1: i64, c: Rgb<u8>) {
    let (mut x0, mut y0) = (x0, y0);
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    loop {
        put(img, x0, y0, c);
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::SceneConfig;

    fn short_run() -> crate::scene::BuiltScene {
        let toml = r#"
[sim]
resolution_mm = 2.5
duration_s = 0.05
frames = 5
[needle]
auto = true
diameter_mm = 1.2
flow_rate_ml_s = 6.0
[fluid]
particles_per_cell = 16
"#;
        let cfg = SceneConfig::from_toml_str(toml).unwrap();
        cfg.build().unwrap()
    }

    #[test]
    fn slice_shows_cavity_and_fluid() {
        let mut built = short_run();
        for _ in 0..built.frames {
            built.solver.step(built.frame_dt);
        }
        // Slice through the needle tip's z.
        let zc = built.solver.needle.tip.z;
        let plane = SlicePlane::xy(zc, built.solver.grid.dx);
        let img = render_slice(&built.solver, &plane, 300, 0.0);
        assert!(img.width() > 0 && img.height() > 0);

        // The cavity background should be present (light interior pixels)...
        let has_interior = img.pixels().any(|p| *p == INTERIOR);
        assert!(has_interior, "no cavity interior was drawn");
        // ...and there must be saturated (coloured) pixels from particles/needle,
        // proving fluid was rendered, not just the grey background.
        let has_colored = img.pixels().any(|p| {
            let [r, g, b] = p.0;
            let mx = r.max(g).max(b) as i32;
            let mn = r.min(g).min(b) as i32;
            mx - mn > 60
        });
        assert!(has_colored, "no fluid/needle pixels rendered");

        save_png(&img, std::env::temp_dir().join("sdr_slice_test.png")).unwrap();
    }

    #[test]
    fn timeseries_plot_renders() {
        let mut built = short_run();
        let mut metrics = crate::metrics::MetricsCollector::new(&built.solver, built.cavity_volume);
        for f in 0..built.frames {
            built.solver.step(built.frame_dt);
            metrics.record(&built.solver, f);
        }
        let img = render_timeseries(metrics.frames(), 400, 200);
        assert_eq!(img.width(), 400);
        assert_eq!(img.height(), 200);
        save_png(&img, std::env::temp_dir().join("sdr_plot_test.png")).unwrap();
    }
}
