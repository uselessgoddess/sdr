//! Parametric model of a human maxillary sinus with an oroantral
//! (tooth-socket) communication.
//!
//! The maxillary sinus is the pyramidal air cavity in the cheekbone, sitting
//! directly above the roots of the upper molars and premolars. When such a
//! tooth is extracted, the socket can open into the sinus floor — an *oroantral
//! communication*. Clinicians irrigate the sinus by passing a thin cannula up
//! through this communication.
//!
//! This module builds an anatomically-plausible, fully parametric closed
//! surface for that cavity so the solver has a domain to work in even without
//! patient DICOM data. Real patient meshes (segmented in 3D Slicer / Blender as
//! in the thesis) can be loaded instead via [`crate::mesh::TriMesh::load`].
//!
//! **Coordinate frame** (metres): `+x` anteroposterior, `+y` up (towards the
//! sinus roof), `+z` mediolateral. Gravity points along `-y`. The socket sits
//! on the floor and the irrigation needle enters from below, pointing up.

use crate::math::{Aabb, Vec3};
use crate::mesh::TriMesh;
use crate::surface_nets::surface_nets;
use serde::{Deserialize, Serialize};

/// Parameters of the parametric sinus cavity. All lengths in metres.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SinusParams {
    /// Half-extents of the cavity body ellipsoid (x: AP, y: vertical, z: ML).
    pub semi_axes: Vec3,
    /// Centre of the cavity body.
    pub center: Vec3,
    /// Horizontal width fraction at the floor relative to the roof
    /// (`1.0` = no taper, `0.6` ≈ pyramidal). Models the narrowing towards the
    /// alveolar floor.
    pub taper: f64,
    /// Amplitude of gentle organic undulation of the wall (fraction of radius).
    pub bumpiness: f64,
    /// Horizontal (x, z) position of the tooth-socket communication on the floor.
    pub socket_xz: (f64, f64),
    /// Radius of the socket channel.
    pub socket_radius: f64,
    /// How far the socket channel protrudes below the cavity floor.
    pub socket_depth: f64,
    /// Node spacing used when polygonising the implicit model.
    pub model_dx: f64,
}

impl Default for SinusParams {
    /// Average adult maxillary sinus (~13 ml after taper), with a ~2.5 mm
    /// communication on the posterior floor.
    fn default() -> Self {
        SinusParams {
            semi_axes: Vec3::new(0.017, 0.016, 0.013),
            center: Vec3::ZERO,
            taper: 0.6,
            bumpiness: 0.12,
            socket_xz: (0.006, 0.0),
            socket_radius: 0.0025,
            socket_depth: 0.006,
            model_dx: 0.0007,
        }
    }
}

/// Polynomial smooth-minimum (negative-inside union of implicit fields).
#[inline]
fn smooth_min(a: f64, b: f64, k: f64) -> f64 {
    if k <= 0.0 {
        return a.min(b);
    }
    let h = (0.5 + 0.5 * (b - a) / k).clamp(0.0, 1.0);
    (b * (1.0 - h) + a * h) - k * h * (1.0 - h)
}

impl SinusParams {
    /// The y-coordinate of the bottom of the socket channel (its opening below
    /// the floor, where the needle enters).
    pub fn socket_bottom_y(&self) -> f64 {
        self.center.y - self.semi_axes.y - self.socket_depth
    }

    /// The top of the socket channel, set inside the lower cavity body so the
    /// channel reliably connects to the cavity.
    fn socket_top_y(&self) -> f64 {
        self.center.y - 0.5 * self.semi_axes.y
    }

    /// Implicit field of the cavity: negative inside (fluid/air), positive in
    /// the wall. The zero level set is the cavity surface.
    pub fn implicit(&self, p: Vec3) -> f64 {
        let q = p - self.center;
        let a = self.semi_axes;

        // Pyramidal taper: horizontal extent shrinks towards the floor.
        let h = ((q.y / a.y) + 1.0) * 0.5; // 0 at bottom, 1 at top
        let scale = self.taper + (1.0 - self.taper) * h.clamp(0.0, 1.5);

        // Gentle organic undulation of the wall radius.
        let bump = self.bumpiness
            * (0.6 * (4.0 * q.x / a.x).sin() * (3.0 * q.z / a.z).cos()
                + 0.4 * (5.0 * q.y / a.y + 1.3).sin());

        let body =
            ((q.x / (a.x * scale)).powi(2) + (q.y / a.y).powi(2) + (q.z / (a.z * scale)).powi(2))
                .sqrt()
                - (1.0 + bump);
        // Convert the (dimensionless) ellipsoid field to an approximate metric
        // distance so the smooth-union blend width is meaningful.
        let body = body * a.min_component();

        // Socket channel: a vertical cylinder segment opening through the floor.
        let (sx, sz) = self.socket_xz;
        let radial = ((p.x - sx).powi(2) + (p.z - sz).powi(2)).sqrt() - self.socket_radius;
        let ytop = self.socket_top_y();
        let ybot = self.socket_bottom_y();
        let axial = (p.y - ytop).max(ybot - p.y);
        let socket = radial.max(axial);

        smooth_min(body, socket, 0.0015)
    }

    /// World-space bounding box that comfortably contains the whole model.
    pub fn bounds(&self) -> Aabb {
        let a = self.semi_axes * (1.0 + self.bumpiness);
        let mut bb = Aabb::new(self.center - a, self.center + a);
        // Include the socket stub.
        let (sx, sz) = self.socket_xz;
        bb.expand(Vec3::new(
            sx + self.socket_radius,
            self.socket_bottom_y(),
            sz + self.socket_radius,
        ));
        bb.expand(Vec3::new(
            sx - self.socket_radius,
            self.socket_bottom_y(),
            sz - self.socket_radius,
        ));
        bb.padded(2.0 * self.model_dx)
    }

    /// Generate the closed triangle mesh of the cavity.
    pub fn generate(&self) -> TriMesh {
        let bb = self.bounds();
        let size = bb.size();
        let dims = (
            (size.x / self.model_dx).ceil() as usize + 1,
            (size.y / self.model_dx).ceil() as usize + 1,
            (size.z / self.model_dx).ceil() as usize + 1,
        );
        surface_nets(bb.min, self.model_dx, dims, |p| self.implicit(p))
    }

    /// Suggested needle entry: tip just inside the cavity floor at the socket,
    /// with the needle axis pointing up into the cavity.
    pub fn suggested_needle(&self) -> (Vec3, Vec3) {
        let (sx, sz) = self.socket_xz;
        // Tip a little above the socket top, inside the cavity.
        let tip = Vec3::new(sx, self.socket_top_y() + self.semi_axes.y * 0.15, sz);
        let axis = Vec3::new(0.0, 1.0, 0.0);
        (tip, axis)
    }
}

/// Estimate the interior (air) volume of a closed cavity mesh by Monte-Carlo /
/// grid sampling of an SDF. Used for reporting and for fill-fraction metrics.
pub fn cavity_volume(mesh: &TriMesh, dx: f64) -> f64 {
    use crate::sdf::Sdf;
    let sdf = Sdf::from_mesh(mesh, dx, 2.0 * dx);
    let cell = dx * dx * dx;
    let mut vol = 0.0;
    for &phi in &sdf.data {
        if phi < 0.0 {
            vol += cell;
        }
    }
    vol
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_closed_cavity() {
        let params = SinusParams::default();
        let mesh = params.generate();
        assert!(!mesh.is_empty(), "sinus mesh should not be empty");
        // Outward winding => positive enclosed volume.
        let vol = mesh.signed_volume();
        assert!(vol > 0.0, "expected positive signed volume, got {vol}");
        // Should be in a plausible range for a maxillary sinus: 5–30 ml
        // (1 ml = 1e-6 m^3).
        let ml = vol * 1e6;
        assert!(
            ml > 4.0 && ml < 35.0,
            "implausible sinus volume: {ml:.2} ml"
        );
    }

    #[test]
    fn needle_tip_is_inside_cavity() {
        let params = SinusParams::default();
        let (tip, axis) = params.suggested_needle();
        // The suggested tip should lie inside the cavity (implicit < 0).
        assert!(
            params.implicit(tip) < 0.0,
            "needle tip should be inside the cavity"
        );
        assert!(axis.y > 0.0, "needle should point upward into the cavity");
    }

    #[test]
    fn socket_is_below_body() {
        let params = SinusParams::default();
        // A point on the socket axis below the floor should be inside the cavity
        // (the communication channel).
        let (sx, sz) = params.socket_xz;
        let p = Vec3::new(sx, params.socket_bottom_y() + 0.001, sz);
        assert!(
            params.implicit(p) < 0.0,
            "socket channel should be open below the floor"
        );
    }
}
