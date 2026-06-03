//! Minimal, dependency-free 3D vector math used throughout the solver.
//!
//! Everything is `f64`. For a sinus-scale simulation (centimetre cavity,
//! sub-millimetre cells) the extra precision keeps the pressure-projection
//! conjugate-gradient solve well behaved and removes any `f32`/`f64` juggling
//! from the rest of the code base.

use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

/// A 3-component vector / point in space (metres, internally).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const ZERO: Vec3 = Vec3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    pub const ONE: Vec3 = Vec3 {
        x: 1.0,
        y: 1.0,
        z: 1.0,
    };

    #[inline]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Vec3 { x, y, z }
    }

    #[inline]
    pub const fn splat(v: f64) -> Self {
        Vec3 { x: v, y: v, z: v }
    }

    #[inline]
    pub fn dot(self, o: Vec3) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    #[inline]
    pub fn cross(self, o: Vec3) -> Vec3 {
        Vec3::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }

    #[inline]
    pub fn length_squared(self) -> f64 {
        self.dot(self)
    }

    #[inline]
    pub fn length(self) -> f64 {
        self.length_squared().sqrt()
    }

    #[inline]
    pub fn distance(self, o: Vec3) -> f64 {
        (self - o).length()
    }

    /// Normalised copy. Returns the zero vector when the length is ~0 so the
    /// caller never gets a NaN from dividing by zero.
    #[inline]
    pub fn normalize_or_zero(self) -> Vec3 {
        let len = self.length();
        if len > 1e-12 {
            self / len
        } else {
            Vec3::ZERO
        }
    }

    #[inline]
    pub fn min(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x.min(o.x), self.y.min(o.y), self.z.min(o.z))
    }

    #[inline]
    pub fn max(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x.max(o.x), self.y.max(o.y), self.z.max(o.z))
    }

    /// Clamp each component into `[lo, hi]` component-wise.
    #[inline]
    pub fn clamp(self, lo: Vec3, hi: Vec3) -> Vec3 {
        self.max(lo).min(hi)
    }

    #[inline]
    pub fn abs(self) -> Vec3 {
        Vec3::new(self.x.abs(), self.y.abs(), self.z.abs())
    }

    #[inline]
    pub fn max_component(self) -> f64 {
        self.x.max(self.y).max(self.z)
    }

    #[inline]
    pub fn min_component(self) -> f64 {
        self.x.min(self.y).min(self.z)
    }

    #[inline]
    pub fn to_array(self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }

    #[inline]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
}

impl From<[f64; 3]> for Vec3 {
    fn from(a: [f64; 3]) -> Self {
        Vec3::new(a[0], a[1], a[2])
    }
}

impl Add for Vec3 {
    type Output = Vec3;
    #[inline]
    fn add(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}

impl AddAssign for Vec3 {
    #[inline]
    fn add_assign(&mut self, o: Vec3) {
        self.x += o.x;
        self.y += o.y;
        self.z += o.z;
    }
}

impl Sub for Vec3 {
    type Output = Vec3;
    #[inline]
    fn sub(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}

impl SubAssign for Vec3 {
    #[inline]
    fn sub_assign(&mut self, o: Vec3) {
        self.x -= o.x;
        self.y -= o.y;
        self.z -= o.z;
    }
}

impl Mul<f64> for Vec3 {
    type Output = Vec3;
    #[inline]
    fn mul(self, s: f64) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }
}

impl Mul<Vec3> for f64 {
    type Output = Vec3;
    #[inline]
    fn mul(self, v: Vec3) -> Vec3 {
        v * self
    }
}

impl Div<f64> for Vec3 {
    type Output = Vec3;
    #[inline]
    fn div(self, s: f64) -> Vec3 {
        Vec3::new(self.x / s, self.y / s, self.z / s)
    }
}

impl Neg for Vec3 {
    type Output = Vec3;
    #[inline]
    fn neg(self) -> Vec3 {
        Vec3::new(-self.x, -self.y, -self.z)
    }
}

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Aabb { min, max }
    }

    /// An "empty" box that grows correctly under [`Aabb::expand`].
    pub fn empty() -> Self {
        Aabb {
            min: Vec3::splat(f64::INFINITY),
            max: Vec3::splat(f64::NEG_INFINITY),
        }
    }

    pub fn from_points(points: &[Vec3]) -> Self {
        let mut bb = Aabb::empty();
        for &p in points {
            bb.expand(p);
        }
        bb
    }

    #[inline]
    pub fn expand(&mut self, p: Vec3) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }

    #[inline]
    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }

    #[inline]
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Grow the box by `margin` in every direction.
    pub fn padded(&self, margin: f64) -> Aabb {
        Aabb {
            min: self.min - Vec3::splat(margin),
            max: self.max + Vec3::splat(margin),
        }
    }

    #[inline]
    pub fn contains(&self, p: Vec3) -> bool {
        p.x >= self.min.x
            && p.x <= self.max.x
            && p.y >= self.min.y
            && p.y <= self.max.y
            && p.z >= self.min.z
            && p.z <= self.max.z
    }
}

/// Linear interpolation between `a` and `b` by `t`.
#[inline]
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn dot_and_cross() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        assert_relative_eq!(a.dot(b), 0.0);
        let c = a.cross(b);
        assert_relative_eq!(c.x, 0.0);
        assert_relative_eq!(c.y, 0.0);
        assert_relative_eq!(c.z, 1.0);
    }

    #[test]
    fn length_and_normalize() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        assert_relative_eq!(v.length(), 5.0);
        let n = v.normalize_or_zero();
        assert_relative_eq!(n.length(), 1.0);
        assert_eq!(Vec3::ZERO.normalize_or_zero(), Vec3::ZERO);
    }

    #[test]
    fn aabb_expands() {
        let bb = Aabb::from_points(&[Vec3::new(-1.0, 2.0, 0.0), Vec3::new(3.0, -2.0, 5.0)]);
        assert_eq!(bb.min, Vec3::new(-1.0, -2.0, 0.0));
        assert_eq!(bb.max, Vec3::new(3.0, 2.0, 5.0));
        assert_eq!(bb.size(), Vec3::new(4.0, 4.0, 5.0));
        assert!(bb.contains(Vec3::new(0.0, 0.0, 1.0)));
        assert!(!bb.contains(Vec3::new(10.0, 0.0, 0.0)));
    }
}
