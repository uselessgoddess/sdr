//! Fluid marker particles for the FLIP/PIC solver.
//!
//! Each particle carries a position and a velocity. The fluid is *represented*
//! by these particles (a cell is "fluid" if it contains at least one), while
//! the velocity field is solved on the grid. Particles advect through the grid
//! velocity, which is what gives FLIP its low numerical dissipation compared to
//! a purely grid-based advection.

use crate::math::Vec3;

/// A collection of marker particles (structure-of-arrays for cache efficiency).
#[derive(Debug, Clone, Default)]
pub struct ParticleSet {
    pub positions: Vec<Vec3>,
    pub velocities: Vec<Vec3>,
}

impl ParticleSet {
    pub fn new() -> Self {
        ParticleSet::default()
    }

    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    pub fn push(&mut self, position: Vec3, velocity: Vec3) {
        self.positions.push(position);
        self.velocities.push(velocity);
    }

    /// Remove particles for which `keep` returns false. Used to delete
    /// particles that drain out of the domain or escape the cavity.
    pub fn retain(&mut self, mut keep: impl FnMut(Vec3, Vec3) -> bool) {
        let mut w = 0;
        for r in 0..self.positions.len() {
            if keep(self.positions[r], self.velocities[r]) {
                self.positions[w] = self.positions[r];
                self.velocities[w] = self.velocities[r];
                w += 1;
            }
        }
        self.positions.truncate(w);
        self.velocities.truncate(w);
    }

    /// Total kinetic energy assuming unit particle mass (relative measure).
    pub fn kinetic_energy(&self) -> f64 {
        0.5 * self.velocities.iter().map(|v| v.length_squared()).sum::<f64>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_retain() {
        let mut ps = ParticleSet::new();
        for i in 0..10 {
            ps.push(Vec3::new(i as f64, 0.0, 0.0), Vec3::ZERO);
        }
        assert_eq!(ps.len(), 10);
        ps.retain(|p, _| p.x < 5.0);
        assert_eq!(ps.len(), 5);
        assert!(ps.positions.iter().all(|p| p.x < 5.0));
    }
}
