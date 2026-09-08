//! The noise source the flow-matching solver samples from.
//!
//! Every frontend defined its own copy of this: a seeded `StdRng` plus a
//! `Normal(0, sqrt(temperature))` distribution.

use crate::flow_lm::Rng;

/// Lets a boxed noise source be passed where an `impl Rng` is expected, so
/// callers can choose one at runtime.
impl Rng for Box<dyn Rng + Send> {
    fn sample(&mut self) -> f32 {
        (**self).sample()
    }
}

/// Gaussian noise at a given temperature, seeded for reproducibility.
///
/// The standard deviation is `sqrt(temperature)`, so `temperature` scales the
/// variance of the sampled latents.
pub struct GaussianRng {
    inner: Box<rand::rngs::StdRng>,
    distr: rand_distr::Normal<f32>,
}

impl GaussianRng {
    pub fn new(temperature: f32, seed: u64) -> xn::Result<Self> {
        use rand::SeedableRng;
        if temperature.is_nan() || temperature < 0.0 {
            xn::bail!("temperature must be non-negative, got {temperature}");
        }
        let distr = rand_distr::Normal::new(0f32, temperature.sqrt()).map_err(xn::Error::wrap)?;
        let inner = Box::new(rand::rngs::StdRng::seed_from_u64(seed));
        Ok(Self { inner, distr })
    }
}

impl Rng for GaussianRng {
    fn sample(&mut self) -> f32 {
        use rand::Rng as _;
        self.inner.sample(self.distr)
    }
}

/// Replays a fixed sequence of values, cycling when exhausted. Used to compare
/// this implementation against the reference one step for step.
pub struct ReplayRng {
    values: Vec<f32>,
    index: usize,
}

impl ReplayRng {
    pub fn new(values: Vec<f32>) -> xn::Result<Self> {
        if values.is_empty() {
            xn::bail!("ReplayRng needs at least one value");
        }
        Ok(Self { values, index: 0 })
    }
}

impl Rng for ReplayRng {
    fn sample(&mut self) -> f32 {
        let value = self.values[self.index % self.values.len()];
        self.index += 1;
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_stream() {
        let mut a = GaussianRng::new(0.7, 42).unwrap();
        let mut b = GaussianRng::new(0.7, 42).unwrap();
        for _ in 0..16 {
            assert_eq!(a.sample(), b.sample());
        }
    }

    #[test]
    fn different_seed_different_stream() {
        let mut a = GaussianRng::new(0.7, 1).unwrap();
        let mut b = GaussianRng::new(0.7, 2).unwrap();
        assert_ne!(
            (0..8).map(|_| a.sample()).collect::<Vec<_>>(),
            (0..8).map(|_| b.sample()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn zero_temperature_is_deterministic_zero() {
        let mut rng = GaussianRng::new(0.0, 7).unwrap();
        for _ in 0..8 {
            assert_eq!(rng.sample(), 0.0);
        }
    }

    #[test]
    fn negative_temperature_is_rejected() {
        assert!(GaussianRng::new(-1.0, 0).is_err());
    }

    #[test]
    fn replay_cycles() {
        let mut rng = ReplayRng::new(vec![1.0, 2.0]).unwrap();
        assert_eq!([rng.sample(), rng.sample(), rng.sample()], [1.0, 2.0, 1.0]);
        assert!(ReplayRng::new(vec![]).is_err());
    }
}
