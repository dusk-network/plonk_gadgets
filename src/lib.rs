#![feature(maybe_uninit_uninit_array)]

use plonk::cs::{composer::StandardComposer as PlonkStandardComposer, proof::Proof as PlonkProof};

pub use algebra::{
    curves::bls12_381::Bls12_381 as Curve,
    fields::{bls12_381::fr::Fr as Scalar, Field},
};
pub use plonk::cs::constraint_system::Variable;

pub type StandardComposer = PlonkStandardComposer<Curve>;
pub type Proof = PlonkProof<Curve>;

pub mod gadgets;
pub mod helpers;
