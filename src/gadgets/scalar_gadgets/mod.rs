//! This module contains the implementation of all of the
//! Scalar-related gadgets.
pub mod range;
pub mod scalar;
use dusk_plonk::prelude::*;

/// An allocated scalar holds the underlying witness assignment for the Prover
/// and a dummy value for the verifier
/// XXX: This could possibly be added to the PLONK API
#[derive(Clone, Debug)]
pub struct AllocatedScalar {
    /// Variable associated to the `Scalar`.
    pub var: Variable,
    /// Scalar associated to the `Varaible`
    pub scalar: BlsScalar,
}

impl AllocatedScalar {
    /// Allocates a BlsScalar into the constraint system as a witness
    pub fn allocate(composer: &mut StandardComposer, scalar: BlsScalar) -> AllocatedScalar {
        let var = composer.add_input(scalar);
        AllocatedScalar { var, scalar }
    }
}
