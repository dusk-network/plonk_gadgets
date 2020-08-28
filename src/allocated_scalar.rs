// Copyright (c) DUSK NETWORK. All rights reserved.
// Licensed under the MPL 2.0 license. See LICENSE file in the project root for details.”
//! This module contains the implementation of the
//! `AllocatedScalar` helper structure.
use dusk_plonk::{
    bls12_381::Scalar as BlsScalar,
    constraint_system::{StandardComposer, Variable},
};

/// An allocated scalar holds the underlying witness assignment for the Prover
/// and a dummy value for the verifier
/// XXX: This could possibly be added to the PLONK API
#[derive(Copy, Clone, Debug)]
pub struct AllocatedScalar {
    /// Variable associated to the `Scalar`.
    pub var: Variable,
    /// Scalar associated to the `Variable`
    pub scalar: BlsScalar,
}

impl AllocatedScalar {
    /// Allocates a BlsScalar into the constraint system as a witness
    pub fn allocate(composer: &mut StandardComposer, scalar: BlsScalar) -> AllocatedScalar {
        let var = composer.add_input(scalar);
        AllocatedScalar { var, scalar }
    }
}
