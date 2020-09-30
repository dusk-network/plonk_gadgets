// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (c) DUSK NETWORK. All rights reserved.

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
