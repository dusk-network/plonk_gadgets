// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (c) DUSK NETWORK. All rights reserved.

//! Basic `Scalar` oriented gadgets collection.
//!
//! This module actually contains conditional selection implementations as
//! well as equalty-checking gadgets.
use super::AllocatedScalar;
use crate::Error as GadgetsError;
use dusk_plonk::prelude::*;

/// Conditionally selects the value provided or a zero instead.
/// NOTE that the `select` input has to be previously constrained to
/// be either `one` or `zero`.
/// ## Performs:
/// x' = x if select = 1
/// x' = 0 if select = 0
pub fn conditionally_select_zero(
    composer: &mut StandardComposer,
    x: Variable,
    select: Variable,
) -> Variable {
    composer.mul(BlsScalar::one(), x, select, BlsScalar::zero(), None)
}

/// Conditionally selects the value provided or a one instead.
/// NOTE that the `select` input has to be previously constrained to
/// be either `one` or `zero`.
/// ## Performs:
/// y' = y if selector = 1
/// y' = 1 if selector = 0 =>
/// y' = selector * y + (1 - selector)
pub fn conditionally_select_one(
    composer: &mut StandardComposer,
    y: Variable,
    selector: Variable,
) -> Variable {
    let one = composer.add_witness_to_circuit_description(BlsScalar::one());
    // selector * y
    let selector_y = composer.mul(BlsScalar::one(), y, selector, BlsScalar::zero(), None);
    // 1 - selector
    let one_min_selector = composer.add(
        (BlsScalar::one(), one),
        (-BlsScalar::one(), selector),
        BlsScalar::zero(),
        None,
    );

    // selector * y + (1 - selector)
    composer.add(
        (BlsScalar::one(), selector_y),
        (BlsScalar::one(), one_min_selector),
        BlsScalar::zero(),
        None,
    )
}

/// Provided a `Variable` and the `Scalar` it is attached to, the function
/// constraints the `Variable` to be != Zero.
pub fn is_non_zero(
    composer: &mut StandardComposer,
    var: Variable,
    value_assigned: BlsScalar,
) -> Result<(), GadgetsError> {
    // Add original scalar which is equal to `var`.
    let var_assigned = composer.add_input(value_assigned);
    // Constrain `var` to actually be equal to the `var_assigment` provided.
    composer.assert_equal(var, var_assigned);
    // Compute the inverse of `value_assigned`.
    let inverse = value_assigned.invert();
    let inv: Variable;
    if inverse.is_some().unwrap_u8() == 1u8 {
        // Safe to unwrap here.
        inv = composer.add_input(inverse.unwrap());
    } else {
        return Err(GadgetsError::NonExistingInverse);
    }

    // Var * Inv(Var) = 1
    let one = composer.add_witness_to_circuit_description(BlsScalar::one());
    composer.poly_gate(
        var,
        inv,
        one,
        BlsScalar::one(),
        BlsScalar::zero(),
        BlsScalar::zero(),
        -BlsScalar::one(),
        BlsScalar::zero(),
        None,
    );

    Ok(())
}

/// Returns 1 if a = b and zero otherwise.
///
/// # NOTE
/// If you need to check equality constraining it, this function is not intended for it,
/// instead we recommend to use `composer.assert_equals()` or `composer.constraint_to_constant()`
/// functions from `dusk-plonk` crate which will introduce less constraints to your circuit.
pub fn maybe_equal(
    composer: &mut StandardComposer,
    a: AllocatedScalar,
    b: AllocatedScalar,
) -> Variable {
    // u = a - b
    let u = {
        let q_l_a = (BlsScalar::one(), a.var);
        let q_r_b = (-BlsScalar::one(), b.var);
        let q_c = BlsScalar::zero();

        composer.add(q_l_a, q_r_b, q_c, None)
    };

    // compute z = inverse of u.
    // This is zero for zero and non-zero otherwise
    let u_scalar = a.scalar - b.scalar;
    let u_inv_scalar = u_scalar.invert().unwrap_or(BlsScalar::zero());
    let z = composer.add_input(u_inv_scalar);

    // y = 1 - uz
    let y = composer.mul(-BlsScalar::one(), z, u, BlsScalar::one(), None);

    // yu = 0
    {
        let a = y;
        let b = u;
        let c = u;
        let q_m = BlsScalar::one();
        let q_o = BlsScalar::zero();
        let q_c = BlsScalar::zero();

        composer.mul_gate(a, b, c, q_m, q_o, q_c, None);
    }
    y
}
