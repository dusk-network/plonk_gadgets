///! Basic `Scalar` oriented gadgets collection.
use super::GadgetErrors;
use anyhow::{Error, Result};
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
    composer.mul(
        BlsScalar::one(),
        x,
        select,
        BlsScalar::zero(),
        BlsScalar::zero(),
    )
}

/// Conditionally selects the value provided or a one instead.
/// NOTE that the `select` input has to be previously constrained to
/// be either `one` or `zero`.
/// ## Performs:
/// y' = y if bit = 1
/// y' = 1 if bit = 0 =>
/// y' = bit * y + (1 - bit)
pub fn conditionally_select_one(
    composer: &mut StandardComposer,
    y: Variable,
    select: Variable,
) -> Variable {
    let one = composer.add_constant_witness(BlsScalar::one());
    // bit * y
    let bit_y = composer.mul(
        BlsScalar::one(),
        y,
        select,
        BlsScalar::zero(),
        BlsScalar::zero(),
    );
    // 1 - bit
    let one_min_bit = composer.add(
        (BlsScalar::one(), one),
        (-BlsScalar::one(), select),
        BlsScalar::zero(),
        BlsScalar::zero(),
    );

    // bit * y + (1 - bit)
    composer.add(
        (BlsScalar::one(), bit_y),
        (BlsScalar::one(), one_min_bit),
        BlsScalar::zero(),
        BlsScalar::zero(),
    )
}

/// Provided a `Variable` and the `Scalar` it is attached to, the function
/// constraints the `Variable` to be != Zero.
pub fn is_non_zero(
    composer: &mut StandardComposer,
    var: Variable,
    value_assigned: BlsScalar,
) -> Result<(), Error> {
    // Add original scalar which is equal to `var`.
    let var_assigned = composer.add_input(value_assigned);
    // Constrain `var` to actually be equal to the `var_assigment` provided.
    composer.assert_equal(var, var_assigned);
    // Compute the inverse of `value_assigned`.
    let mut inverse = value_assigned.invert();
    let inv: Variable;
    if inverse.is_some().unwrap_u8() == 1u8 {
        // Safe to unwrap here.
        inv = composer.add_input(inverse.unwrap());
    } else {
        return Err(GadgetErrors::NonExistingInverse.into());
    }

    // Var * Inv(Var) = 1
    let one = composer.add_constant_witness(BlsScalar::one());
    composer.poly_gate(
        var,
        inv,
        one,
        BlsScalar::one(),
        BlsScalar::zero(),
        BlsScalar::zero(),
        -BlsScalar::one(),
        BlsScalar::zero(),
        BlsScalar::zero(),
    );

    Ok(())
}

mod tests {
    use super::*;
}
