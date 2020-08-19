//! Basic `Scalar` oriented gadgets collection.
//!
//! This module actually contains conditional selection implementations as
//! well as equalty-checking gadgets.
use super::AllocatedScalar;
use crate::gadgets::GadgetErrors;
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
    let selector_y = composer.mul(
        BlsScalar::one(),
        y,
        selector,
        BlsScalar::zero(),
        BlsScalar::zero(),
    );
    // 1 - selector
    let one_min_selector = composer.add(
        (BlsScalar::one(), one),
        (-BlsScalar::one(), selector),
        BlsScalar::zero(),
        BlsScalar::zero(),
    );

    // selector * y + (1 - selector)
    composer.add(
        (BlsScalar::one(), selector_y),
        (BlsScalar::one(), one_min_selector),
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
    let inverse = value_assigned.invert();
    let inv: Variable;
    if inverse.is_some().unwrap_u8() == 1u8 {
        // Safe to unwrap here.
        inv = composer.add_input(inverse.unwrap());
    } else {
        return Err(GadgetErrors::NonExistingInverse.into());
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
        BlsScalar::zero(),
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
        let pi = BlsScalar::zero();

        composer.add(q_l_a, q_r_b, q_c, pi)
    };

    // compute z = inverse of u.
    // This is zero for zero and non-zero otherwise
    let u_scalar = a.scalar - b.scalar;
    let u_inv_scalar = u_scalar.invert().unwrap_or(BlsScalar::zero());
    let z = composer.add_input(u_inv_scalar);

    // y = 1 - uz
    let y = composer.mul(-BlsScalar::one(), z, u, BlsScalar::one(), BlsScalar::zero());

    // yu = 0
    {
        let a = y;
        let b = u;
        let c = u;
        let q_m = BlsScalar::one();
        let q_o = BlsScalar::zero();
        let q_c = BlsScalar::zero();
        let pi = BlsScalar::zero();

        composer.mul_gate(a, b, c, q_m, q_o, q_c, pi);
    }
    y
}

mod tests {
    use super::*;

    #[test]
    fn test_conditionally_select_0() {
        // The circuit closure runs the conditionally_select_zero fn and constraints the result
        // to actually be 0
        let circuit = |composer: &mut StandardComposer, value: BlsScalar, selector: BlsScalar| {
            let value = composer.add_input(value);
            let selector = composer.add_input(selector);
            let res = conditionally_select_zero(composer, value, selector);
            composer.constrain_to_constant(res, BlsScalar::zero(), BlsScalar::zero());
        };

        // Generate Composer & Public Parameters
        let pub_params =
            PublicParameters::setup(1 << 8, &mut rand::thread_rng()).expect("Unexpected error");
        let (ck, vk) = pub_params.trim(1 << 7).expect("Unexpected error");

        // Selector set to 0 should select 0
        // Proving
        let mut prover = Prover::new(b"testing");
        circuit(
            prover.mut_cs(),
            BlsScalar::random(&mut rand::thread_rng()),
            BlsScalar::zero(),
        );
        prover.preprocess(&ck).expect("Error on preprocessing");
        let proof = prover.prove(&ck).expect("Error on proving");

        // Verification
        let mut verifier = Verifier::new(b"testing");
        circuit(
            verifier.mut_cs(),
            BlsScalar::random(&mut rand::thread_rng()),
            BlsScalar::zero(),
        );
        verifier.preprocess(&ck).expect("Error on preprocessing");
        // This should pass since we sent 0 as selector and the circuit closure is constraining the
        // result to be equal to 0.
        assert!(verifier.verify(&proof, &vk, &[BlsScalar::zero()]).is_ok());

        // Selector set to 1 shouldn't assign 0.
        // Proving
        prover.clear_witness();
        circuit(
            prover.mut_cs(),
            BlsScalar::random(&mut rand::thread_rng()),
            BlsScalar::one(),
        );
        let proof = prover.prove(&ck).expect("Error on proving");
        // This shouldn't pass since we sent 1 as selector and the circuit closure is constraining the
        // result to be equal to 0 while the value assigned is indeed the randomly generated one.
        assert!(verifier.verify(&proof, &vk, &[BlsScalar::zero()]).is_err());
    }

    #[test]
    fn test_conditionally_select_1() {
        // The circuit closure runs the conditionally_select_one fn and constraints the result
        // to actually to be equal to the provided expected_result.
        let circuit = |composer: &mut StandardComposer,
                       value: BlsScalar,
                       selector: BlsScalar,
                       expected_result: BlsScalar| {
            let value = composer.add_input(value);
            let selector = composer.add_input(selector);
            let res = conditionally_select_one(composer, value, selector);
            composer.constrain_to_constant(res, BlsScalar::zero(), -expected_result);
        };

        // Generate Composer & Public Parameters
        let pub_params =
            PublicParameters::setup(1 << 8, &mut rand::thread_rng()).expect("Unexpected error");
        let (ck, vk) = pub_params.trim(1 << 7).expect("Unexpected error");

        // Selector set to 0 should asign 1
        // Proving
        let mut prover = Prover::new(b"testing");
        circuit(
            prover.mut_cs(),
            BlsScalar::random(&mut rand::thread_rng()),
            BlsScalar::zero(),
            BlsScalar::one(),
        );
        let mut pi = prover.mut_cs().public_inputs.clone();
        prover.preprocess(&ck).expect("Error on preprocessing");
        let proof = prover.prove(&ck).expect("Error on proving");

        // Verification
        let mut verifier = Verifier::new(b"testing");
        circuit(
            verifier.mut_cs(),
            BlsScalar::random(&mut rand::thread_rng()),
            BlsScalar::zero(),
            BlsScalar::one(),
        );
        verifier.preprocess(&ck).expect("Error on preprocessing");
        // This should pass since we sent 0 as selector and the circuit should then return
        // 1.
        assert!(verifier.verify(&proof, &vk, &pi).is_ok());

        // Selector set to 1 should assign the randomly-generated value.
        // Proving
        prover.clear_witness();
        let rand = BlsScalar::random(&mut rand::thread_rng());
        circuit(prover.mut_cs(), rand, BlsScalar::one(), rand);
        pi = prover.mut_cs().public_inputs.clone();
        let proof = prover.prove(&ck).expect("Error on proving");
        // This should pass since we sent 1 as selector and the circuit closure should assign the randomly-generated
        // value as a result.
        assert!(verifier.verify(&proof, &vk, &pi).is_ok());
    }

    #[test]
    fn test_is_not_zero() {
        // The circuit closure runs the is_not_zero fn and constraints the input to
        // not be zero.
        let circuit = |composer: &mut StandardComposer,
                       value: BlsScalar,
                       value_assigned: BlsScalar|
         -> Result<(), Error> {
            let value = composer.add_input(value);
            is_non_zero(composer, value, value_assigned)
        };

        // Generate Composer & Public Parameters
        let pub_params =
            PublicParameters::setup(1 << 8, &mut rand::thread_rng()).expect("Unexpected error");
        let (ck, vk) = pub_params.trim(1 << 7).expect("Unexpected error");

        // Value  & Value assigned set to 0 should err
        // Proving
        let mut prover = Prover::new(b"testing");
        assert!(circuit(prover.mut_cs(), BlsScalar::zero(), BlsScalar::zero()).is_err());
        prover.clear_witness();

        // Value and value_assigned with different values should fail on verification.
        // Proving
        let mut prover = Prover::new(b"testing");
        assert!(circuit(
            prover.mut_cs(),
            BlsScalar::random(&mut rand::thread_rng()),
            BlsScalar::random(&mut rand::thread_rng()),
        )
        .is_ok());
        let mut pi = prover.mut_cs().public_inputs.clone();
        prover.preprocess(&ck).expect("Error on preprocessing");
        let proof = prover.prove(&ck).expect("Error on proving");

        // Verification
        let mut verifier = Verifier::new(b"testing");
        circuit(
            verifier.mut_cs(),
            BlsScalar::random(&mut rand::thread_rng()),
            BlsScalar::random(&mut rand::thread_rng()),
        )
        .expect("Error on gadget run");
        verifier.preprocess(&ck).expect("Error on preprocessing");
        assert!(verifier.verify(&proof, &vk, &pi).is_err());

        // Value & value assigned set correctly and != 0. This should pass.
        // Proving
        prover.clear_witness();
        let rand = BlsScalar::random(&mut rand::thread_rng());
        circuit(prover.mut_cs(), rand, rand).expect("Error on gadget run");
        pi = prover.mut_cs().public_inputs.clone();
        let proof = prover.prove(&ck).expect("Error on proving");
        // This should pass since we sent 1 as selector and the circuit closure should assign the randomly-generated
        // value as a result.
        assert!(verifier.verify(&proof, &vk, &pi).is_ok());
    }
}
