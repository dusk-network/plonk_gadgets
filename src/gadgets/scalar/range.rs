//! Collection of range-checking gadgets for Bls12_381 scalars.
use crate::gadgets::GadgetErrors;
use anyhow::{Error, Result};
use dusk_plonk::prelude::*;

/// Builds a complex range-proof (not bounded to a pow_of_two) given a
/// composer, the max range and the witness.
///
/// Checks that `witness < max_range` returning a boolean `Variable` as a result
/// where `1 = holds` and `0 = Does not hold`.
pub fn single_complex_rangeproof_gadget(
    composer: &mut StandardComposer,
    witness: BlsScalar,
    witness_var: Variable,
    max_range: BlsScalar,
    min_range: Option<BlsScalar>,
) -> Result<Variable, Error> {
    // First, we need to ensure that the witnesses we got are the same.
    let scalar_witness = composer.add_input(witness);
    composer.assert_equal(witness_var, scalar_witness);

    // Create constant witness one
    let one = composer.add_constant_witness(BlsScalar::one());
    // Create 0 as witness value
    let zero = composer.add_constant_witness(BlsScalar::zero());
    // Compute number of bits needed to represent the maximum range
    let num_bits = bits_count(max_range);
    // Compute the closest power of two knowing the bits needed to represent the max range.
    let closest_pow_of_two = BlsScalar::from(2u64).pow(&[num_bits, 0, 0, 0]);
    // Compute b' max range.
    let b_prime = closest_pow_of_two - max_range;
    // Assing the minimum range to `a` when specified and zero otherways.
    let a = min_range.unwrap_or(BlsScalar::zero());
    // Obtain 256-bit representation of `witness + b'`.
    let bits_witness_plus_bprime = scalar_to_bits(&(witness + b_prime));
    // Obtain 256-bit representation of `witness -a `.
    let bits_witness_min_a = scalar_to_bits(&(witness - a));
    let mut witness_plus_b_prime_accumulator = zero;
    let mut witness_min_a_accumulator = zero;
    // Compute the sum of the bit representation of `witness - a` inside and outside
    // of the circuit.
    //
    // Effectively, we're doing the following: `Sum_i(w_i * 2^(i-1))`.
    let _accumulator_witness_min_a = bits_witness_min_a[..=num_bits as usize]
        .iter()
        .enumerate()
        .fold(BlsScalar::zero(), |scalar_accum, (idx, bit)| {
            let bit_var = composer.add_input(BlsScalar::from(*bit as u64));
            // Apply boolean constraint to the bit.
            composer.boolean_gate(bit_var);
            // Accumulate the sum of bits multiplied by the corresponding 2^(i-1) as a `Variable`
            witness_min_a_accumulator = composer.add(
                (BlsScalar::one(), witness_min_a_accumulator),
                (BlsScalar::from(2u64).pow(&[idx as u64, 0, 0, 0]), bit_var),
                BlsScalar::zero(),
                BlsScalar::zero(),
            );
            // Accumulate the sum of bits multiplied by the corresponding 2^(i-1) as a `Scalar`
            scalar_accum
                + (BlsScalar::from(*bit as u64) * BlsScalar::from(2u64).pow(&[idx as u64, 0, 0, 0]))
        });
    // Constrain : `Sum(wi * 2^(i-1)) - (witness - a).`
    let witness_min_a = composer.add(
        (BlsScalar::one(), witness_var),
        (-a, one),
        BlsScalar::zero(),
        BlsScalar::zero(),
    );
    composer.assert_equal(witness_min_a, witness_min_a_accumulator);
    // Compute the sum of the bit representation of `witness + b_prime` inside and outside
    // of the circuit.
    //
    // Effectively, we're doing the following: `Sum_i(v_i * 2^(i-1))`.
    let accumulator_witness_plus_b_prime = bits_witness_plus_bprime[..=num_bits as usize]
        .iter()
        .enumerate()
        .fold(BlsScalar::zero(), |scalar_accum, (idx, bit)| {
            let bit_var = composer.add_input(BlsScalar::from(*bit as u64));
            // Apply boolean constraint to the bit.
            composer.boolean_gate(bit_var);
            // Accumulate the sum of bits_witness_plus_bprime multiplied by the corresponding 2^(i-1) as a `Variable`
            witness_plus_b_prime_accumulator = composer.add(
                (BlsScalar::one(), witness_plus_b_prime_accumulator),
                (BlsScalar::from(2u64).pow(&[idx as u64, 0, 0, 0]), bit_var),
                BlsScalar::zero(),
                BlsScalar::zero(),
            );
            // Accumulate the sum of bits_witness_plus_bprime multiplied by the corresponding 2^(i-1) as a `Scalar`
            scalar_accum
                + (BlsScalar::from(*bit as u64) * BlsScalar::from(2u64).pow(&[idx as u64, 0, 0, 0]))
        });
    // Compute `Chi(x)` =  Sum(vi * 2^(i-1)) - (witness + b').
    // Note that the result will be equal to: `0 (if the reangeproof holds)
    // or any other value if it doesn't.
    dbg!(witness);
    dbg!(b_prime);
    dbg!(accumulator_witness_plus_b_prime);
    dbg!(witness + b_prime);
    dbg!(witness + b_prime - accumulator_witness_plus_b_prime);
    let witness_plus_b_prime = composer.add(
        (BlsScalar::one(), witness_var),
        (BlsScalar::from(b_prime), one),
        BlsScalar::zero(),
        BlsScalar::zero(),
    );
    let chi_x_var = composer.add(
        (BlsScalar::one(), witness_plus_b_prime),
        (-BlsScalar::one(), witness_plus_b_prime_accumulator),
        BlsScalar::zero(),
        BlsScalar::zero(),
    );

    // It is possible to replace a constraint `chi(x)=0` on variables
    // `x` with a set of constraints on  new variables
    // `(u,y,z)` such that `y=1 if chi(x) holds` and `y=0 otherwise`.
    // We introduce new variables `u, y, z` that are computed as follows:

    // `u = witness + b_prime - accumulator` which should equal `chi_x`.
    let u = witness + b_prime - accumulator_witness_plus_b_prime;
    let u_var = composer.big_add(
        (BlsScalar::one(), witness_var),
        (-BlsScalar::one(), witness_plus_b_prime_accumulator),
        None,
        b_prime,
        BlsScalar::zero(),
    );
    // Conditionally assign `1` or `0` to `y`.
    let y = if u == BlsScalar::zero() {
        composer.add_input(BlsScalar::one())
    } else {
        composer.add_input(BlsScalar::zero())
    };
    // Conditionally assign `1/u` or `0` to z
    let mut z = zero;
    if u != BlsScalar::zero() {
        // If u != zero -> `z = 1/u`
        // Otherways, `u = 0` as it was defined avobe.
        // Check inverse existance, otherways, err.
        if u.invert().is_none().into() {
            return Err(GadgetErrors::NonExistingInverse.into());
        };
        // Safe to unwrap here.
        z = composer.add_input(u.invert().unwrap());
    }
    // We can safely unwrap `u` now.
    // Now we need to check the following to ensure we can provide a boolean
    // result representing wether the rangeproof holds or not:
    // `u = Chi(x)`.
    // `u * z = 1 - y`.
    // `y * u = 0`.
    composer.assert_equal(u_var, chi_x_var);
    let one_min_y = composer.add(
        (BlsScalar::one(), one),
        (-BlsScalar::one(), y),
        BlsScalar::zero(),
        BlsScalar::zero(),
    );
    let u_times_z = composer.mul(
        BlsScalar::one(),
        u_var,
        z,
        BlsScalar::zero(),
        BlsScalar::zero(),
    );
    composer.assert_equal(one_min_y, u_times_z);
    let y_times_u = composer.mul(
        BlsScalar::one(),
        u_var,
        y,
        BlsScalar::zero(),
        BlsScalar::zero(),
    );
    composer.assert_equal(y_times_u, zero);
    // Constraint the result to be boolean
    composer.boolean_gate(y);
    Ok(y)
}

/// Builds a complex range-proof (not bounded to powers_of_two) given a
/// composer, the max range, the min_range and the witness.
///
/// Checks that `min_range < witness < max_range` returning a boolean `Variable` as a result
/// where `1 = holds` and `0 = Does not hold`.
fn complete_complex_rangeproof_gadget(
    composer: &mut StandardComposer,
    witness: BlsScalar,
    witness_var: Variable,
    min_range: BlsScalar,
    max_range: BlsScalar,
) -> Result<Variable, Error> {
    let one = composer.add_constant_witness(BlsScalar::one());
    // The goal is to convert a double bound check into a single one.
    // We can achive that by doing the following:
    // 1. Witness = Witness - min_range
    // 2. max_range = max_range - min_range
    // 3. Compute a single complex rangeproof using the witness value obtained
    // at step (1.) and set the `max_range` to be the one obtained at step (2.)
    let new_witness = witness - min_range;
    let new_witness_var = composer.add(
        (BlsScalar::one(), witness_var),
        (-min_range, one),
        BlsScalar::zero(),
        BlsScalar::zero(),
    );
    let new_upper_bound = max_range - min_range;
    dbg!(new_witness, new_upper_bound);
    single_complex_rangeproof_gadget(
        composer,
        new_witness,
        new_witness_var,
        new_upper_bound,
        Some(min_range),
    )
}

// Decompose a `BlsScalar` into its 256-bit representation.
fn scalar_to_bits(scalar: &BlsScalar) -> [u8; 256] {
    let mut res = [0u8; 256];
    let bytes = scalar.to_bytes();
    for (byte, bits) in bytes.iter().zip(res.chunks_mut(8)) {
        bits.iter_mut()
            .enumerate()
            .for_each(|(i, bit)| *bit = (byte >> i) & 1)
    }
    res
}

// Count the minimum amount of bits necessary to represent a `BlsScalar`.
fn bits_count(mut scalar: BlsScalar) -> u64 {
    scalar = scalar.reduce();
    let mut counter = 1u64;
    while scalar > BlsScalar::one().reduce() {
        scalar.divn(1);
        counter += 1;
    }
    counter
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counting_scalar_bits() {
        assert_eq!(bits_count(BlsScalar::zero()), 1);
        assert_eq!(bits_count(BlsScalar::one()), 1);
        assert_eq!(bits_count(BlsScalar::from(3u64)), 2);
        let two_pow_128 = BlsScalar::from(2u64).pow(&[128u64, 0, 0, 0]);
        assert_eq!(bits_count(two_pow_128), 129);
    }

    #[test]
    fn correct_complex_rangeproof() -> Result<(), Error> {
        // Generate Composer & Public Parameters
        let pub_params = PublicParameters::setup(1 << 17, &mut rand::thread_rng())
            .expect("Pub Params generation error");
        let (ck, vk) = pub_params
            .trim(1 << 16)
            .expect("Pub Params generation error");

        let complex_rangeproof_gadget = |composer: &mut StandardComposer,
                                         range: BlsScalar,
                                         witness: BlsScalar|
         -> Result<(), Error> {
            let witness_var = composer.add_input(witness);
            let res =
                single_complex_rangeproof_gadget(composer, witness, witness_var, range, None)?;
            // Constraint res to be true, since the range should hold.
            composer.constrain_to_constant(res, BlsScalar::one(), BlsScalar::zero());
            Ok(())
        };

        // ----------------------------------------------
        // 1st Testcase. A value inside of the range should pass.
        // Proving
        let mut prover = Prover::new(b"testing");
        complex_rangeproof_gadget(
            prover.mut_cs(),
            BlsScalar::from(2).pow(&[127u64, 0, 0, 0]),
            BlsScalar::from(2).pow(&[127u64, 0, 0, 0]) - BlsScalar::one(),
        )?;
        prover.preprocess(&ck).expect("Unexpected proving error");
        let proof = prover.prove(&ck).expect("Unexpected proving error");

        // Verification
        let mut verifier = Verifier::new(b"testing");
        complex_rangeproof_gadget(
            verifier.mut_cs(),
            BlsScalar::from(2).pow(&[127u64, 0, 0, 0]),
            BlsScalar::from(2).pow(&[127u64, 0, 0, 0]) - BlsScalar::one(),
        )?;
        verifier.preprocess(&ck).expect("Preprocessing error");
        assert!(verifier
            .verify(&proof, &vk, &vec![BlsScalar::zero()])
            .is_ok());

        // ----------------------------------------------
        // 2nd Testcase. A value bigger than the range should fail.
        // Proving
        prover.clear_witness();
        complex_rangeproof_gadget(
            prover.mut_cs(),
            BlsScalar::from(2).pow(&[127u64, 0, 0, 0]),
            BlsScalar::from(2).pow(&[128u64, 0, 0, 0]) + BlsScalar::one(),
        )?;
        let proof = prover.prove(&ck).expect("Unexpected proving error");

        // Verification
        assert!(verifier
            .verify(&proof, &vk, &vec![BlsScalar::zero()])
            .is_err());

        // ----------------------------------------------
        // 3rd Testcase. A value much bigger than the range should fail.
        // Proving
        prover.clear_witness();
        complex_rangeproof_gadget(
            prover.mut_cs(),
            BlsScalar::from(2).pow(&[127u64, 0, 0, 0]),
            BlsScalar::from(2).pow(&[215u64, 0, 0, 0]) - BlsScalar::one(),
        )?;
        let proof = prover.prove(&ck).expect("Unexpected proving error");

        // Verification
        assert!(verifier
            .verify(&proof, &vk, &vec![BlsScalar::zero()])
            .is_err());
        Ok(())
    }

    #[test]
    fn complete_complex_rangeproof() -> Result<(), Error> {
        // Generate Composer & Public Parameters
        let pub_params = PublicParameters::setup(1 << 17, &mut rand::thread_rng())
            .expect("Pub Params generation error");
        let (ck, vk) = pub_params
            .trim(1 << 16)
            .expect("Pub Params generation error");

        let complex_rangeproof_gadget = |composer: &mut StandardComposer,
                                         lower_bound: BlsScalar,
                                         upper_bound: BlsScalar,
                                         witness: BlsScalar|
         -> Result<(), Error> {
            let witness_var = composer.add_input(witness);
            let res = complete_complex_rangeproof_gadget(
                composer,
                witness,
                witness_var,
                lower_bound,
                upper_bound,
            )?;
            // Constraint res to be true, since the range should hold.
            composer.constrain_to_constant(res, BlsScalar::one(), BlsScalar::zero());
            Ok(())
        };

        // ---------------------------------------------------------
        // 1st case to test should pass since the value is in the range.
        // Proving
        let mut prover = Prover::new(b"testing");
        complex_rangeproof_gadget(
            prover.mut_cs(),
            BlsScalar::from(50_000u64),
            BlsScalar::from(250_000u64),
            BlsScalar::from(50_001u64),
        )?;
        prover.preprocess(&ck).expect("Unexpected proving error");
        let proof = prover.prove(&ck).expect("Unexpected proving error");

        // Verification
        let mut verifier = Verifier::new(b"testing");
        complex_rangeproof_gadget(
            verifier.mut_cs(),
            BlsScalar::from(50_000u64),
            BlsScalar::from(250_000u64),
            BlsScalar::from(50_001u64),
        )?;
        verifier.preprocess(&ck).expect("Preprocessing error");
        assert!(verifier
            .verify(&proof, &vk, &vec![BlsScalar::zero()])
            .is_ok());
        // ---------------------------------------------------------
        // 2nd case should fail since we are below the minimum_range
        prover.clear_witness();
        complex_rangeproof_gadget(
            prover.mut_cs(),
            BlsScalar::from(50_000u64),
            BlsScalar::from(250_000u64),
            BlsScalar::from(18_598u64),
        )?;
        let proof = prover.prove(&ck).expect("Unexpected proving error");
        // Verification
        assert!(verifier
            .verify(&proof, &vk, &vec![BlsScalar::zero()])
            .is_err());

        // ---------------------------------------------------------
        // 3rd case should fail since we are avobe the maximum_range
        prover.clear_witness();
        complex_rangeproof_gadget(
            prover.mut_cs(),
            BlsScalar::from(50_000u64),
            BlsScalar::from(250_000u64),
            BlsScalar::from(250_001u64),
        )?;
        let proof = prover.prove(&ck).expect("Unexpected proving error");
        // Verification
        assert!(verifier
            .verify(&proof, &vk, &vec![BlsScalar::zero()])
            .is_err());
        Ok(())
    }
}

// Call previous gadget -> Variable(sk)

// Let real_sk = composer.add_input(sk_as_scalar);
// composer.assert_equals(sk, real_sk);
// Call_gadget(sk_as_scalar);
// Call gadget needs to return the real_sk
// Then we constrain real_sk to the Variable(sk)
