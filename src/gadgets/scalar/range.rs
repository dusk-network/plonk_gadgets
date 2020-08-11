//! Collection of range-checking gadgets for Bls12_381 scalars.
use crate::gadgets::GadgetErrors;
use anyhow::{Error, Result};
use dusk_plonk::prelude::*;

/// Builds a complex range-proof (not bounded to a pow_of_two) given a
/// composer, the max range and the witness.
///
/// Checks that `witness < max_range` returning a boolean `Variable`.
pub fn single_complex_range_proof(
    composer: &mut StandardComposer,
    witness: BlsScalar,
    max_range: BlsScalar,
) -> Result<Variable, Error> {
    let num_bits = bits_count(max_range);
    let closest_pow_of_two = BlsScalar::from(2u64).pow(&[num_bits, 0, 0, 0]);
    // Compute b' max range.
    let b_prime = closest_pow_of_two - max_range;
    // Obtain 128-bit representation of `witness + b'`.
    let bits = scalar_to_bits(&(witness + b_prime));

    // Create 0 as witness value
    let zero = composer.add_constant_witness(BlsScalar::zero());

    let mut var_accumulator = zero;

    let accumulator =
        bits[..]
            .iter()
            .enumerate()
            .fold(BlsScalar::zero(), |scalar_accum, (idx, mut bit)| {
                if idx >= num_bits as usize {
                    bit = &0u8;
                };
                let bit_var = composer.add_input(BlsScalar::from(*bit as u64));
                // Apply boolean constraint to the bit.
                composer.boolean_gate(bit_var);
                // Accumulate the bit multiplied by 2^(i-1) as a variable
                var_accumulator = composer.add(
                    (BlsScalar::one(), var_accumulator),
                    (BlsScalar::from(2u64).pow(&[idx as u64, 0, 0, 0]), bit_var),
                    BlsScalar::zero(),
                    BlsScalar::zero(),
                );
                scalar_accum
                    + (BlsScalar::from(*bit as u64)
                        * BlsScalar::from(2u64).pow(&[idx as u64, 0, 0, 0]))
            });
    // Compute `Chi(x)` =  Sum(vi * 2^(i-1)) - (x + b').
    let witness_plus_b_prime = composer.add_input(witness + b_prime);
    // Note that the result will be equal to: `0 (if the reangeproof holds)
    // or any other value if it doesn't.
    let chi_x_var = composer.add(
        (BlsScalar::one(), witness_plus_b_prime),
        (-BlsScalar::one(), var_accumulator),
        BlsScalar::zero(),
        BlsScalar::zero(),
    );
    // It is possible to replace a constraint $\chi(\mathbf{x})=0$ on variables
    // $\mathbf{x}$ with a set of constraints $\psi$ on  new variables
    // $(u,y,z)$ such that $y=1$ if $\chi$ holds and $y=0$ otherwise.
    // We introduce new variables $u,y,z$ that are computed as follows:
    //
    // u &= \chi(\mathbf{x});\\
    // y &=\begin{cases}
    // 0,& \text{if }u\neq 0;\\
    // 1,& \text{if }u=0.
    // \end{cases}\\
    // z&=\begin{cases}
    // 1/u,& \text{if }u\neq 0;\\
    // 0,& \text{if }u=0.
    // \end{cases}
    let u = witness + b_prime - accumulator;
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
    let one = composer.add_input(BlsScalar::one());
    composer.add_gate(
        one,
        chi_x_var,
        zero,
        u,
        -BlsScalar::one(),
        BlsScalar::zero(),
        BlsScalar::zero(),
        BlsScalar::zero(),
    );
    let one_min_y = composer.add(
        (BlsScalar::one(), one),
        (-BlsScalar::one(), y),
        BlsScalar::zero(),
        BlsScalar::zero(),
    );
    let u_times_z = composer.mul(u, one, z, BlsScalar::zero(), BlsScalar::zero());
    composer.assert_equal(one_min_y, u_times_z);
    let y_times_u = composer.mul(u, one, y, BlsScalar::zero(), BlsScalar::zero());
    composer.assert_equal(y_times_u, zero);
    // Constraint the result to be boolean
    composer.boolean_gate(y);
    Ok(y)
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
    use dusk_plonk::jubjub::{AffinePoint, GENERATOR, GENERATOR_NUMS};

    pub(self) fn gen_val_blinder_and_commitment() -> (JubJubScalar, JubJubScalar, AffinePoint) {
        let value = JubJubScalar::from(250_000u64);
        let blinder = JubJubScalar::random(&mut rand::thread_rng());

        let commitment: AffinePoint = AffinePoint::from(
            &(GENERATOR.to_niels() * value) + &(GENERATOR_NUMS.to_niels() * blinder),
        );
        (value, blinder, commitment)
    }

    #[test]
    fn counting_scalar_bits() {
        assert_eq!(bits_count(BlsScalar::zero()), 1);
        assert_eq!(bits_count(BlsScalar::one()), 1);

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

        let complex_rangeproof_gadget = |composer: &mut StandardComposer| -> Result<(), Error> {
            let res = single_complex_range_proof(
                composer,
                BlsScalar::from(2u64).pow(&[127u64, 0, 0, 0]),
                BlsScalar::from(2u64).pow(&[128u64, 0, 0, 0]) - BlsScalar::one(),
            )?;
            // Constraint res to be true, since the range should hold.
            composer.constrain_to_constant(res, BlsScalar::one(), BlsScalar::zero());
            // Since we don't use all of the wires, we set some dummy constraints to
            // avoid Committing to zero polynomials.
            composer.add_dummy_constraints();
            Ok(())
        };
        // Proving
        let mut prover = Prover::new(b"testing");
        complex_rangeproof_gadget(prover.mut_cs())?;
        prover.preprocess(&ck).expect("Unexpected proving error");
        let proof = prover.prove(&ck).expect("Unexpected proving error");

        // Verification
        let mut verifier = Verifier::new(b"testing");
        complex_rangeproof_gadget(verifier.mut_cs())?;
        verifier.preprocess(&ck).expect("Preprocessing error");
        assert!(verifier
            .verify(&proof, &vk, &vec![BlsScalar::zero()])
            .is_ok());
        Ok(())
    }

    #[test]
    fn wrong_complex_rangeproof() -> Result<(), Error> {
        // Generate Composer & Public Parameters
        let pub_params = PublicParameters::setup(1 << 17, &mut rand::thread_rng())
            .expect("Pub Params generation error");
        let (ck, vk) = pub_params
            .trim(1 << 16)
            .expect("Pub Params generation error");

        let complex_rangeproof_gadget = |composer: &mut StandardComposer| -> Result<(), Error> {
            let res = single_complex_range_proof(
                composer,
                BlsScalar::from(2u64).pow(&[130u64, 0, 0, 0]),
                BlsScalar::from(2u64).pow(&[128u64, 0, 0, 0]) - BlsScalar::one(),
            )?;
            // Constraint res to be true, even the range should not hold.
            // That should cause the proof to fail on the verification step.
            composer.constrain_to_constant(res, BlsScalar::one(), BlsScalar::zero());
            // Since we don't use all of the wires, we set some dummy constraints to
            // avoid Committing to zero polynomials.
            composer.add_dummy_constraints();
            Ok(())
        };

        // Proving
        let mut prover = Prover::new(b"testing");
        complex_rangeproof_gadget(prover.mut_cs())?;
        prover.preprocess(&ck).expect("Unexpected Prooving error");
        let proof = prover.prove(&ck).expect("Unexpected Prooving error");

        // Verification
        let mut verifier = Verifier::new(b"testing");
        complex_rangeproof_gadget(verifier.mut_cs())?;
        verifier.preprocess(&ck).expect("Error on preprocessing");
        assert!(verifier
            .verify(&proof, &vk, &vec![BlsScalar::zero()])
            .is_err());

        Ok(())
    }
}
