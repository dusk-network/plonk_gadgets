// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (c) DUSK NETWORK. All rights reserved.

extern crate anyhow;
extern crate dusk_plonk;
extern crate plonk_gadgets;

#[cfg(test)]
mod tests {
    use anyhow::{Error, Result};
    use dusk_plonk::prelude::*;
    use plonk_gadgets::AllocatedScalar;
    use plonk_gadgets::RangeGadgets::*;

    fn max_bound_gadget(
        composer: &mut StandardComposer,
        max_range: BlsScalar,
        witness: BlsScalar,
        result: bool,
    ) {
        let witness = AllocatedScalar::allocate(composer, witness);
        let (res, _) = max_bound(composer, max_range, witness);

        let mut outcome = BlsScalar::zero();
        if result {
            outcome = BlsScalar::one()
        }
        composer.constrain_to_constant(res, outcome, None);
    }

    fn range_check_gadget(
        composer: &mut StandardComposer,
        max_range: BlsScalar,
        min_range: BlsScalar,
        witness: BlsScalar,
        result: bool,
    ) {
        let witness = AllocatedScalar::allocate(composer, witness);
        let res = range_check(composer, min_range, max_range, witness);

        let mut outcome = BlsScalar::zero();
        if result {
            outcome = BlsScalar::one()
        }
        composer.constrain_to_constant(res, outcome, None);
    }

    #[test]
    fn max_bound_test() -> Result<(), Error> {
        // Generate Composer & Public Parameters
        let pub_params = PublicParameters::setup(1 << 11, &mut rand::thread_rng())?;
        let (ck, vk) = pub_params.trim(1 << 10)?;

        struct TestCase {
            max_range: BlsScalar,
            witness: BlsScalar,
            expected_result: bool,
        }
        let test_cases = vec![
            TestCase {
                max_range: BlsScalar::from(2u64).pow(&[128u64, 0, 0, 0]) - BlsScalar::one(),
                witness: BlsScalar::from(2u64).pow(&[127u64, 0, 0, 0]),
                expected_result: true,
            },
            TestCase {
                max_range: BlsScalar::from(200u64),
                witness: BlsScalar::from(100u64),
                expected_result: true,
            },
            TestCase {
                max_range: BlsScalar::from(100u64),
                witness: BlsScalar::from(200u64),
                expected_result: false,
            },
            TestCase {
                max_range: BlsScalar::from(2u64).pow(&[128u64, 0, 0, 0]) - BlsScalar::one(),
                witness: BlsScalar::from(2u64).pow(&[130u64, 0, 0, 0]),
                expected_result: false,
            },
        ];

        for case in test_cases.into_iter() {
            // Proving
            let mut prover = Prover::default();

            max_bound_gadget(
                prover.mut_cs(),
                case.max_range,
                case.witness,
                case.expected_result,
            );
            prover.preprocess(&ck)?;
            let proof = prover.prove(&ck)?;

            // Verification
            let mut verifier = Verifier::default();
            max_bound_gadget(
                verifier.mut_cs(),
                case.max_range,
                case.witness,
                case.expected_result,
            );
            verifier.preprocess(&ck)?;
            assert!(verifier
                .verify(&proof, &vk, &vec![BlsScalar::zero()])
                .is_ok());
        }
        Ok(())
    }
    #[test]
    fn range_check_test() -> Result<(), Error> {
        // Generate Composer & Public Parameters
        let pub_params = PublicParameters::setup(1 << 11, &mut rand::thread_rng())?;
        let (ck, vk) = pub_params.trim(1 << 10)?;

        struct TestCase {
            max_range: BlsScalar,
            min_range: BlsScalar,
            witness: BlsScalar,
            expected_result: bool,
        }
        let test_cases = vec![
            TestCase {
                min_range: BlsScalar::from(50_000u64),
                max_range: BlsScalar::from(250_000u64),
                witness: BlsScalar::from(50_001u64),
                expected_result: true,
            },
            TestCase {
                min_range: BlsScalar::from(50_000u64),
                max_range: BlsScalar::from(250_000u64),
                witness: BlsScalar::from(250_001u64),
                expected_result: false,
            },
            TestCase {
                min_range: BlsScalar::from(50_000u64),
                max_range: BlsScalar::from(250_000u64),
                witness: BlsScalar::from(250_000u64),
                expected_result: false,
            },
            TestCase {
                min_range: BlsScalar::from(50_000u64),
                max_range: BlsScalar::from(250_000u64),
                witness: BlsScalar::from(249_000u64),
                expected_result: true,
            },
            TestCase {
                min_range: BlsScalar::from(50_000u64),
                max_range: BlsScalar::from(250_000u64),
                witness: BlsScalar::from(50_000u64),
                expected_result: true,
            },
            TestCase {
                min_range: BlsScalar::from(50_000u64),
                max_range: BlsScalar::from(250_000u64),
                witness: BlsScalar::from(49_999u64),
                expected_result: false,
            },
            TestCase {
                min_range: BlsScalar::from(2u64).pow(&[126u64, 0, 0, 0]),
                max_range: BlsScalar::from(2u64).pow(&[127u64, 0, 0, 0]) + BlsScalar::one(),
                witness: BlsScalar::from(2u64).pow(&[127u64, 0, 0, 0]) - BlsScalar::one(),
                expected_result: true,
            },
            TestCase {
                min_range: BlsScalar::from(50_000u64),
                max_range: BlsScalar::from(250_000u64),
                witness: BlsScalar::from(18_598u64),
                expected_result: false,
            },
        ];

        for case in test_cases.into_iter() {
            // Proving
            let mut prover = Prover::default();

            range_check_gadget(
                prover.mut_cs(),
                case.max_range,
                case.min_range,
                case.witness,
                case.expected_result,
            );
            prover.preprocess(&ck)?;
            let proof = prover.prove(&ck)?;

            // Verification
            let mut verifier = Verifier::default();
            range_check_gadget(
                verifier.mut_cs(),
                case.max_range,
                case.min_range,
                case.witness,
                case.expected_result,
            );
            verifier.preprocess(&ck)?;
            assert!(verifier
                .verify(&proof, &vk, &vec![BlsScalar::zero()])
                .is_ok());
        }

        Ok(())
    }
}
