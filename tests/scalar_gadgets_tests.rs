extern crate anyhow;
extern crate dusk_plonk;
extern crate plonk_gadgets;

#[cfg(test)]
mod tests {
    use anyhow::{Error, Result};
    use dusk_plonk::prelude::*;
    use plonk_gadgets::AllocatedScalar;
    use plonk_gadgets::ScalarGadgets::*;

    #[test]
    fn test_maybe_equal() -> Result<(), Error> {
        // Generate Composer & Public Parameters
        let pub_params = PublicParameters::setup(1 << 10, &mut rand::thread_rng())?;
        let (ck, vk) = pub_params.trim(1 << 9)?;

        let is_equal_gadget =
            |composer: &mut StandardComposer, num_1: u64, num_2: u64, result: bool| {
                let a = AllocatedScalar::allocate(composer, BlsScalar::from(num_1));
                let b = AllocatedScalar::allocate(composer, BlsScalar::from(num_2));

                let bit = maybe_equal(composer, a, b);

                let mut outcome = BlsScalar::zero();
                if result {
                    outcome = BlsScalar::one()
                }
                composer.constrain_to_constant(bit, outcome, BlsScalar::zero());
            };

        // Proving
        // Should pass as 100 == 100
        let mut prover = Prover::new(b"testing");
        is_equal_gadget(prover.mut_cs(), 100, 100, true);

        prover.preprocess(&ck)?;
        let proof = prover.prove(&ck)?;

        // Verification
        let mut verifier = Verifier::new(b"testing");
        is_equal_gadget(verifier.mut_cs(), 0, 0, true);

        verifier.preprocess(&ck).expect("Preprocessing error");
        assert!(verifier
            .verify(&proof, &vk, &vec![BlsScalar::zero()])
            .is_ok());

        // Proving
        // Should fail as 20 != 3330
        let mut prover = Prover::new(b"testing");
        is_equal_gadget(prover.mut_cs(), 20, 3330, false);

        prover.preprocess(&ck)?;
        let proof = prover.prove(&ck)?;

        // Verification
        let mut verifier = Verifier::new(b"testing");
        is_equal_gadget(verifier.mut_cs(), 0, 0, false);

        verifier.preprocess(&ck)?;
        assert!(verifier
            .verify(&proof, &vk, &vec![BlsScalar::zero()])
            .is_ok());

        Ok(())
    }

    #[test]
    fn test_conditionally_select_0() -> Result<(), Error> {
        // The circuit closure runs the conditionally_select_zero fn and constraints the result
        // to actually be 0
        let circuit = |composer: &mut StandardComposer, value: BlsScalar, selector: BlsScalar| {
            let value = composer.add_input(value);
            let selector = composer.add_input(selector);
            let res = conditionally_select_zero(composer, value, selector);
            composer.constrain_to_constant(res, BlsScalar::zero(), BlsScalar::zero());
        };

        // Generate Composer & Public Parameters
        let pub_params = PublicParameters::setup(1 << 8, &mut rand::thread_rng())?;
        let (ck, vk) = pub_params.trim(1 << 7)?;

        // Selector set to 0 should select 0
        // Proving
        let mut prover = Prover::new(b"testing");
        circuit(
            prover.mut_cs(),
            BlsScalar::random(&mut rand::thread_rng()),
            BlsScalar::zero(),
        );
        prover.preprocess(&ck)?;
        let proof = prover.prove(&ck)?;

        // Verification
        let mut verifier = Verifier::new(b"testing");
        circuit(
            verifier.mut_cs(),
            BlsScalar::random(&mut rand::thread_rng()),
            BlsScalar::zero(),
        );
        verifier.preprocess(&ck)?;
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
        let proof = prover.prove(&ck)?;
        // This shouldn't pass since we sent 1 as selector and the circuit closure is constraining the
        // result to be equal to 0 while the value assigned is indeed the randomly generated one.
        assert!(verifier.verify(&proof, &vk, &[BlsScalar::zero()]).is_err());

        Ok(())
    }

    #[test]
    fn test_conditionally_select_1() -> Result<(), Error> {
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
        let pub_params = PublicParameters::setup(1 << 8, &mut rand::thread_rng())?;
        let (ck, vk) = pub_params.trim(1 << 7)?;

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
        prover.preprocess(&ck)?;
        let proof = prover.prove(&ck)?;

        // Verification
        let mut verifier = Verifier::new(b"testing");
        circuit(
            verifier.mut_cs(),
            BlsScalar::random(&mut rand::thread_rng()),
            BlsScalar::zero(),
            BlsScalar::one(),
        );
        verifier.preprocess(&ck)?;
        // This should pass since we sent 0 as selector and the circuit should then return
        // 1.
        assert!(verifier.verify(&proof, &vk, &pi).is_ok());

        // Selector set to 1 should assign the randomly-generated value.
        // Proving
        prover.clear_witness();
        let rand = BlsScalar::random(&mut rand::thread_rng());
        circuit(prover.mut_cs(), rand, BlsScalar::one(), rand);
        pi = prover.mut_cs().public_inputs.clone();
        let proof = prover.prove(&ck)?;
        // This should pass since we sent 1 as selector and the circuit closure should assign the randomly-generated
        // value as a result.
        verifier.verify(&proof, &vk, &pi)
    }

    #[test]
    fn test_is_not_zero() -> Result<(), Error> {
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
        let pub_params = PublicParameters::setup(1 << 8, &mut rand::thread_rng())?;
        let (ck, vk) = pub_params.trim(1 << 7)?;

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
        prover.preprocess(&ck)?;
        let proof = prover.prove(&ck)?;

        // Verification
        let mut verifier = Verifier::new(b"testing");
        circuit(
            verifier.mut_cs(),
            BlsScalar::random(&mut rand::thread_rng()),
            BlsScalar::random(&mut rand::thread_rng()),
        )?;
        verifier.preprocess(&ck)?;
        assert!(verifier.verify(&proof, &vk, &pi).is_err());

        // Value & value assigned set correctly and != 0. This should pass.
        // Proving
        prover.clear_witness();
        let rand = BlsScalar::random(&mut rand::thread_rng());
        circuit(prover.mut_cs(), rand, rand)?;
        pi = prover.mut_cs().public_inputs.clone();
        let proof = prover.prove(&ck)?;
        // This should pass since we sent 1 as selector and the circuit closure should assign the randomly-generated
        // value as a result.
        verifier.verify(&proof, &vk, &pi)
    }
}
