use crate::gadgets::boolean::BoolVar;
use dusk_bls12_381::Scalar;
use dusk_plonk::constraint_system::composer::StandardComposer;
use dusk_plonk::constraint_system::Variable;

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
    composer.mul(Scalar::one(), x, select, Scalar::zero(), Scalar::zero())
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
    select: BoolVar,
) -> Variable {
    let one = composer.add_input(Scalar::one());
    // bit * y
    let bit_y = composer.mul(
        Scalar::one(),
        select.into(),
        y,
        Scalar::zero(),
        Scalar::zero(),
    );
    // 1 - bit
    let one_min_bit = composer.add(
        (Scalar::one(), one),
        (-Scalar::one(), select.into()),
        Scalar::zero(),
        Scalar::zero(),
    );
    // bit * y + (1 - bit)
    composer.add(
        (Scalar::one(), bit_y),
        (Scalar::one(), one_min_bit),
        Scalar::zero(),
        Scalar::zero(),
    )
}

/// Adds constraints to the CS which check that a Variable != 0
pub fn is_non_zero(composer: &mut StandardComposer, var: Variable, var_assigment: Option<Scalar>) {
    let one = composer.add_input(Scalar::one());
    // XXX: We use this Fq random obtention but we will use the random variable generator
    // that we will include in the PLONK API on the future.
    let inv = match var_assigment {
        Some(fr) => fr,
        // XXX: Should be a randomly generated variable
        None => Scalar::from(127u64),
    };
    let inv_var = composer.add_input(inv);
    // Var * Inv(Var) = 1
    composer.poly_gate(
        var,
        inv_var,
        one,
        Scalar::one(),
        Scalar::zero(),
        Scalar::zero(),
        -Scalar::one(),
        Scalar::zero(),
        Scalar::zero(),
    );
}

/// Constraints a `LinearCombination` to be equal to zero or one with:
/// `(1 - a) * a = 0` returning a `BoolVar` that preserves
/// the original Variable value.
pub fn binary_constrain(composer: &mut StandardComposer, bit: Variable) -> BoolVar {
    composer.bool_gate(bit);
    BoolVar(bit)
}

mod tests {
    use super::*;
    use dusk_plonk::commitment_scheme::kzg10::srs::*;
    use dusk_plonk::commitment_scheme::kzg10::{ProverKey, VerifierKey};
    use dusk_plonk::constraint_system::composer::StandardComposer;
    use dusk_plonk::fft::EvaluationDomain;
    use dusk_plonk::proof_system::Proof;
    use merlin::Transcript;

    fn gen_transcript() -> Transcript {
        Transcript::new(b"TESTING")
    }

    fn prove_binary(domain: &EvaluationDomain, ck: &ProverKey, possible_bit: Scalar) -> Proof {
        let mut transcript = gen_transcript();
        let mut composer = StandardComposer::new();
        let possible_bit = composer.add_input(possible_bit);
        binary_constrain(&mut composer, possible_bit.into());
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        let preprocessed_circuit = composer.preprocess(ck, &mut transcript, &domain);
        composer.prove(ck, &preprocessed_circuit, &mut transcript)
    }

    fn verify_binary(
        domain: &EvaluationDomain,
        ck: &ProverKey,
        vk: &VerifierKey,
        proof: &Proof,
    ) -> bool {
        let mut transcript = gen_transcript();
        let mut composer = StandardComposer::new();
        let possible_bit = composer.add_input(Scalar::from(56u64));
        binary_constrain(&mut composer, possible_bit.into());
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        let preprocessed_circuit = composer.preprocess(&ck, &mut transcript, &domain);
        proof.verify(
            &preprocessed_circuit,
            &mut transcript,
            vk,
            &vec![Scalar::zero(), Scalar::zero(), Scalar::zero()],
        )
    }

    fn binary_roundtrip_helper(possible_bit: Scalar) -> bool {
        let public_parameters = PublicParameters::setup(8, &mut rand::thread_rng()).unwrap();
        let (ck, vk) = public_parameters.trim(8).unwrap();
        let domain: EvaluationDomain = EvaluationDomain::new(8).unwrap();

        let proof = prove_binary(&domain, &ck, possible_bit);
        verify_binary(&domain, &ck, &vk, &proof)
    }

    #[test]
    fn binary_constraint_test() {
        assert!(binary_roundtrip_helper(Scalar::zero()));
        assert!(binary_roundtrip_helper(Scalar::one()));
        assert!(!binary_roundtrip_helper(Scalar::one() + Scalar::one()));
    }

    fn prove_cond_select_zero(
        domain: &EvaluationDomain,
        ck: &ProverKey,
        num: Scalar,
        select: Scalar,
        expected: Scalar,
    ) -> Proof {
        let mut transcript = gen_transcript();
        let mut composer = StandardComposer::new();
        let num = composer.add_input(num);
        let select = composer.add_input(select);
        let select = binary_constrain(&mut composer, select.into());
        let selected = conditionally_select_zero(&mut composer, num.into(), select.into());
        composer.constrain_to_constant(selected, expected, Scalar::zero());
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        let preprocessed_circuit = composer.preprocess(&ck, &mut transcript, &domain);
        composer.prove(&ck, &preprocessed_circuit, &mut transcript)
    }

    fn verify_cond_select_zero(
        domain: &EvaluationDomain,
        ck: &ProverKey,
        vk: &VerifierKey,
        proof: &Proof,
        expected: Scalar,
    ) -> bool {
        let mut transcript = gen_transcript();
        let mut composer = StandardComposer::new();
        let num = composer.add_input(Scalar::from(46u64));
        let select = composer.add_input(Scalar::from(36u64));
        let select = binary_constrain(&mut composer, select.into());
        let selected = conditionally_select_zero(&mut composer, num.into(), select.into());
        composer.constrain_to_constant(selected, expected, Scalar::zero());
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        let preprocessed_circuit = composer.preprocess(&ck, &mut transcript, &domain);
        proof.verify(
            &preprocessed_circuit,
            &mut transcript,
            vk,
            &vec![Scalar::zero()],
        )
    }

    fn cond_select_zero_roundtrip_helper(num: Scalar, selector: Scalar, expected: Scalar) -> bool {
        let public_parameters = PublicParameters::setup(16, &mut rand::thread_rng()).unwrap();
        let (ck, vk) = public_parameters.trim(16).unwrap();
        let domain: EvaluationDomain = EvaluationDomain::new(16).unwrap();

        let proof = prove_cond_select_zero(&domain, &ck, num, selector, expected);
        verify_cond_select_zero(&domain, &ck, &vk, &proof, expected)
    }

    #[test]
    fn test_conditionally_select_zero() {
        let one = Scalar::one();
        let two = one + one;
        let zero = Scalar::zero();

        assert!(cond_select_zero_roundtrip_helper(two, one, two));
        assert!(cond_select_zero_roundtrip_helper(two, zero, zero));
    }

    fn prove_cond_select_one(
        domain: &EvaluationDomain,
        ck: &ProverKey,
        num: Scalar,
        select: Scalar,
        expected: Scalar,
    ) -> Proof {
        let mut transcript = gen_transcript();
        let mut composer = StandardComposer::new();
        let num = composer.add_input(num);
        let select = composer.add_input(select);
        let select = binary_constrain(&mut composer, select.into());
        let selected = conditionally_select_one(&mut composer, num, select.into());
        composer.constrain_to_constant(selected, expected, Scalar::zero());
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        let preprocessed_circuit = composer.preprocess(&ck, &mut transcript, &domain);
        composer.prove(&ck, &preprocessed_circuit, &mut transcript)
    }

    fn verify_cond_select_one(
        domain: &EvaluationDomain,
        ck: &ProverKey,
        vk: &VerifierKey,
        proof: &Proof,
        expected: Scalar,
    ) -> bool {
        let mut transcript = gen_transcript();
        let mut composer = StandardComposer::new();
        let num = composer.add_input(Scalar::from(46u64));
        let select = composer.add_input(Scalar::from(36u64));
        let select = binary_constrain(&mut composer, select.into());
        let selected = conditionally_select_one(&mut composer, num, select.into());
        composer.constrain_to_constant(selected, expected, Scalar::zero());
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        let preprocessed_circuit = composer.preprocess(&ck, &mut transcript, &domain);
        proof.verify(
            &preprocessed_circuit,
            &mut transcript,
            vk,
            &vec![Scalar::zero()],
        )
    }

    fn cond_select_one_roundtrip_helper(num: Scalar, selector: Scalar, expected: Scalar) -> bool {
        let public_parameters = PublicParameters::setup(16, &mut rand::thread_rng()).unwrap();
        let (ck, vk) = public_parameters.trim(16).unwrap();
        let domain: EvaluationDomain = EvaluationDomain::new(16).unwrap();

        let proof = prove_cond_select_one(&domain, &ck, num, selector, expected);
        verify_cond_select_one(&domain, &ck, &vk, &proof, expected)
    }

    #[test]
    fn test_conditionally_select_one() {
        let one = Scalar::one();
        let two = one + one;
        let zero = Scalar::zero();

        assert!(cond_select_one_roundtrip_helper(two, one, two));
        assert!(cond_select_one_roundtrip_helper(two, zero, one));
    }

    fn prove_is_non_zero(
        domain: &EvaluationDomain,
        ck: &ProverKey,
        num: Scalar,
        inv_num: Scalar,
    ) -> Proof {
        let mut transcript = gen_transcript();
        let mut composer = StandardComposer::new();
        let num = composer.add_input(num);
        is_non_zero(&mut composer, num, Some(inv_num));
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        let preprocessed_circuit = composer.preprocess(ck, &mut transcript, &domain);
        composer.prove(ck, &preprocessed_circuit, &mut transcript)
    }

    fn verify_is_non_zero(
        domain: &EvaluationDomain,
        ck: &ProverKey,
        vk: &VerifierKey,
        proof: &Proof,
        num: Scalar,
    ) -> bool {
        let mut transcript = gen_transcript();
        let mut composer = StandardComposer::new();
        let num = composer.add_input(Scalar::from(46u64));
        is_non_zero(&mut composer, num, None);
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        let preprocessed_circuit = composer.preprocess(ck, &mut transcript, domain);
        proof.verify(
            &preprocessed_circuit,
            &mut transcript,
            vk,
            &vec![Scalar::zero()],
        )
    }

    fn is_non_zero_roundtrip_helper(num: Scalar, inv_num: Scalar) -> bool {
        let public_parameters = PublicParameters::setup(16, &mut rand::thread_rng()).unwrap();
        let (ck, vk) = public_parameters.trim(16).unwrap();
        let domain: EvaluationDomain = EvaluationDomain::new(16).unwrap();

        let proof = prove_is_non_zero(&domain, &ck, num, inv_num);
        verify_is_non_zero(&domain, &ck, &vk, &proof, num)
    }

    #[test]
    fn test_is_non_zero() {
        let one = Scalar::one();
        let three = Scalar::from(3u8);
        let zero = Scalar::zero();
        use algebra::fields::Field;
        let inv_three = three.inverse().unwrap();

        assert!(is_non_zero_roundtrip_helper(one, one));
        assert!(is_non_zero_roundtrip_helper(three, inv_three));
        assert!(!is_non_zero_roundtrip_helper(zero, one));
    }
}
