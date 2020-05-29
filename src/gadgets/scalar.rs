use crate::gadgets::boolean::BoolVar;
use algebra::curves::bls12_381::Bls12_381;
use algebra::fields::bls12_381::fr::Fr;
use algebra::fields::PrimeField;
use num_traits::{One, Zero};
use dusk_plonk::cs::composer::StandardComposer;
use dusk_plonk::cs::constraint_system::Variable;
use rand::RngCore;

/// Conditionally selects the value provided or a zero instead.
/// NOTE that the `select` input has to be previously constrained to
/// be either `one` or `zero`.
/// ## Performs:
/// x' = x if select = 1
/// x' = 0 if select = 0
pub fn conditionally_select_zero(
    composer: &mut StandardComposer<Bls12_381>,
    x: Variable,
    select: Variable,
) -> Variable {
    composer.mul(x, select, Fr::one(), -Fr::one(), Fr::zero(), Fr::zero())
}

/// Conditionally selects the value provided or a one instead.
/// NOTE that the `select` input has to be previously constrained to
/// be either `one` or `zero`.
/// ## Performs:
/// y' = y if bit = 1
/// y' = 1 if bit = 0 =>
/// y' = bit * y + (1 - bit)
pub fn conditionally_select_one(
    composer: &mut StandardComposer<Bls12_381>,
    y: Variable,
    select: BoolVar,
) -> Variable {
    let one = composer.add_input(Fr::one());
    // bit * y
    let bit_y = composer.mul(
        select.into(),
        y,
        Fr::one(),
        -Fr::one(),
        Fr::zero(),
        Fr::zero(),
    );
    // 1 - bit
    let one_min_bit = composer.add(
        one,
        select.into(),
        Fr::one(),
        -Fr::one(),
        -Fr::one(),
        Fr::zero(),
        Fr::zero(),
    );
    // bit * y + (1 - bit)
    composer.add(
        bit_y,
        one_min_bit,
        Fr::one(),
        Fr::one(),
        -Fr::one(),
        Fr::zero(),
        Fr::zero(),
    )
}

/// Adds constraints to the CS which check that a Variable != 0
pub fn is_non_zero(
    composer: &mut StandardComposer<Bls12_381>,
    var: Variable,
    var_assigment: Option<Fr>,
) {
    let one = composer.add_input(Fr::one());
    // XXX: We use this Fq random obtention but we will use the random variable generator
    // that we will include in the PLONK API on the future.
    let inv = match var_assigment {
        Some(fr) => fr,
        // XXX: Should be a randomly generated variable
        None => Fr::from(127u8),
    };
    let inv_var = composer.add_input(inv);
    // Var * Inv(Var) = 1
    composer.poly_gate(
        var,
        inv_var,
        one,
        Fr::one(),
        Fr::zero(),
        Fr::zero(),
        -Fr::one(),
        Fr::zero(),
        Fr::zero(),
    );
}

/// Constraints a `LinearCombination` to be equal to zero or one with:
/// `(1 - a) * a = 0` returning a `BoolVar` that preserves
/// the original Variable value.
pub fn binary_constrain(composer: &mut StandardComposer<Bls12_381>, bit: Variable) -> BoolVar {
    composer.bool_gate(bit);
    BoolVar(bit)
}

mod tests {
    use super::*;
    use ff_fft::EvaluationDomain;
    use merlin::Transcript;
    use dusk_plonk::cs::proof::Proof;
    use dusk_plonk::cs::Composer;
    use dusk_plonk::srs::*;
    use poly_commit::kzg10::{Powers, UniversalParams, VerifierKey};
    use std::str::FromStr;

    fn gen_transcript() -> Transcript {
        Transcript::new(b"TESTING")
    }

    fn prove_binary(
        domain: &EvaluationDomain<Fr>,
        ck: &Powers<Bls12_381>,
        possible_bit: Fr,
    ) -> Proof<Bls12_381> {
        let mut transcript = gen_transcript();
        let mut composer = StandardComposer::new();
        let possible_bit = composer.add_input(possible_bit);
        binary_constrain(&mut composer, possible_bit.into());
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        let preprocessed_circuit = composer.preprocess(&ck, &mut transcript, &domain);
        composer.prove(&ck, &preprocessed_circuit, &mut transcript)
    }

    fn verify_binary(
        domain: &EvaluationDomain<Fr>,
        ck: &Powers<Bls12_381>,
        vk: &VerifierKey<Bls12_381>,
        proof: &Proof<Bls12_381>,
    ) -> bool {
        let mut transcript = gen_transcript();
        let mut composer = StandardComposer::new();
        let possible_bit = composer.add_input(Fr::from_str("56").unwrap());
        binary_constrain(&mut composer, possible_bit.into());
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        let preprocessed_circuit = composer.preprocess(&ck, &mut transcript, &domain);
        proof.verify(
            &preprocessed_circuit,
            &mut transcript,
            vk,
            &vec![Fr::zero(), Fr::zero(), Fr::zero()],
        )
    }

    fn binary_roundtrip_helper(possible_bit: Fr) -> bool {
        let public_parameters = setup(8, &mut rand::thread_rng());
        let (ck, vk) = trim(&public_parameters, 8).unwrap();
        let domain: EvaluationDomain<Fr> = EvaluationDomain::new(8).unwrap();

        let proof = prove_binary(&domain, &ck, possible_bit);
        verify_binary(&domain, &ck, &vk, &proof)
    }

    #[test]
    fn binary_constraint_test() {
        assert!(binary_roundtrip_helper(Fr::zero()));
        assert!(binary_roundtrip_helper(Fr::one()));
        assert!(!binary_roundtrip_helper(Fr::one() + Fr::one()));
    }

    fn prove_cond_select_zero(
        domain: &EvaluationDomain<Fr>,
        ck: &Powers<Bls12_381>,
        num: Fr,
        select: Fr,
        expected: Fr,
    ) -> Proof<Bls12_381> {
        let mut transcript = gen_transcript();
        let mut composer = StandardComposer::new();
        let num = composer.add_input(num);
        let select = composer.add_input(select);
        let select = binary_constrain(&mut composer, select.into());
        let selected = conditionally_select_zero(&mut composer, num.into(), select.into());
        composer.constrain_to_constant(selected, expected, Fr::zero());
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        let preprocessed_circuit = composer.preprocess(&ck, &mut transcript, &domain);
        composer.prove(&ck, &preprocessed_circuit, &mut transcript)
    }

    fn verify_cond_select_zero(
        domain: &EvaluationDomain<Fr>,
        ck: &Powers<Bls12_381>,
        vk: &VerifierKey<Bls12_381>,
        proof: &Proof<Bls12_381>,
        expected: Fr,
    ) -> bool {
        let mut transcript = gen_transcript();
        let mut composer = StandardComposer::new();
        let num = composer.add_input(Fr::from_str("46").unwrap());
        let select = composer.add_input(Fr::from_str("36").unwrap());
        let select = binary_constrain(&mut composer, select.into());
        let selected = conditionally_select_zero(&mut composer, num.into(), select.into());
        composer.constrain_to_constant(selected, expected, Fr::zero());
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        let preprocessed_circuit = composer.preprocess(&ck, &mut transcript, &domain);
        proof.verify(
            &preprocessed_circuit,
            &mut transcript,
            vk,
            &vec![Fr::zero()],
        )
    }

    fn cond_select_zero_roundtrip_helper(num: Fr, selector: Fr, expected: Fr) -> bool {
        let public_parameters = setup(16, &mut rand::thread_rng());
        let (ck, vk) = trim(&public_parameters, 16).unwrap();
        let domain: EvaluationDomain<Fr> = EvaluationDomain::new(16).unwrap();

        let proof = prove_cond_select_zero(&domain, &ck, num, selector, expected);
        verify_cond_select_zero(&domain, &ck, &vk, &proof, expected)
    }

    #[test]
    fn test_conditionally_select_zero() {
        let one = Fr::one();
        let two = one + one;
        let zero = Fr::zero();

        assert!(cond_select_zero_roundtrip_helper(two, one, two));
        assert!(cond_select_zero_roundtrip_helper(two, zero, zero));
    }

    fn prove_cond_select_one(
        domain: &EvaluationDomain<Fr>,
        ck: &Powers<Bls12_381>,
        num: Fr,
        select: Fr,
        expected: Fr,
    ) -> Proof<Bls12_381> {
        let mut transcript = gen_transcript();
        let mut composer = StandardComposer::new();
        let num = composer.add_input(num);
        let select = composer.add_input(select);
        let select = binary_constrain(&mut composer, select.into());
        let selected = conditionally_select_one(&mut composer, num, select.into());
        composer.constrain_to_constant(selected, expected, Fr::zero());
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        let preprocessed_circuit = composer.preprocess(&ck, &mut transcript, &domain);
        composer.prove(&ck, &preprocessed_circuit, &mut transcript)
    }

    fn verify_cond_select_one(
        domain: &EvaluationDomain<Fr>,
        ck: &Powers<Bls12_381>,
        vk: &VerifierKey<Bls12_381>,
        proof: &Proof<Bls12_381>,
        expected: Fr,
    ) -> bool {
        let mut transcript = gen_transcript();
        let mut composer = StandardComposer::new();
        let num = composer.add_input(Fr::from_str("46").unwrap());
        let select = composer.add_input(Fr::from_str("36").unwrap());
        let select = binary_constrain(&mut composer, select.into());
        let selected = conditionally_select_one(&mut composer, num, select.into());
        composer.constrain_to_constant(selected, expected, Fr::zero());
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        let preprocessed_circuit = composer.preprocess(&ck, &mut transcript, &domain);
        proof.verify(
            &preprocessed_circuit,
            &mut transcript,
            vk,
            &vec![Fr::zero()],
        )
    }

    fn cond_select_one_roundtrip_helper(num: Fr, selector: Fr, expected: Fr) -> bool {
        let public_parameters = setup(16, &mut rand::thread_rng());
        let (ck, vk) = trim(&public_parameters, 16).unwrap();
        let domain: EvaluationDomain<Fr> = EvaluationDomain::new(16).unwrap();

        let proof = prove_cond_select_one(&domain, &ck, num, selector, expected);
        verify_cond_select_one(&domain, &ck, &vk, &proof, expected)
    }

    #[test]
    fn test_conditionally_select_one() {
        let one = Fr::one();
        let two = one + one;
        let zero = Fr::zero();

        assert!(cond_select_one_roundtrip_helper(two, one, two));
        assert!(cond_select_one_roundtrip_helper(two, zero, one));
    }

    fn prove_is_non_zero(
        domain: &EvaluationDomain<Fr>,
        ck: &Powers<Bls12_381>,
        num: Fr,
        inv_num: Fr,
    ) -> Proof<Bls12_381> {
        let mut transcript = gen_transcript();
        let mut composer = StandardComposer::new();
        let num = composer.add_input(num);
        is_non_zero(&mut composer, num, Some(inv_num));
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        let preprocessed_circuit = composer.preprocess(&ck, &mut transcript, &domain);
        composer.prove(&ck, &preprocessed_circuit, &mut transcript)
    }

    fn verify_is_non_zero(
        domain: &EvaluationDomain<Fr>,
        ck: &Powers<Bls12_381>,
        vk: &VerifierKey<Bls12_381>,
        proof: &Proof<Bls12_381>,
        num: Fr,
    ) -> bool {
        let mut transcript = gen_transcript();
        let mut composer = StandardComposer::new();
        let num = composer.add_input(Fr::from_str("46").unwrap());
        is_non_zero(&mut composer, num, None);
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        let preprocessed_circuit = composer.preprocess(&ck, &mut transcript, &domain);
        proof.verify(
            &preprocessed_circuit,
            &mut transcript,
            vk,
            &vec![Fr::zero()],
        )
    }

    fn is_non_zero_roundtrip_helper(num: Fr, inv_num: Fr) -> bool {
        let public_parameters = setup(16, &mut rand::thread_rng());
        let (ck, vk) = trim(&public_parameters, 16).unwrap();
        let domain: EvaluationDomain<Fr> = EvaluationDomain::new(16).unwrap();

        let proof = prove_is_non_zero(&domain, &ck, num, inv_num);
        verify_is_non_zero(&domain, &ck, &vk, &proof, num)
    }

    #[test]
    fn test_is_non_zero() {
        let one = Fr::one();
        let three = Fr::from(3u8);
        let zero = Fr::zero();
        use algebra::fields::Field;
        let inv_three = three.inverse().unwrap();

        assert!(is_non_zero_roundtrip_helper(one, one));
        assert!(is_non_zero_roundtrip_helper(three, inv_three));
        assert!(!is_non_zero_roundtrip_helper(zero, one));
    }
}
