use crate::gadgets::boolean::BoolVar;
use crate::gadgets::ecc::*;
use crate::gadgets::scalar;
use algebra::fields::PrimeField;
use algebra::ToBytes;
use bls12_381::Bls12_381;
use dusk_plonk::constraint_system::composer::StandardComposer;
use dusk_plonk::constraint_system::Variable;
use jubjub::JubJubProjective;
use jubjub::{fq::Fq, fr::Fr};
use rand::{thread_rng, RngCore};

pub fn sk_knowledge(
    composer: &mut StandardComposer,
    basepoint: &JubJubPointGadget,
    pub_key: &JubJubPointGadget,
    scalar: Option<Fr>,
) {
    // Convert the Scalar into bytes
    // XXX: We use this Fq random obtention but we will use the random variable generator
    // that we will include in the PLONK API on the future.
    let sk = match scalar {
        Some(fr) => fr,
        // XXX: Should be a randomly generated variable
        None => Fr::from(55u8),
    };
    let sk_bits = scalar_to_bits(&sk);

    let committed_vars = sk_bits
        .iter()
        .map(|bit| composer.add_input(Fq::from(*bit)))
        .collect::<Vec<Variable>>();
    let committed_boolvars = committed_vars
        .into_iter()
        .map(|var| scalar::binary_constrain(composer, var.into()))
        .collect::<Vec<BoolVar>>();
    // Compute Basep * sk
    let pk_prime = basepoint.scalar_mul(composer, &committed_boolvars);
    // Constrain pk' == pk
    pk_prime.equal(composer, pub_key);
    // Constraint pk & basep to satisfy the curve eq.
    pk_prime.satisfy_curve_eq(composer);
    basepoint.satisfy_curve_eq(composer);
}

fn is_even(bit: u8) -> bool {
    if bit == 0 {
        return true;
    }
    false
}
/// Turn Scalar into bits
fn scalar_to_bits(scalar: &Fr) -> Vec<u8> {
    let mut bytes = Vec::new();
    scalar.write(&mut bytes).unwrap();
    // Compute bit-array
    let mut j = 0;
    let mut res = [0u8; 256];
    for byte in bytes {
        for i in 0..8 {
            let bit = byte >> i as u8;
            res[j] = !is_even(bit) as u8;
            j += 1;
        }
    }
    res.to_vec()
}

mod test {
    use super::*;
    use algebra::curves::jubjub::{JubJubAffine, JubJubParameters};
    use algebra::curves::models::TEModelParameters;
    use dusk_plonk::constraint_system::{proof::Proof, Composer};
    use dusk_plonk::srs::*;
    use ff_fft::EvaluationDomain;
    use merlin::Transcript;
    use num_traits::identities::{One, Zero};
    use poly_commit::kzg10::{Powers, UniversalParams, VerifierKey};
    use std::ops::{Add, Mul};

    fn gen_transcript() -> Transcript {
        Transcript::new(b"jubjub_ecc_ops")
    }

    // Provides points for a later usage in testing
    fn testing_points() -> (
        JubJubProjective,
        JubJubProjective,
        JubJubProjective,
        JubJubProjective,
        JubJubProjective,
    ) {
        let (x, y) = JubJubParameters::AFFINE_GENERATOR_COEFFS;
        let identity = JubJubProjective::zero();
        let gen = JubJubAffine::new(x, y);
        let two_gen = gen.mul(Fr::from(2u8));
        let gen_p_two_gen = two_gen.add(gen);
        let k_times_gen = gen.mul(Fr::from(127u8));

        (
            identity,
            JubJubProjective::from(gen),
            JubJubProjective::from(two_gen),
            JubJubProjective::from(gen_p_two_gen),
            JubJubProjective::from(k_times_gen),
        )
    }

    fn prove_sk_knowledge(
        domain: &EvaluationDomain<Fq>,
        ck: &Powers<Bls12_381>,
        basep: &JubJubProjective,
        pk: &JubJubProjective,
        scalar: Option<Fr>,
    ) -> Proof<Bls12_381> {
        let mut transcript = gen_transcript();
        let mut composer = StandardComposer::new();
        // Gen Point gadgets
        let basep = JubJubPointGadget::from_point(&mut composer, basep);
        let pk = JubJubPointGadget::from_point(&mut composer, pk);
        // Use sk_knowledge gadget
        sk_knowledge(&mut composer, &basep, &pk, scalar);
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        let preprocessed_circuit = composer.preprocess(&ck, &mut transcript, domain);
        composer.prove(&ck, &preprocessed_circuit, &mut transcript)
    }

    fn verify_sk_knowledge(
        domain: &EvaluationDomain<Fq>,
        ck: &Powers<Bls12_381>,
        vk: &VerifierKey<Bls12_381>,
        proof: &Proof<Bls12_381>,
        basep: &JubJubProjective,
        pk: &JubJubProjective,
        scalar: Option<Fr>,
    ) -> bool {
        let mut transcript = gen_transcript();
        let mut composer = StandardComposer::new();
        // Gen Point gadgets
        let basep = JubJubPointGadget::from_point(&mut composer, basep);
        let pk = JubJubPointGadget::from_point(&mut composer, pk);
        // Use sk_knowledge gadget
        sk_knowledge(&mut composer, &basep, &pk, scalar);
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        let preprocessed_circuit = composer.preprocess(&ck, &mut transcript, &domain);
        proof.verify(
            &preprocessed_circuit,
            &mut transcript,
            vk,
            &vec![Fq::zero()],
        )
    }

    fn sk_knowledge_roundtrip_helper(
        basep: &JubJubProjective,
        pk: &JubJubProjective,
        scalar: Option<Fr>,
    ) -> bool {
        let public_parameters = setup(16384, &mut rand::thread_rng());
        let (ck, vk) = trim(&public_parameters, 16384).unwrap();
        let domain: EvaluationDomain<Fq> = EvaluationDomain::new(16384).unwrap();

        let proof = prove_sk_knowledge(&domain, &ck, basep, pk, scalar);
        verify_sk_knowledge(&domain, &ck, &vk, &proof, basep, pk, None)
    }

    #[test]
    fn test_sk_knowledge() {
        let (_, basep, _, _, pk) = testing_points();
        assert!(sk_knowledge_roundtrip_helper(
            &basep,
            &pk,
            Some(Fr::from(127u8))
        ));
    }
}
