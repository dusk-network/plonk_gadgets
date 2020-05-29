use crate::gadgets::boolean::BoolVar;
use crate::gadgets::ecc::*;
use crate::gadgets::scalar;
use dusk_bls12_381::Scalar;
use dusk_plonk::constraint_system::composer::StandardComposer;
use dusk_plonk::constraint_system::Variable;
use jubjub::JubJubProjective;
use jubjub::{Fq, Fr};

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
        None => Fr::from(55u64),
    };
    let sk_bits = scalar_to_bits(&sk);

    let committed_vars = sk_bits
        .iter()
        .map(|bit| composer.add_input(Scalar::from(*bit as u64)))
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
    use dusk_plonk::commitment_scheme::kzg10::srs::*;
    use dusk_plonk::commitment_scheme::kzg10::{ProverKey, VerifierKey};
    use dusk_plonk::constraint_system::StandardComposer;
    use dusk_plonk::fft::EvaluationDomain;
    use dusk_plonk::proof_system::Proof;
    use jubjub::{JubJubAffine, JubJubParameters};
    use merlin::Transcript;

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
        let two_gen = gen.mul(Fr::from(2u64));
        let gen_p_two_gen = two_gen.add(gen);
        let k_times_gen = gen.mul(Fr::from(127u64));

        (
            identity,
            JubJubProjective::from(gen),
            JubJubProjective::from(two_gen),
            JubJubProjective::from(gen_p_two_gen),
            JubJubProjective::from(k_times_gen),
        )
    }

    fn prove_sk_knowledge(
        domain: &EvaluationDomain,
        ck: &ProverKey,
        basep: &JubJubProjective,
        pk: &JubJubProjective,
        scalar: Option<Fr>,
    ) -> Proof {
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
        domain: &EvaluationDomain,
        ck: &ProverKey,
        vk: &VerifierKey,
        proof: &Proof,
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
        let public_parameters = PublicParameters::setup(16384, &mut rand::thread_rng()).unwrap();
        let (ck, vk) = public_parameters.trim(16384).unwrap();
        let domain: EvaluationDomain = EvaluationDomain::new(16384).unwrap();

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
