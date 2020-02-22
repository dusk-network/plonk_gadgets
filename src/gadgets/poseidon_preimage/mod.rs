use crate::{helpers, Curve, Proof, Scalar, StandardComposer};

use ff_fft::EvaluationDomain;
use merlin::Transcript;
use plonk::cs::Composer;
use poly_commit::kzg10::{Powers, VerifierKey};

pub fn poseidon(x: Scalar) -> Scalar {
    // TODO - Implement Poseidon
    let x = x + &Scalar::from(2u64);
    let x = x * &Scalar::from(3u64);
    let x = x + &Scalar::from(5u64);
    x
}

pub fn gen_transcript() -> Transcript {
    Transcript::new(b"poseidon-plonk")
}

pub fn poseidon_gadget(composer: &mut StandardComposer, x: Option<Scalar>, h: Scalar) {
    let a = composer.add_input(x.unwrap_or_default());
    let b = composer.add_input(Scalar::from(2u64));
    let o = helpers::add_gate(composer, a, b);

    let b = composer.add_input(Scalar::from(3u64));
    let o = helpers::mul_gate(composer, o, b);

    let b = composer.add_input(Scalar::from(5u64));
    let o = helpers::add_gate(composer, o, b);

    helpers::constrain_gate(composer, o, h);
}

pub fn prove(domain: &EvaluationDomain<Scalar>, ck: &Powers<Curve>, x: Scalar, h: Scalar) -> Proof {
    let mut transcript = gen_transcript();
    let mut composer = StandardComposer::new();

    poseidon_gadget(&mut composer, Some(x), h);
    composer.add_dummy_constraints();
    composer.add_dummy_constraints();
    composer.add_dummy_constraints();

    let preprocessed_circuit = composer.preprocess(&ck, &mut transcript, &domain);
    composer.prove(&ck, &preprocessed_circuit, &mut transcript)
}

pub fn verify(
    domain: &EvaluationDomain<Scalar>,
    ck: &Powers<Curve>,
    vk: &VerifierKey<Curve>,
    proof: &Proof,
    h: Scalar,
) -> bool {
    let mut transcript = gen_transcript();
    let mut composer = StandardComposer::new();

    poseidon_gadget(&mut composer, None, h);
    composer.add_dummy_constraints();
    composer.add_dummy_constraints();
    composer.add_dummy_constraints();

    let preprocessed_circuit = composer.preprocess(&ck, &mut transcript, &domain);
    proof.verify(
        &preprocessed_circuit,
        &mut transcript,
        vk,
        composer.public_inputs(),
    )
}

#[cfg(test)]
mod tests {
    use crate::Scalar;

    use ff_fft::EvaluationDomain;
    use plonk::srs;

    #[test]
    fn poseidon_det() {
        let x = Scalar::from(17u64);
        let y = Scalar::from(17u64);
        let z = Scalar::from(19u64);

        let a = super::poseidon(x);
        let b = super::poseidon(y);
        let c = super::poseidon(z);

        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn poseidon_preimage() {
        // Trusted setup
        // TODO - Create a trusted setup struct
        let public_parameters = srs::setup(32);
        let (ck, vk) = srs::trim(&public_parameters, 32).unwrap();
        let domain: EvaluationDomain<Scalar> = EvaluationDomain::new(10).unwrap();

        let x = Scalar::from(31u64);
        let h = super::poseidon(x);

        let y = Scalar::from(30u64);
        let i = super::poseidon(y);

        let proof = super::prove(&domain, &ck, x, h);
        assert!(super::verify(&domain, &ck, &vk, &proof, h));

        let proof = super::prove(&domain, &ck, y, i);
        assert!(super::verify(&domain, &ck, &vk, &proof, i));

        // Wrong pre-image
        let wrong_proof = super::prove(&domain, &ck, y, h);
        assert!(!super::verify(&domain, &ck, &vk, &wrong_proof, h));

        // Wrong public image
        let wrong_proof = super::prove(&domain, &ck, x, i);
        assert!(!super::verify(&domain, &ck, &vk, &wrong_proof, i));

        // Inconsistent public image
        let proof = super::prove(&domain, &ck, x, h);
        assert!(!super::verify(&domain, &ck, &vk, &proof, i));
    }
}
