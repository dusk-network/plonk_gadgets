use crate::{helpers, Curve, Field, PreProcessedCircuit, Proof, Scalar, StandardComposer};

use ff_fft::EvaluationDomain;
use merlin::Transcript;
use plonk::cs::Composer;
use poly_commit::kzg10::{Powers, VerifierKey};

const POSEIDON_WIDTH: usize = 5;
const FULL_ROUNDS: usize = 8;
const PARTIAL_ROUNDS: usize = 59;
const BITFLAGS: u8 = 0b1000;

lazy_static::lazy_static! {
    // TODO - Generate constants and MDS
    static ref ROUND_CONSTANTS: [Scalar; 960] = [Scalar::one(); 960];
    static ref MDS: [[Scalar; POSEIDON_WIDTH]; POSEIDON_WIDTH] = [[Scalar::from(17u64); POSEIDON_WIDTH]; POSEIDON_WIDTH];
}

pub fn poseidon(x: Scalar) -> Scalar {
    let mut input = [Scalar::zero(); POSEIDON_WIDTH];
    input[0] = Scalar::from(BITFLAGS);
    input[1] = x;

    let mut constants_offset = 0;

    for _ in 0..FULL_ROUNDS / 2 {
        input.iter_mut().for_each(|i| {
            *i += &ROUND_CONSTANTS[constants_offset];

            let j = *i;
            for _ in 0..5 {
                *i *= &j;
            }

            constants_offset += 1;
        });

        let mut product = [Scalar::zero(); POSEIDON_WIDTH];
        for j in 0..POSEIDON_WIDTH {
            for k in 0..POSEIDON_WIDTH {
                product[j] += &(MDS[j][k] * &input[k]);
            }
        }

        input.copy_from_slice(&product);
    }

    for _ in 0..PARTIAL_ROUNDS {
        input.iter_mut().for_each(|i| {
            *i += &ROUND_CONSTANTS[constants_offset];
            constants_offset += 1;
        });

        let j = input[POSEIDON_WIDTH - 1];
        for _ in 0..5 {
            input[POSEIDON_WIDTH - 1] *= &j;
        }

        let mut product = [Scalar::zero(); POSEIDON_WIDTH];
        for j in 0..POSEIDON_WIDTH {
            for k in 0..POSEIDON_WIDTH {
                product[j] += &(MDS[j][k] * &input[k]);
            }
        }

        input.copy_from_slice(&product);
    }

    for _ in 0..FULL_ROUNDS / 2 {
        input.iter_mut().for_each(|i| {
            *i += &ROUND_CONSTANTS[constants_offset];

            let j = *i;
            for _ in 0..5 {
                *i *= &j;
            }

            constants_offset += 1;
        });

        let mut product = [Scalar::zero(); POSEIDON_WIDTH];
        for j in 0..POSEIDON_WIDTH {
            for k in 0..POSEIDON_WIDTH {
                product[j] += &(MDS[j][k] * &input[k]);
            }
        }

        input.copy_from_slice(&product);
    }

    input[1]
}

pub fn gen_transcript() -> Transcript {
    Transcript::new(b"poseidon-plonk")
}

pub fn poseidon_gadget(composer: &mut StandardComposer, x: Option<Scalar>, h: Scalar) {
    let mut input = [Scalar::zero(); POSEIDON_WIDTH];
    input[0] = Scalar::from(BITFLAGS);
    input[1] = x.unwrap_or_default();

    let zero = composer.add_input(Scalar::zero());
    let mut buf = [zero; POSEIDON_WIDTH];

    buf.iter_mut()
        .enumerate()
        .for_each(|(i, item)| *item = composer.add_input(input[i]));

    let mut constants_offset = 0;

    for _ in 0..FULL_ROUNDS / 2 {
        buf.iter_mut().for_each(|i| {
            let b = composer.add_input(ROUND_CONSTANTS[constants_offset]);
            let mut o = helpers::add_gate(composer, *i, b);

            let j = o;
            for _ in 0..5 {
                o = helpers::mul_gate(composer, o, j);
            }

            *i = o;
            constants_offset += 1;
        });

        let mut product = [zero; POSEIDON_WIDTH];
        for j in 0..POSEIDON_WIDTH {
            for k in 0..POSEIDON_WIDTH {
                let a = composer.add_input(MDS[j][k]);
                let o = helpers::mul_gate(composer, a, buf[k]);
                let o = helpers::add_gate(composer, product[j], o);
                product[j] = o;
            }
        }

        buf.copy_from_slice(&product);
    }

    for _ in 0..PARTIAL_ROUNDS {
        buf.iter_mut().for_each(|i| {
            let b = composer.add_input(ROUND_CONSTANTS[constants_offset]);
            *i = helpers::add_gate(composer, *i, b);
            constants_offset += 1;
        });

        let j = buf[POSEIDON_WIDTH - 1];
        for _ in 0..5 {
            buf[POSEIDON_WIDTH - 1] = helpers::mul_gate(composer, buf[POSEIDON_WIDTH - 1], j);
        }

        let mut product = [zero; POSEIDON_WIDTH];
        for j in 0..POSEIDON_WIDTH {
            for k in 0..POSEIDON_WIDTH {
                let a = composer.add_input(MDS[j][k]);
                let o = helpers::mul_gate(composer, a, buf[k]);
                let o = helpers::add_gate(composer, product[j], o);
                product[j] = o;
            }
        }

        buf.copy_from_slice(&product);
    }

    for _ in 0..FULL_ROUNDS / 2 {
        buf.iter_mut().for_each(|i| {
            let b = composer.add_input(ROUND_CONSTANTS[constants_offset]);
            let mut o = helpers::add_gate(composer, *i, b);

            let j = o;
            for _ in 0..5 {
                o = helpers::mul_gate(composer, o, j);
            }

            *i = o;
            constants_offset += 1;
        });

        let mut product = [zero; POSEIDON_WIDTH];
        for j in 0..POSEIDON_WIDTH {
            for k in 0..POSEIDON_WIDTH {
                let a = composer.add_input(MDS[j][k]);
                let o = helpers::mul_gate(composer, a, buf[k]);
                let o = helpers::add_gate(composer, product[j], o);
                product[j] = o;
            }
        }

        buf.copy_from_slice(&product);
    }

    helpers::constrain_gate(composer, buf[1], h);
}

pub fn circuit(
    domain: &EvaluationDomain<Scalar>,
    ck: &Powers<Curve>,
    h: Scalar,
) -> (Transcript, PreProcessedCircuit, Vec<Scalar>) {
    let mut transcript = gen_transcript();
    let mut composer = StandardComposer::new();

    poseidon_gadget(&mut composer, None, h);
    composer.add_dummy_constraints();
    composer.add_dummy_constraints();
    composer.add_dummy_constraints();

    let pi = composer.public_inputs().to_vec();
    let circuit = composer.preprocess(&ck, &mut transcript, &domain);

    (transcript, circuit, pi)
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
    transcript: &mut Transcript,
    circuit: &PreProcessedCircuit,
    vk: &VerifierKey<Curve>,
    proof: &Proof,
    pi: &[Scalar],
) -> bool {
    proof.verify(&circuit, transcript, vk, pi)
}

#[cfg(test)]
mod tests {
    use crate::Scalar;

    use algebra::fields::Field;
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
        let public_parameters = srs::setup(8192);
        let (ck, vk) = srs::trim(&public_parameters, 8192).unwrap();
        let domain: EvaluationDomain<Scalar> = EvaluationDomain::new(4100).unwrap();
        let e = super::poseidon(Scalar::zero());
        let (transcript, circuit, mut pi) = super::circuit(&domain, &ck, e);
        let pi_h = pi.iter().position(|p| p == &e).unwrap();

        let x = Scalar::from(31u64);
        let h = super::poseidon(x);

        let y = Scalar::from(30u64);
        let i = super::poseidon(y);

        let proof = super::prove(&domain, &ck, x, h);
        pi[pi_h] = h;
        assert!(super::verify(
            &mut transcript.clone(),
            &circuit,
            &vk,
            &proof,
            pi.as_slice()
        ));

        let proof = super::prove(&domain, &ck, y, i);
        pi[pi_h] = i;
        assert!(super::verify(
            &mut transcript.clone(),
            &circuit,
            &vk,
            &proof,
            pi.as_slice()
        ));

        // Wrong pre-image
        let wrong_proof = super::prove(&domain, &ck, y, h);
        pi[pi_h] = h;
        assert!(!super::verify(
            &mut transcript.clone(),
            &circuit,
            &vk,
            &wrong_proof,
            pi.as_slice()
        ));

        // Wrong public image
        let wrong_proof = super::prove(&domain, &ck, x, i);
        pi[pi_h] = i;
        assert!(!super::verify(
            &mut transcript.clone(),
            &circuit,
            &vk,
            &wrong_proof,
            pi.as_slice()
        ));

        // Inconsistent public image
        let proof = super::prove(&domain, &ck, x, h);
        pi[pi_h] = i;
        assert!(!super::verify(
            &mut transcript.clone(),
            &circuit,
            &vk,
            &proof,
            pi.as_slice()
        ));
    }
}
