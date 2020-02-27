use plonk_gadgets::{gadgets::poseidon_preimage, Curve, PreProcessedCircuit, Proof, Scalar};

use algebra::fields::Field;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ff_fft::domain::EvaluationDomain;
use merlin::Transcript;
use plonk::srs;
use poly_commit::kzg10::{Powers, UniversalParams, VerifierKey};

fn poseidon_prove(
    _public_parameters: &UniversalParams<Curve>,
    domain: &EvaluationDomain<Scalar>,
    ck: &Powers<Curve>,
    x: Scalar,
    h: Scalar,
) -> Proof {
    poseidon_preimage::prove(domain, ck, x, h)
}

fn poseidon_verify(
    transcript: &Transcript,
    circuit: &PreProcessedCircuit,
    vk: &VerifierKey<Curve>,
    proof: &Proof,
    pi: &[Scalar],
) -> bool {
    let mut transcript = transcript.clone();
    let ok = poseidon_preimage::verify(&mut transcript, circuit, vk, proof, pi);
    assert!(ok);
    ok
}

fn benchmark_poseidon(c: &mut Criterion) {
    let public_parameters = srs::setup(8192);
    let (ck, vk) = srs::trim(&public_parameters, 8192).unwrap();
    let domain: EvaluationDomain<Scalar> = EvaluationDomain::new(4100).unwrap();

    let x = Scalar::from(31u64);
    let h = poseidon_preimage::poseidon(x);
    let proof = poseidon_prove(&public_parameters, &domain, &ck, x, h);

    let e = poseidon_preimage::poseidon(Scalar::zero());
    let (transcript, circuit, mut pi) = poseidon_preimage::circuit(&domain, &ck, e);
    let pi_h = pi.iter().position(|p| p == &e).unwrap();
    pi[pi_h] = h;

    c.bench_function("poseidon prove 8192 gates 4100 coef", |b| {
        b.iter(|| poseidon_prove(&public_parameters, &domain, &ck, black_box(x), black_box(h)))
    });

    c.bench_function("poseidon verify 8192 gates 4100 coef", |b| {
        b.iter(|| poseidon_verify(&transcript, &circuit, &vk, &proof, black_box(pi.as_slice())))
    });
}

criterion_group! {
    name = poseidon_group;

    config = Criterion::default()
        .sample_size(10);

    targets = benchmark_poseidon
}
criterion_main!(poseidon_group);
