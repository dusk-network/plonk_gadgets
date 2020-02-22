use plonk_gadgets::{gadgets::poseidon_preimage, Curve, Proof, Scalar};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ff_fft::domain::EvaluationDomain;
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
    _public_parameters: &UniversalParams<Curve>,
    domain: &EvaluationDomain<Scalar>,
    ck: &Powers<Curve>,
    vk: &VerifierKey<Curve>,
    proof: &Proof,
    h: Scalar,
) -> bool {
    poseidon_preimage::verify(domain, ck, vk, &proof, h)
}

fn benchmark_poseidon(c: &mut Criterion) {
    let public_parameters = srs::setup(8192);
    let (ck, vk) = srs::trim(&public_parameters, 8192).unwrap();
    let domain: EvaluationDomain<Scalar> = EvaluationDomain::new(4100).unwrap();

    let x = Scalar::from(31u64);
    let h = poseidon_preimage::poseidon(x);
    let proof = poseidon_prove(&public_parameters, &domain, &ck, x, h);

    c.bench_function("poseidon prove 8192 gates 4100 coef", |b| {
        b.iter(|| poseidon_prove(&public_parameters, &domain, &ck, black_box(x), black_box(h)))
    });

    c.bench_function("poseidon verify 8192 gates 4100 coef", |b| {
        b.iter(|| poseidon_verify(&public_parameters, &domain, &ck, &vk, &proof, black_box(h)))
    });
}

criterion_group! {
    name = poseidon_group;

    config = Criterion::default()
        .sample_size(10);

    targets = benchmark_poseidon
}
criterion_main!(poseidon_group);
