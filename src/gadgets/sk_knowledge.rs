use crate::gadgets::boolean::BoolVar;
use crate::gadgets::ecc::*;
use crate::gadgets::scalar;
use algebra::curves::bls12_381::Bls12_381;
use algebra::curves::jubjub::{JubJubParameters, JubJubProjective};
use algebra::curves::models::TEModelParameters;
use algebra::fields::jubjub::{fq::Fq, fr::Fr};
use algebra::fields::PrimeField;
use algebra::ToBytes;
use plonk::cs::composer::StandardComposer;
use plonk::cs::constraint_system::{LinearCombination as LC, Variable};
use rand::{thread_rng, RngCore};

pub fn sk_knowledge(
    composer: &mut StandardComposer<Bls12_381>,
    basepoint: &JubJubPointGadget<Fq>,
    pub_key: &JubJubPointGadget<Fq>,
    scalar: Option<Fr>,
) {
    // Convert the Scalar into bytes
    let sk = scalar
        .unwrap_or_else(|| Fr::from_random_bytes(&thread_rng().next_u64().to_le_bytes()).unwrap());
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
