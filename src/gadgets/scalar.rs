use algebra::curves::bls12_381::Bls12_381;
use algebra::fields::{jubjub::fq::Fq, PrimeField};
use num_traits::{One, Zero};
use plonk::cs::composer::StandardComposer;
use plonk::cs::constraint_system::{LinearCombination as LC, Variable};
use rand::RngCore;

/// Conditionally selects the value provided or a zero instead.
/// NOTE that the `select` input has to be previously constrained to
/// be either `one` or `zero`.
/// ## Performs:
/// x' = x if select = 1
/// x' = 0 if select = 0
pub fn conditionally_select_zero(
    composer: &mut StandardComposer<Bls12_381>,
    x: LC<Fq>,
    select: LC<Fq>,
) -> Variable {
    composer
        .mul_gate(x, select, Fq::one(), Fq::one(), Fq::zero(), Fq::zero())
        .2
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
    x: LC<Fq>,
    select: LC<Fq>,
) -> Variable {
    let one = composer.add_input(Fq::one());
    let (select, _, bit_t_x) =
        composer.mul_gate(x, select, Fq::one(), Fq::one(), Fq::zero(), Fq::zero());
    // XXX: We can expres the triple addition as a LC and then constrain, but two gates is more readable ATM
    let (_, _, bit_ty_one) = composer.add_gate(
        bit_t_x.into(),
        one.into(),
        Fq::one(),
        Fq::one(),
        Fq::one(),
        Fq::zero(),
        Fq::zero(),
    );
    composer
        .add_gate(
            bit_ty_one.into(),
            select.into(),
            -Fq::one(),
            Fq::one(),
            Fq::one(),
            Fq::zero(),
            Fq::zero(),
        )
        .2
}

/// Adds constraints to the CS which check that a Variable != 0
pub fn is_non_zero(
    composer: &mut StandardComposer<Bls12_381>,
    var: LC<Fq>,
    var_assigment: Option<Fq>,
) {
    let one = composer.add_input(Fq::one());
    // XXX: We use this Fq random obtention but we will use the random variable generator
    // that we will include in the PLONK API on the future.
    let inv = var_assigment.unwrap_or_else(|| {
        Fq::from_random_bytes(&rand::thread_rng().next_u64().to_le_bytes()).unwrap()
    });
    let inv_var = composer.add_input(inv);
    // Var * Inv(Var) = 1
    composer.poly_gate(
        var,
        inv_var.into(),
        one.into(),
        Fq::one(),
        Fq::zero(),
        Fq::zero(),
        Fq::one(),
        Fq::zero(),
        Fq::zero(),
    );
}
