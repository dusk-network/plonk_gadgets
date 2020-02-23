use crate::{Field, Scalar, StandardComposer, Variable};
use num_traits::{One, Zero};

pub fn add(composer: &mut StandardComposer, a: Variable, b: Variable) -> Variable {
    composer
        .add_gate(
            a.into(),
            b.into(),
            Scalar::one(),
            Scalar::one(),
            -Scalar::one(),
            Scalar::zero(),
            Scalar::zero(),
        )
        .2
}

pub fn mul(composer: &mut StandardComposer, a: Variable, b: Variable) -> Variable {
    composer
        .mul_gate(
            a.into(),
            b.into(),
            Scalar::one(),
            -Scalar::one(),
            Scalar::zero(),
            Scalar::zero(),
        )
        .2
}

pub fn constrain(composer: &mut StandardComposer, a: Variable, b: Scalar) {
    let zero = composer.add_input(Scalar::zero());
    composer.poly_gate(
        a.into(),
        zero.into(),
        zero.into(),
        Scalar::zero(),
        -Scalar::one(),
        Scalar::zero(),
        Scalar::zero(),
        Scalar::zero(),
        b,
    );
}
