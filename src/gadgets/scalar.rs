use algebra::curves::bls12_381::Bls12_381;
use algebra::curves::jubjub::JubJubProjective;
use algebra::fields::jubjub::fq::Fq;
use plonk::cs::composer::StandardComposer;
use plonk::cs::constraint_system::{LinearCombination as LC, Variable};

pub type Bls12_381Composer = StandardComposer<Bls12_381>;
// Represents a JubJub Point using Twisted Edwards Extended Coordinates
pub struct JubJubPointGadget<Fq: algebra::fields::Field> {
    pub X: LC<Fq>,
    pub Y: LC<Fq>,
    pub Z: LC<Fq>,
    pub T: LC<Fq>,
}

impl JubJubPointGadget<Fq> {
    pub fn from_point(point: &JubJubProjective, composer: &mut Bls12_381Composer) -> Self {
        JubJubPointGadget {
            X: composer.add_input(point.x).into(),
            Y: composer.add_input(point.y).into(),
            Z: composer.add_input(point.z).into(),
            T: composer.add_input(point.t).into(),
        }
    }
}
