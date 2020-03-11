use algebra::fields::jubjub::fq::Fq;
use plonk::cs::constraint_system::{LinearCombination, Variable};

/// Represents a Variable that has already
/// been constrained to be either one or zero.
#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub struct BoolVar(pub(crate) Variable);

impl From<BoolVar> for LinearCombination<Fq> {
    fn from(v: BoolVar) -> LinearCombination<Fq> {
        let lc: LinearCombination<Fq> = v.0.into();
        lc
    }
}

impl Into<Variable> for BoolVar {
    fn into(self) -> Variable {
        self.0
    }
}