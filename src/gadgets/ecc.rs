use algebra::curves::bls12_381::Bls12_381;
use algebra::curves::jubjub::{JubJubParameters, JubJubProjective};
use algebra::curves::models::TEModelParameters;
use algebra::fields::jubjub::fq::Fq;
use num_traits::{One, Zero};
use plonk::cs::composer::StandardComposer;
use plonk::cs::constraint_system::LinearCombination as LC;

pub type Bls12_381Composer = StandardComposer<Bls12_381>;
// Represents a JubJub Point using Twisted Edwards Extended Coordinates
pub struct JubJubPointGadget<Fq: algebra::fields::PrimeField> {
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

    pub fn add(&self, other: &Self, composer: &mut Bls12_381Composer) -> JubJubPointGadget<Fq> {
        // Add a & d curve params to the circuit or get the reference if
        // they've been already committed
        let a = composer.add_input(JubJubParameters::COEFF_A);
        let d = composer.add_input(JubJubParameters::COEFF_D);

        // Point addition impl
        // A = p1_x * p2_x
        // B = p1_y * p2_y
        // C = d*(p1_t * p2_t)
        // D = p1_z * p2_z
        // E = (p1_x + p1_y) * (p2_x + p2_y) + a*A + a*B
        // F = D - C
        // G = D + C
        // H = B + A
        // X3 = E * F , Y3 = G * H, Z3 = F * G, T3 = E * H
        //
        // Compute A
        let (A, X, other_x) = composer.mul_gate(
            self.X.clone(),
            other.X.clone(),
            Fq::one(),
            Fq::one(),
            Fq::zero(),
            Fq::zero(),
        );
        // Compute B
        let (Y, other_y, B) = composer.mul_gate(
            self.Y.clone(),
            other.Y.clone(),
            Fq::one(),
            Fq::one(),
            Fq::zero(),
            Fq::zero(),
        );
        // Compute C
        let (_, _, pt) = composer.mul_gate(
            self.T.clone(),
            other.T.clone(),
            Fq::one(),
            Fq::one(),
            Fq::zero(),
            Fq::zero(),
        );
        let (_, _, C) = composer.mul_gate(
            pt.into(),
            d.into(),
            Fq::one(),
            Fq::one(),
            Fq::zero(),
            Fq::zero(),
        );
        // Compute D
        let (_, _, D) = composer.mul_gate(
            self.Z.clone(),
            other.Z.clone(),
            Fq::one(),
            Fq::one(),
            Fq::zero(),
            Fq::zero(),
        );
        // Compute E
        let E = {
            let (_, _, E1) = composer.add_gate(
                X.into(),
                Y.into(),
                Fq::one(),
                Fq::one(),
                Fq::one(),
                Fq::zero(),
                Fq::zero(),
            );
            let (_, _, E2) = composer.add_gate(
                other_x.into(),
                other_y.into(),
                Fq::one(),
                Fq::one(),
                Fq::one(),
                Fq::zero(),
                Fq::zero(),
            );
            let (_, _, E12) = composer.mul_gate(
                E1.into(),
                E2.into(),
                Fq::one(),
                Fq::one(),
                Fq::zero(),
                Fq::zero(),
            );
            let (_, _, aA) = composer.mul_gate(
                a.into(),
                A.into(),
                Fq::one(),
                Fq::one(),
                Fq::zero(),
                Fq::zero(),
            );
            let (_, _, aB) = composer.mul_gate(
                a.into(),
                B.into(),
                Fq::one(),
                Fq::one(),
                Fq::zero(),
                Fq::zero(),
            );
            // XXX: This can be translated to one single gate if we accept `q_c` as LC instead of E::Fr in plonk
            let (_, _, aAaB) = composer.add_gate(
                aA.into(),
                aB.into(),
                Fq::one(),
                Fq::one(),
                Fq::one(),
                Fq::zero(),
                Fq::zero(),
            );
            // Return E
            composer
                .add_gate(
                    E12.into(),
                    aAaB.into(),
                    Fq::one(),
                    Fq::one(),
                    Fq::one(),
                    Fq::zero(),
                    Fq::zero(),
                )
                .2
        };
        // Compute F
        let F = composer
            .add_gate(
                D.into(),
                C.into(),
                Fq::one(),
                -Fq::one(),
                Fq::one(),
                Fq::zero(),
                Fq::zero(),
            )
            .2;

        // Compute G
        let G = composer
            .add_gate(
                D.into(),
                C.into(),
                Fq::one(),
                Fq::one(),
                Fq::one(),
                Fq::zero(),
                Fq::zero(),
            )
            .2;

        // Compute H
        let H = composer
            .add_gate(
                A.into(),
                B.into(),
                Fq::one(),
                Fq::one(),
                Fq::one(),
                Fq::zero(),
                Fq::zero(),
            )
            .2;
        // Compute new point coords
        let (E, F, new_x) = composer.mul_gate(
            E.into(),
            F.into(),
            Fq::one(),
            Fq::one(),
            Fq::zero(),
            Fq::zero(),
        );
        let (G, H, new_y) = composer.mul_gate(
            G.into(),
            H.into(),
            Fq::one(),
            Fq::one(),
            Fq::zero(),
            Fq::zero(),
        );
        let (_, _, new_z) = composer.mul_gate(
            F.into(),
            G.into(),
            Fq::one(),
            Fq::one(),
            Fq::zero(),
            Fq::zero(),
        );
        let (_, _, new_t) = composer.mul_gate(
            E.into(),
            H.into(),
            Fq::one(),
            Fq::one(),
            Fq::zero(),
            Fq::zero(),
        );

        JubJubPointGadget {
            X: new_x.into(),
            Y: new_y.into(),
            Z: new_z.into(),
            T: new_t.into(),
        }
    }

    // Builds and adds to the CS the circuit that corresponds to the
    /// doubling of a Twisted Edwards point in Extended Coordinates.
    pub fn double(&self, composer: &mut Bls12_381Composer) -> JubJubPointGadget<Fq> {
        // Add a & d curve params to the circuit or get the reference if
        // they've been already committed
        let a = composer.add_input(JubJubParameters::COEFF_A);
        let d = composer.add_input(JubJubParameters::COEFF_D);

        // Point doubling impl
        // A = p1_x²
        // B = p1_y²
        // C = 2*p1_z²
        // D = a*A
        // E = (p1_x + p1_y)² - A - B
        // G = D + B
        // F = G - C
        // H = D - B
        // X3 = E * F,  Y3 = G * H, Z3 = F * G, T3 = E * H

        // Compute A
        let (X, _, A) = composer.mul_gate(
            self.X.clone().into(),
            self.X.clone().into(),
            Fq::one(),
            Fq::one(),
            Fq::zero(),
            Fq::zero(),
        );
        // Compute B
        let (Y, _, B) = composer.mul_gate(
            self.Y.clone().into(),
            self.Y.clone().into(),
            Fq::one(),
            Fq::one(),
            Fq::zero(),
            Fq::zero(),
        );
        // Compute C
        let (_, C, Z) = composer.mul_gate(
            self.Z.clone().into(),
            self.Z.clone().into(),
            Fq::from(2u8),
            Fq::one(),
            Fq::zero(),
            Fq::zero(),
        );
        // Compute D
        let (_, _, D) = composer.mul_gate(
            a.into(),
            A.into(),
            Fq::one(),
            Fq::one(),
            Fq::zero(),
            Fq::zero(),
        );
        // Compute E
        let E = {
            let (_, _, p1_x_y) = composer.add_gate(
                X.into(),
                Y.into(),
                Fq::one(),
                Fq::one(),
                Fq::one(),
                Fq::zero(),
                Fq::zero(),
            );
            let (_, _, p1_x_y_sq) = composer.mul_gate(
                p1_x_y.into(),
                p1_x_y.into(),
                Fq::one(),
                Fq::one(),
                Fq::zero(),
                Fq::zero(),
            );
            let (_, _, min_a_min_b) = composer.add_gate(
                A.into(),
                B.into(),
                -Fq::one(),
                -Fq::one(),
                Fq::one(),
                Fq::zero(),
                Fq::zero(),
            );
            composer
                .add_gate(
                    p1_x_y_sq.into(),
                    min_a_min_b.into(),
                    Fq::one(),
                    -Fq::one(),
                    Fq::one(),
                    Fq::zero(),
                    Fq::zero(),
                )
                .2
        };
        // Compute G
        let (_, _, G) = composer.add_gate(
            D.into(),
            B.into(),
            Fq::one(),
            Fq::one(),
            Fq::one(),
            Fq::zero(),
            Fq::zero(),
        );
        // Compute F
        let (_, _, F) = composer.add_gate(
            G.into(),
            C.into(),
            Fq::one(),
            -Fq::one(),
            Fq::one(),
            Fq::zero(),
            Fq::zero(),
        );
        // Compute H
        let (_, _, H) = composer.add_gate(
            D.into(),
            B.into(),
            Fq::one(),
            -Fq::one(),
            Fq::one(),
            Fq::zero(),
            Fq::zero(),
        );
        // Compute point coordinates
        let (_, _, new_x) = composer.mul_gate(
            E.into(),
            F.into(),
            Fq::one(),
            Fq::one(),
            Fq::zero(),
            Fq::zero(),
        );
        let (_, _, new_y) = composer.mul_gate(
            G.into(),
            H.into(),
            Fq::one(),
            Fq::one(),
            Fq::zero(),
            Fq::zero(),
        );
        let (_, _, new_z) = composer.mul_gate(
            F.into(),
            G.into(),
            Fq::one(),
            Fq::one(),
            Fq::zero(),
            Fq::zero(),
        );
        let (_, _, new_t) = composer.mul_gate(
            E.into(),
            H.into(),
            Fq::one(),
            Fq::one(),
            Fq::zero(),
            Fq::zero(),
        );

        JubJubPointGadget {
            X: new_x.into(),
            Y: new_y.into(),
            Z: new_z.into(),
            T: new_t.into(),
        }
    }
    // self.x * other.z = other.x * self.z AND self.y * other.z == other.y * self.z
    pub fn equal(&self, other: &JubJubPointGadget<Fq>, composer: &mut StandardComposer<Bls12_381>) {
        let (_, other_z, a) = composer.mul_gate(
            self.X.clone(),
            other.Z.clone(),
            Fq::one(),
            Fq::one(),
            Fq::zero(),
            Fq::zero(),
        );
        let (_, Z, b) = composer.mul_gate(
            other.X.clone(),
            self.Z.clone(),
            Fq::one(),
            Fq::one(),
            Fq::zero(),
            Fq::zero(),
        );
        // Constraint a - b == 0
        let _ = composer.add_gate(
            a.into(),
            b.into(),
            -Fq::one(),
            Fq::one(),
            Fq::one(),
            Fq::zero(),
            Fq::zero(),
        );
        let (_, _, c) = composer.mul_gate(
            self.Y.clone(),
            other_z.into(),
            Fq::one(),
            Fq::one(),
            Fq::zero(),
            Fq::zero(),
        );
        let (_, _, d) = composer.mul_gate(
            other.Y.clone(),
            Z.into(),
            Fq::one(),
            Fq::one(),
            Fq::zero(),
            Fq::zero(),
        );
        // Constraint a - b == 0
        let _ = composer.add_gate(
            a.into(),
            b.into(),
            -Fq::one(),
            Fq::one(),
            Fq::one(),
            Fq::zero(),
            Fq::zero(),
        );
    }
    /// Adds constraints to ensure that the point satisfies the JubJub curve eq
    /// by verifying `(aX^{2}+Y^{2})Z^{2} = Z^{4}+d(X^{2})Y^{2}`
    pub fn satisfy_curve_eq(&self, composer: &mut StandardComposer<Bls12_381>) {
        // Add a & d curve params to the circuit or get the reference if
        // they've been already committed
        let a = composer.add_input(JubJubParameters::COEFF_A);
        let d = composer.add_input(JubJubParameters::COEFF_D);

        // Compute X²
        let (_, _, x_sq) = composer.mul_gate(
            self.X.clone(),
            self.X.clone(),
            Fq::one(),
            Fq::one(),
            Fq::zero(),
            Fq::zero(),
        );

        // Compute a * X²
        let (_, _, a_x_sq) = composer.mul_gate(
            x_sq.into(),
            a.into(),
            Fq::one(),
            Fq::one(),
            Fq::zero(),
            Fq::zero(),
        );
        // Compute Y²
        let (_, _, y_sq) = composer.mul_gate(
            self.Y.clone(),
            self.Y.clone(),
            Fq::one(),
            Fq::one(),
            Fq::zero(),
            Fq::zero(),
        );
        // Compute a*X² + Y²
        let (_, _, a_xsq_ysq) = composer.add_gate(
            a_x_sq.into(),
            y_sq.into(),
            Fq::one(),
            Fq::one(),
            Fq::one(),
            Fq::zero(),
            Fq::zero(),
        );
        // Compute Z²
        let (_, _, z_sq) = composer.mul_gate(
            self.Z.clone(),
            self.Z.clone(),
            Fq::one(),
            Fq::one(),
            Fq::zero(),
            Fq::zero(),
        );
        // Compute left assigment
        let (_, _, left_assig) = composer.mul_gate(
            a_xsq_ysq.into(),
            z_sq.into(),
            Fq::one(),
            Fq::one(),
            Fq::zero(),
            Fq::zero(),
        );

        // Compute Z⁴
        let (_, _, z_sq_sq) = composer.mul_gate(
            z_sq.into(),
            z_sq.into(),
            Fq::one(),
            Fq::one(),
            Fq::zero(),
            Fq::zero(),
        );
        // Compute d(X²)
        let (_, _, d_x_sq) = composer.mul_gate(
            d.into(),
            x_sq.into(),
            Fq::one(),
            Fq::one(),
            Fq::zero(),
            Fq::zero(),
        );
        // Compute d*(X²) * Y²
        let (_, _, d_x_sq_y_sq) = composer.mul_gate(
            d_x_sq.into(),
            y_sq.into(),
            Fq::one(),
            Fq::one(),
            Fq::zero(),
            Fq::zero(),
        );
        // Compute right assigment
        let (_, _, right_assig) = composer.add_gate(
            z_sq_sq.into(),
            d_x_sq_y_sq.into(),
            Fq::one(),
            Fq::one(),
            Fq::one(),
            Fq::zero(),
            Fq::zero(),
        );
        // Create Variable 0
        let var_zero = composer.add_input(Fq::zero());
        // Constrain right_assig = left_assig
        composer.poly_gate(
            left_assig.into(),
            right_assig.into(),
            var_zero.into(),
            Fq::zero(),
            -Fq::one(),
            Fq::one(),
            Fq::one(),
            Fq::zero(),
            Fq::zero(),
        );
    }
}

mod tests {
    use super::*;
    use ff_fft::EvaluationDomain;
    use merlin::Transcript;
    use plonk::cs::proof::Proof;
    use plonk::cs::Composer;
    use plonk::srs;
    use poly_commit::kzg10::UniversalParams;
    use poly_commit::kzg10::{Powers, VerifierKey};
    fn gen_transcript() -> Transcript {
        Transcript::new(b"jubjub_ecc_ops")
    }

    fn prove(
        composer: &mut StandardComposer<Bls12_381>,
    ) -> (
        Proof<Bls12_381>,
        UniversalParams<Bls12_381>,
        Powers<Bls12_381>,
        Vec<Fq>,
    ) {
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();

        let mut transcript = gen_transcript();
        let public_params = srs::setup(2 * composer.circuit_size().next_power_of_two());
        let (ck, vk) = srs::trim(
            &public_params,
            2 * composer.circuit_size().next_power_of_two(),
        )
        .unwrap();
        let eval_domain = EvaluationDomain::<Fq>::new(composer.circuit_size()).unwrap();

        let prep_circ = composer.preprocess(&ck, &mut transcript, &eval_domain);
        let proof = composer.prove(&ck, &prep_circ, &mut transcript);
        unimplemented!()
        /*(
            proof,
            public_params.clone(),
            ck,
            composer.public_inputs().to_owned(),
        )*/
    }

    fn verify(
        composer: &mut StandardComposer<Bls12_381>,
        public_params: UniversalParams<Bls12_381>,
        proof: Proof<Bls12_381>,
        pub_inputs: &Vec<Fq>,
    ) -> bool {
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();

        let mut transcript = gen_transcript();
        let public_params = srs::setup(2 * composer.circuit_size().next_power_of_two());
        let (ck, vk) = srs::trim(
            &public_params,
            2 * composer.circuit_size().next_power_of_two(),
        )
        .unwrap();
        let eval_domain = EvaluationDomain::<Fq>::new(composer.circuit_size()).unwrap();
        let prep_circ = composer.preprocess(&ck, &mut transcript, &eval_domain);
        proof.verify(&prep_circ, &mut transcript, &vk, pub_inputs)
    }

    #[test]
    #[ignore]
    fn dummy_test() {
        let mut composer = StandardComposer::<Bls12_381>::new();
        let one = composer.add_input(Fq::one());
        composer.add_gate(
            one.into(),
            one.into(),
            Fq::one(),
            Fq::one(),
            Fq::one(),
            Fq::zero(),
            Fq::zero(),
        );
        composer.add_dummy_constraints();
        let (proof, public_params, ck, public_inputs) = prove(&mut composer);
        let mut verif_composer = StandardComposer::<Bls12_381>::new();
        let one = verif_composer.add_input(Fq::one());
        verif_composer.add_gate(
            one.into(),
            one.into(),
            Fq::one(),
            Fq::one(),
            Fq::one(),
            Fq::zero(),
            Fq::zero(),
        );
        verif_composer.add_dummy_constraints();
        assert!(verify(
            &mut verif_composer,
            public_params,
            proof,
            &public_inputs
        ));
    }
}
