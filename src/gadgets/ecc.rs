use crate::gadgets::boolean::BoolVar;
use crate::gadgets::scalar::*;
use algebra::curves::bls12_381::Bls12_381;
use algebra::curves::jubjub::{JubJubParameters, JubJubProjective};
use algebra::curves::models::TEModelParameters;
use algebra::fields::bls12_381::Fr;
use num_traits::{One, Zero};
use plonk::cs::composer::StandardComposer;
use plonk::cs::constraint_system::Variable;

pub type Bls12_381Composer = StandardComposer<Bls12_381>;
/// Represents a JubJub Point using Twisted Edwards Extended Coordinates.
/// Each one of the coordinates is represented by a `LinearCombination<PrimeField>`
pub struct JubJubPointGadget {
    pub X: Variable,
    pub Y: Variable,
    pub Z: Variable,
    pub T: Variable,
}

impl JubJubPointGadget {
    pub fn from_point(composer: &mut Bls12_381Composer, point: &JubJubProjective) -> Self {
        JubJubPointGadget {
            X: composer.add_input(point.x),
            Y: composer.add_input(point.y),
            Z: composer.add_input(point.z),
            T: composer.add_input(point.t),
        }
    }

    pub fn add(&self, composer: &mut Bls12_381Composer, other: &Self) -> JubJubPointGadget {
        // Add a & d curve params to the circuit or get the reference if
        // they've been already committed
        let coeff_a = JubJubParameters::COEFF_A;
        let coeff_d = JubJubParameters::COEFF_D;

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
        let A = composer.mul(
            self.X,
            other.X,
            Fr::one(),
            -Fr::one(),
            Fr::zero(),
            Fr::zero(),
        );
        // Compute B
        let B = composer.mul(
            self.Y,
            other.Y,
            Fr::one(),
            -Fr::one(),
            Fr::zero(),
            Fr::zero(),
        );
        // Compute C
        let C = composer.mul(self.T, other.T, coeff_d, -Fr::one(), Fr::zero(), Fr::zero());
        // Compute D
        let D = composer.mul(
            self.Z,
            other.Z,
            Fr::one(),
            -Fr::one(),
            Fr::zero(),
            Fr::zero(),
        );
        // Compute E
        let E = {
            let E1 = composer.add(
                self.X,
                self.Y,
                Fr::one(),
                Fr::one(),
                -Fr::one(),
                Fr::zero(),
                Fr::zero(),
            );
            let E2 = composer.add(
                other.X,
                other.Y,
                Fr::one(),
                Fr::one(),
                -Fr::one(),
                Fr::zero(),
                Fr::zero(),
            );
            let E12 = composer.mul(E1, E2, Fr::one(), -Fr::one(), Fr::zero(), Fr::zero());
            // aA + aB
            let aAaB = composer.add(A, B, coeff_a, coeff_a, -Fr::one(), Fr::zero(), Fr::zero());
            // Return E
            composer.add(
                E12,
                aAaB,
                Fr::one(),
                Fr::one(),
                -Fr::one(),
                Fr::zero(),
                Fr::zero(),
            )
        };
        // Compute F
        let F = composer.add(
            D.into(),
            C.into(),
            Fr::one(),
            -Fr::one(),
            -Fr::one(),
            Fr::zero(),
            Fr::zero(),
        );

        // Compute G
        let G = composer.add(
            D.into(),
            C.into(),
            Fr::one(),
            Fr::one(),
            -Fr::one(),
            Fr::zero(),
            Fr::zero(),
        );

        // Compute H
        let H = composer.add(
            A.into(),
            B.into(),
            Fr::one(),
            Fr::one(),
            -Fr::one(),
            Fr::zero(),
            Fr::zero(),
        );
        // Compute new point coords
        let new_x = composer.mul(E, F, Fr::one(), -Fr::one(), Fr::zero(), Fr::zero());
        let new_y = composer.mul(G, H, Fr::one(), -Fr::one(), Fr::zero(), Fr::zero());
        let new_z = composer.mul(F, G, Fr::one(), -Fr::one(), Fr::zero(), Fr::zero());
        let new_t = composer.mul(E, H, Fr::one(), -Fr::one(), Fr::zero(), Fr::zero());

        JubJubPointGadget {
            X: new_x,
            Y: new_y,
            Z: new_z,
            T: new_t,
        }
    }

    // Builds and adds to the CS the circuit that corresponds to the
    /// doubling of a Twisted Edwards point in Extended Coordinates.
    pub fn double(&self, composer: &mut Bls12_381Composer) -> JubJubPointGadget {
        // Add a & d curve params to the circuit or get the reference if
        // they've been already committed
        let coeff_a = JubJubParameters::COEFF_A;

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
        let A = composer.mul(
            self.X,
            self.X,
            Fr::one(),
            -Fr::one(),
            Fr::zero(),
            Fr::zero(),
        );
        // Compute B
        let B = composer.mul(
            self.Y,
            self.Y,
            Fr::one(),
            -Fr::one(),
            Fr::zero(),
            Fr::zero(),
        );
        // Compute C
        let C = composer.mul(
            self.Z,
            self.Z,
            Fr::from(2u8),
            -Fr::one(),
            Fr::zero(),
            Fr::zero(),
        );
        // D comp is skipped and scaled when used.
        // Compute E
        let E = {
            let p1_x_y = composer.add(
                self.X,
                self.Y,
                Fr::one(),
                Fr::one(),
                -Fr::one(),
                Fr::zero(),
                Fr::zero(),
            );
            let p1_x_y_sq = composer.mul(
                p1_x_y,
                p1_x_y,
                Fr::one(),
                -Fr::one(),
                Fr::zero(),
                Fr::zero(),
            );
            let min_a_min_b = composer.add(
                A,
                B,
                -Fr::one(),
                -Fr::one(),
                -Fr::one(),
                Fr::zero(),
                Fr::zero(),
            );
            composer.add(
                p1_x_y_sq,
                min_a_min_b,
                Fr::one(),
                Fr::one(),
                -Fr::one(),
                Fr::zero(),
                Fr::zero(),
            )
        };
        // Compute G
        let G = composer.add(A, B, coeff_a, Fr::one(), -Fr::one(), Fr::zero(), Fr::zero());
        // Compute F
        let F = composer.add(
            G,
            C,
            Fr::one(),
            -Fr::one(),
            -Fr::one(),
            Fr::zero(),
            Fr::zero(),
        );
        // Compute H
        let H = composer.add(
            A,
            B,
            coeff_a,
            -Fr::one(),
            -Fr::one(),
            Fr::zero(),
            Fr::zero(),
        );
        // Compute point coordinates
        let new_x = composer.mul(E, F, Fr::one(), -Fr::one(), Fr::zero(), Fr::zero());
        let new_y = composer.mul(G, H, Fr::one(), -Fr::one(), Fr::zero(), Fr::zero());
        let new_z = composer.mul(F, G, Fr::one(), -Fr::one(), Fr::zero(), Fr::zero());
        let new_t = composer.mul(E, H, Fr::one(), -Fr::one(), Fr::zero(), Fr::zero());

        JubJubPointGadget {
            X: new_x.into(),
            Y: new_y.into(),
            Z: new_z.into(),
            T: new_t.into(),
        }
    }
    /// Checks the equalty between two JubJub points in TwEdws Extended Coords according
    /// to the eq: `self.x * other.z = other.x * self.z AND self.y * other.z == other.y * self.z`
    pub fn equal(&self, composer: &mut Bls12_381Composer, other: &JubJubPointGadget) {
        // First assigment
        let a = composer.mul(
            self.X,
            other.Z,
            Fr::one(),
            -Fr::one(),
            Fr::zero(),
            Fr::zero(),
        );
        let b = composer.mul(
            other.X,
            self.Z,
            Fr::one(),
            -Fr::one(),
            Fr::zero(),
            Fr::zero(),
        );
        // Constraint a - b == 0
        let a_min_b = composer.add(
            a.into(),
            b.into(),
            -Fr::one(),
            Fr::one(),
            -Fr::one(),
            Fr::zero(),
            Fr::zero(),
        );
        composer.constrain_to_constant(a_min_b, Fr::zero(), Fr::zero());
        // Second assigment
        let c = composer.mul(
            self.Y,
            other.Z,
            Fr::one(),
            -Fr::one(),
            Fr::zero(),
            Fr::zero(),
        );
        let d = composer.mul(
            other.Y,
            self.Z,
            Fr::one(),
            -Fr::one(),
            Fr::zero(),
            Fr::zero(),
        );
        // Constraint c - d == 0
        let c_min_d = composer.add(
            c,
            d,
            Fr::one(),
            -Fr::one(),
            -Fr::one(),
            Fr::zero(),
            Fr::zero(),
        );
        composer.constrain_to_constant(c_min_d, Fr::zero(), Fr::zero());
    }
    /// Adds constraints to ensure that the point satisfies the JubJub curve eq
    /// by verifying `(aX^{2}+Y^{2})Z^{2} = Z^{4}+d(X^{2})Y^{2}`
    pub fn satisfy_curve_eq(&self, composer: &mut Bls12_381Composer) {
        // Add a & d curve params to the circuit or get the reference if
        // they've been already committed
        let coeff_a = JubJubParameters::COEFF_A;
        let coeff_d = JubJubParameters::COEFF_D;

        // Compute a * X²
        let a_x_sq = composer.mul(self.X, self.X, coeff_a, -Fr::one(), Fr::zero(), Fr::zero());
        // Compute Y²
        let y_sq = composer.mul(
            self.Y,
            self.Y,
            Fr::one(),
            -Fr::one(),
            Fr::zero(),
            Fr::zero(),
        );
        // Compute a*X² + Y²
        let a_xsq_ysq = composer.add(
            a_x_sq,
            y_sq,
            Fr::one(),
            Fr::one(),
            -Fr::one(),
            Fr::zero(),
            Fr::zero(),
        );
        // Compute Z²
        let z_sq = composer.mul(
            self.Z,
            self.Z,
            Fr::one(),
            -Fr::one(),
            Fr::zero(),
            Fr::zero(),
        );
        // Compute left assigment
        let left_assig = composer.mul(
            a_xsq_ysq,
            z_sq,
            Fr::one(),
            -Fr::one(),
            Fr::zero(),
            Fr::zero(),
        );

        // Compute Z⁴
        let z_sq_sq = composer.mul(z_sq, z_sq, Fr::one(), -Fr::one(), Fr::zero(), Fr::zero());
        // Compute d * X²
        let d_x_sq = composer.mul(self.X, self.X, coeff_d, -Fr::one(), Fr::zero(), Fr::zero());
        // Compute d*(X²) * Y²
        let d_x_sq_y_sq = composer.mul(d_x_sq, y_sq, Fr::one(), -Fr::one(), Fr::zero(), Fr::zero());
        // Compute right assigment
        let right_assig = composer.add(
            z_sq_sq.into(),
            d_x_sq_y_sq.into(),
            Fr::one(),
            Fr::one(),
            -Fr::one(),
            Fr::zero(),
            Fr::zero(),
        );
        // Constrain right_assig = left_assig
        let should_be_zero = composer.add(
            left_assig.into(),
            right_assig.into(),
            -Fr::one(),
            Fr::one(),
            -Fr::one(),
            Fr::zero(),
            Fr::zero(),
        );
        composer.constrain_to_constant(should_be_zero, Fr::zero(), Fr::zero());
    }

    /// Gets an Scalar represented as a BoolVar array in Big Endian
    /// and performs an ECC scalar multiplication.
    pub fn scalar_mul(
        &self,
        composer: &mut StandardComposer<Bls12_381>,
        scalar: &[BoolVar],
    ) -> JubJubPointGadget {
        let zero = composer.add_input(Fr::zero());
        let one = composer.add_input(Fr::one());

        let mut Q = JubJubPointGadget {
            X: zero.into(),
            Y: one.into(),
            Z: one.into(),
            T: zero.into(),
        };

        for bit in scalar {
            Q = Q.double(composer);
            // If bit == 1 -> Q = Q + point
            let point_or_id = self.conditionally_select_identity(composer, *bit);
            Q = Q.add(composer, &point_or_id);
        }
        Q
    }

    /// Conditionally selects the point or the identity point according to
    /// the selector bit.
    /// P' = P <=> bit = 1
    /// P' = Identity <=> bit = 0
    pub fn conditionally_select_identity(
        &self,
        composer: &mut Bls12_381Composer,
        selector: BoolVar,
    ) -> Self {
        // x' = x if bit = 1
        // x' = 0 if bit = 0 =>
        // x' = x * bit
        let x_prime = conditionally_select_zero(composer, self.X, selector.into());

        // y' = y if bit = 1
        // y' = 1 if bit = 0 =>
        // y' = bit * y + (1 - bit)
        let y_prime = conditionally_select_one(composer, self.Y, selector.into());

        // z' = z if bit = 1
        // z' = 1 if bit = 0 =>
        // z' = bit * z + (1 - bit)
        let z_prime = conditionally_select_one(composer, self.Z, selector.into());

        // t' = t if bit = 1
        // t' = 0 if bit = 0 =>
        // t' = t * bit
        let t_prime = conditionally_select_zero(composer, self.T, selector.into());

        JubJubPointGadget {
            X: x_prime,
            Y: y_prime,
            Z: z_prime,
            T: t_prime,
        }
    }
}

mod tests {
    use super::*;
    use crate::gadgets::scalar::*;
    use algebra::curves::jubjub::{JubJubAffine, JubJubParameters, JubJubProjective};
    use algebra::fields::jubjub::{fq::Fq, fr::Fr};
    use ff_fft::EvaluationDomain;
    use merlin::Transcript;
    use plonk::cs::{proof::Proof, Composer};
    use plonk::srs::*;
    use poly_commit::kzg10::{Powers, UniversalParams, VerifierKey};
    use std::ops::{Add, Mul};

    fn gen_transcript() -> Transcript {
        Transcript::new(b"jubjub_ecc_ops")
    }

    // Provides points for a later usage in testing
    fn testing_points() -> (
        JubJubProjective,
        JubJubProjective,
        JubJubProjective,
        JubJubProjective,
        JubJubProjective,
    ) {
        let (x, y) = JubJubParameters::AFFINE_GENERATOR_COEFFS;
        let identity = JubJubProjective::zero();
        let gen = JubJubAffine::new(x, y);
        let two_gen = gen.mul(Fr::from(2u8));
        let gen_p_two_gen = two_gen.add(gen);
        let x_times_gen = gen.mul(Fr::from(127u8));

        (
            identity,
            JubJubProjective::from(gen),
            JubJubProjective::from(two_gen),
            JubJubProjective::from(gen_p_two_gen),
            JubJubProjective::from(x_times_gen),
        )
    }

    fn prove_point_equalty(
        domain: &EvaluationDomain<Fq>,
        ck: &Powers<Bls12_381>,
        P1: &JubJubProjective,
        P2: &JubJubProjective,
    ) -> Proof<Bls12_381> {
        let mut transcript = gen_transcript();
        let mut composer = StandardComposer::new();
        // Import point
        let P1_g = JubJubPointGadget::from_point(&mut composer, P1);
        // Import point
        let P2_g = JubJubPointGadget::from_point(&mut composer, P2);
        // Constrain equalty between P1 instances
        P1_g.equal(&mut composer, &P2_g);
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        let preprocessed_circuit = composer.preprocess(&ck, &mut transcript, domain);
        composer.prove(&ck, &preprocessed_circuit, &mut transcript)
    }

    fn verify_point_equalty(
        domain: &EvaluationDomain<Fq>,
        ck: &Powers<Bls12_381>,
        vk: &VerifierKey<Bls12_381>,
        proof: &Proof<Bls12_381>,
        P1: &JubJubProjective,
        P2: &JubJubProjective,
    ) -> bool {
        let mut transcript = gen_transcript();
        let mut composer = StandardComposer::new();
        // Import point
        let P1_g = JubJubPointGadget::from_point(&mut composer, P1);
        // Import point
        let P2_g = JubJubPointGadget::from_point(&mut composer, P2);
        // Constrain equalty between P1 instances
        P1_g.equal(&mut composer, &P2_g);
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        let preprocessed_circuit = composer.preprocess(&ck, &mut transcript, &domain);
        proof.verify(
            &preprocessed_circuit,
            &mut transcript,
            vk,
            &vec![Fq::zero()],
        )
    }

    fn point_equalty_roundtrip_helper(P1: &JubJubProjective, P2: &JubJubProjective) -> bool {
        let public_parameters = setup(16384, &mut rand::thread_rng());
        let (ck, vk) = trim(&public_parameters, 16384).unwrap();
        let domain: EvaluationDomain<Fq> = EvaluationDomain::new(16384).unwrap();

        let proof = prove_point_equalty(&domain, &ck, P1, P2);
        verify_point_equalty(&domain, &ck, &vk, &proof, P1, P2)
    }

    #[test]
    fn test_point_equalty() {
        let (id, P1, P2, _, _) = testing_points();
        let two = Fq::one() + Fq::one();
        let zero = Fq::zero();
        let id_2 = JubJubProjective::new(zero, two, zero, two);
        assert!(!point_equalty_roundtrip_helper(&P1, &P2));
        assert!(point_equalty_roundtrip_helper(&P1, &P1));
        assert!(point_equalty_roundtrip_helper(&id, &id_2));
    }

    fn prove_conditionally_select_identity(
        domain: &EvaluationDomain<Fq>,
        ck: &Powers<Bls12_381>,
        P1: &JubJubProjective,
        P2: &JubJubProjective,
        selector: &Fq,
    ) -> Proof<Bls12_381> {
        let mut transcript = gen_transcript();
        let mut composer = StandardComposer::new();
        let selector = composer.add_input(*selector);
        let selector = binary_constrain(&mut composer, selector);
        let P1_g = JubJubPointGadget::from_point(&mut composer, P1);
        let P2_g = JubJubPointGadget::from_point(&mut composer, P2);
        // Conditionally select the identity point
        P1_g.conditionally_select_identity(&mut composer, selector);
        // Constraint the point to be equal to the identity
        P1_g.equal(&mut composer, &P2_g);
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        let preprocessed_circuit = composer.preprocess(&ck, &mut transcript, domain);
        composer.prove(&ck, &preprocessed_circuit, &mut transcript)
    }

    fn verify_conditionally_select_identity(
        domain: &EvaluationDomain<Fq>,
        ck: &Powers<Bls12_381>,
        vk: &VerifierKey<Bls12_381>,
        proof: &Proof<Bls12_381>,
        P1: &JubJubProjective,
        P2: &JubJubProjective,
        selector: &Fq,
    ) -> bool {
        let mut transcript = gen_transcript();
        let mut composer = StandardComposer::new();
        let selector = composer.add_input(*selector);
        let selector = binary_constrain(&mut composer, selector);
        let P1_g = JubJubPointGadget::from_point(&mut composer, P1);
        let P2_g = JubJubPointGadget::from_point(&mut composer, P2);
        // Conditionally select the identity point
        P1_g.conditionally_select_identity(&mut composer, selector);
        // Constraint the point to be equal to the identity
        P1_g.equal(&mut composer, &P2_g);
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        let preprocessed_circuit = composer.preprocess(&ck, &mut transcript, &domain);
        proof.verify(
            &preprocessed_circuit,
            &mut transcript,
            vk,
            &vec![Fq::zero()],
        )
    }

    fn conditionally_select_identity_roundtrip_helper(
        P1: &JubJubProjective,
        P2: &JubJubProjective,
        selector: &Fq,
    ) -> bool {
        let public_parameters = setup(512, &mut rand::thread_rng());
        let (ck, vk) = trim(&public_parameters, 512).unwrap();
        let domain: EvaluationDomain<Fq> = EvaluationDomain::new(512).unwrap();

        let proof = prove_conditionally_select_identity(&domain, &ck, P1, P2, selector);
        verify_conditionally_select_identity(&domain, &ck, &vk, &proof, P1, P2, selector)
    }

    #[ignore]
    #[test]
    fn test_conditionally_select_identity() {
        let (id_p, P1, _, _, _) = testing_points();
        let one = Fq::one();
        let zero = Fq::zero();
        let id_p = JubJubProjective::new(zero, one, zero, one);
        assert!(conditionally_select_identity_roundtrip_helper(
            &P1, &id_p, &zero
        ));
        assert!(conditionally_select_identity_roundtrip_helper(
            &P1, &P1, &one
        ));
    }

    fn prove_point_addition(
        domain: &EvaluationDomain<Fq>,
        ck: &Powers<Bls12_381>,
        P1: &JubJubProjective,
        P2: &JubJubProjective,
        P_res: &JubJubProjective,
    ) -> Proof<Bls12_381> {
        let mut transcript = gen_transcript();
        let mut composer = StandardComposer::new();
        // Import both points
        let P1_g = JubJubPointGadget::from_point(&mut composer, P1);
        let P2_g = JubJubPointGadget::from_point(&mut composer, P2);
        let P3_g = JubJubPointGadget::from_point(&mut composer, P_res);
        // Perform the addition
        let expected_res = P1_g.add(&mut composer, &P2_g);
        // Constrain equalty with real result
        expected_res.equal(&mut composer, &P3_g);
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        let preprocessed_circuit = composer.preprocess(&ck, &mut transcript, domain);
        composer.prove(&ck, &preprocessed_circuit, &mut transcript)
    }

    fn verify_point_addition(
        domain: &EvaluationDomain<Fq>,
        ck: &Powers<Bls12_381>,
        vk: &VerifierKey<Bls12_381>,
        proof: &Proof<Bls12_381>,
        P1: &JubJubProjective,
        P2: &JubJubProjective,
        P_res: &JubJubProjective,
    ) -> bool {
        let mut transcript = gen_transcript();
        let mut composer = StandardComposer::new();
        // Import both points
        let P1_g = JubJubPointGadget::from_point(&mut composer, P1);
        let P2_g = JubJubPointGadget::from_point(&mut composer, P2);
        let P3_g = JubJubPointGadget::from_point(&mut composer, P_res);
        // Perform the addition
        let expected_res = P1_g.add(&mut composer, &P2_g);
        // Constrain equalty with real result
        expected_res.equal(&mut composer, &P3_g);
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        let preprocessed_circuit = composer.preprocess(&ck, &mut transcript, &domain);
        proof.verify(
            &preprocessed_circuit,
            &mut transcript,
            vk,
            &vec![Fq::zero()],
        )
    }

    fn point_addition_roundtrip_helper(
        P1: &JubJubProjective,
        P2: &JubJubProjective,
        P_res: &JubJubProjective,
    ) -> bool {
        let public_parameters = setup(16384, &mut rand::thread_rng());
        let (ck, vk) = trim(&public_parameters, 16384).unwrap();
        let domain: EvaluationDomain<Fq> = EvaluationDomain::new(16384).unwrap();

        let proof = prove_point_addition(&domain, &ck, P1, P2, P_res);
        verify_point_addition(&domain, &ck, &vk, &proof, P1, P2, P_res)
    }

    #[test]
    fn test_point_addition() {
        let (id_p, P1, P2, P3, _) = testing_points();
        assert!(point_addition_roundtrip_helper(&P1, &P1, &P2));
        assert!(point_addition_roundtrip_helper(&P1, &P2, &P3));
        assert!(!point_addition_roundtrip_helper(&P1, &P1, &P3));
        assert!(point_addition_roundtrip_helper(&P1, &id_p, &P1));
    }

    fn prove_point_doubling(
        domain: &EvaluationDomain<Fq>,
        ck: &Powers<Bls12_381>,
        P1: &JubJubProjective,
        P_res: &JubJubProjective,
    ) -> Proof<Bls12_381> {
        let mut transcript = gen_transcript();
        let mut composer = StandardComposer::new();
        // Import both points
        let P1_g = JubJubPointGadget::from_point(&mut composer, P1);
        let P3_g = JubJubPointGadget::from_point(&mut composer, P_res);
        // Perform the doubling
        let expected_res = P1_g.double(&mut composer);
        // Constrain equalty with real result
        expected_res.equal(&mut composer, &P3_g);
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        let preprocessed_circuit = composer.preprocess(&ck, &mut transcript, domain);
        composer.prove(&ck, &preprocessed_circuit, &mut transcript)
    }

    fn verify_point_doubling(
        domain: &EvaluationDomain<Fq>,
        ck: &Powers<Bls12_381>,
        vk: &VerifierKey<Bls12_381>,
        proof: &Proof<Bls12_381>,
        P1: &JubJubProjective,
        P_res: &JubJubProjective,
    ) -> bool {
        let mut transcript = gen_transcript();
        let mut composer = StandardComposer::new();
        // Import both points
        let P1_g = JubJubPointGadget::from_point(&mut composer, P1);
        let P3_g = JubJubPointGadget::from_point(&mut composer, P_res);
        // Perform the doubling
        let expected_res = P1_g.double(&mut composer);
        // Constrain equalty with real result
        expected_res.equal(&mut composer, &P3_g);
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        let preprocessed_circuit = composer.preprocess(&ck, &mut transcript, &domain);
        proof.verify(
            &preprocessed_circuit,
            &mut transcript,
            vk,
            &vec![Fq::zero()],
        )
    }

    fn point_doubling_roundtrip_helper(P1: &JubJubProjective, P_res: &JubJubProjective) -> bool {
        let public_parameters = setup(16384, &mut rand::thread_rng());
        let (ck, vk) = trim(&public_parameters, 16384).unwrap();
        let domain: EvaluationDomain<Fq> = EvaluationDomain::new(16384).unwrap();

        let proof = prove_point_doubling(&domain, &ck, P1, P_res);
        verify_point_doubling(&domain, &ck, &vk, &proof, P1, P_res)
    }

    #[test]
    fn test_point_doubling() {
        let (id_p, P1, P2, P_err, _) = testing_points();
        assert!(point_doubling_roundtrip_helper(&P1, &P2));
        assert!(point_doubling_roundtrip_helper(&id_p, &id_p));
        assert!(!point_doubling_roundtrip_helper(&P1, &P_err));
    }

    fn prove_curve_eq_satisfy(
        domain: &EvaluationDomain<Fq>,
        ck: &Powers<Bls12_381>,
        P1: &JubJubProjective,
    ) -> Proof<Bls12_381> {
        let mut transcript = gen_transcript();
        let mut composer = StandardComposer::new();
        // Gen Point gadgets
        let P1_g = JubJubPointGadget::from_point(&mut composer, P1);
        // Constrain the point to satisfy curve eq
        P1_g.satisfy_curve_eq(&mut composer);
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        let preprocessed_circuit = composer.preprocess(&ck, &mut transcript, domain);
        composer.prove(&ck, &preprocessed_circuit, &mut transcript)
    }

    fn verify_curve_eq_satisfy(
        domain: &EvaluationDomain<Fq>,
        ck: &Powers<Bls12_381>,
        vk: &VerifierKey<Bls12_381>,
        proof: &Proof<Bls12_381>,
        P1: &JubJubProjective,
    ) -> bool {
        let mut transcript = gen_transcript();
        let mut composer = StandardComposer::new();
        // Gen Point gadgets
        let P1_g = JubJubPointGadget::from_point(&mut composer, P1);
        // Constrain the point to satisfy curve eq
        P1_g.satisfy_curve_eq(&mut composer);
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        composer.add_dummy_constraints();
        let preprocessed_circuit = composer.preprocess(&ck, &mut transcript, &domain);
        proof.verify(
            &preprocessed_circuit,
            &mut transcript,
            vk,
            &vec![Fq::zero()],
        )
    }

    fn curve_eq_satisfy_roundtrip_helper(P1: &JubJubProjective) -> bool {
        let public_parameters = setup(16384, &mut rand::thread_rng());
        let (ck, vk) = trim(&public_parameters, 16384).unwrap();
        let domain: EvaluationDomain<Fq> = EvaluationDomain::new(16384).unwrap();

        let proof = prove_curve_eq_satisfy(&domain, &ck, P1);
        verify_curve_eq_satisfy(&domain, &ck, &vk, &proof, P1)
    }

    #[test]
    fn test_curve_eq_satisfy() {
        let (id_p, P1, _, _, _) = testing_points();
        let incorrect_point = JubJubProjective::new(Fq::one(), Fq::one(), Fq::one(), Fq::one());
        assert!(curve_eq_satisfy_roundtrip_helper(&P1));
        assert!(curve_eq_satisfy_roundtrip_helper(&id_p));
        assert!(!curve_eq_satisfy_roundtrip_helper(&incorrect_point));
    }
}
