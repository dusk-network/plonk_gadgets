# plonk_gadgets
A library that cointains the gadgets that Dusk-phoenix protocol needs to build it's ZK-Circuits.

## WARNING
This is an implementation used for testing purposes at the moment. Use under your own responsability.

This library provides:

- Scalar gadgets: Binary_constraint, Non-zero, conditionally_select_one, conditionally_select_zero.
- ECC gadgets: Add, Double, conditionally_select_identity, curve_eq_satisfaction & scalar_mul.
- Secret Key knowledge gadget appart built-in with the previous mentioned gadgets.
