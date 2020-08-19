//! # Plonk Gadgets
//! This library cointains the gadgets that the Dusk-Network protocol needs to build it's ZK-Circuits.
//! The library **contains generic gadgets** which are used across Dusk's tech stack, all of the other
//! gadgets used which depend on foreign types are placed on the libraries where this types are defined.
//!
//!
//! ## WARNING
//! This implementation is not audited. Use under your own responsability.
//!
//! ## Content
//! This library provides:
//!
//! - Scalar gadgets: `is_non-zero`, `maybe_equals`, `conditionally_select_one`, `conditionally_select_zero`.
//! - Range gadgets: `range_check`, `max_bound`.

#![doc(
    html_logo_url = "https://lh3.googleusercontent.com/SmwswGxtgIANTbDrCOn5EKcRBnVdHjmYsHYxLq2HZNXWCQ9-fZyaea-bNgdX9eR0XGSqiMFi=w128-h128-e365"
)]
#![doc(html_favicon_url = "https://dusk.network/lib/img/favicon-16x16.png")]
#![allow(clippy::suspicious_arithmetic_impl)]
// Some structs do not have AddAssign or MulAssign impl.
#![allow(clippy::suspicious_op_assign_impl)]
// Variables have always the same names in respect to wires.
#![allow(clippy::many_single_char_names)]
// Bool expr are usually easier to read with match statements.
#![allow(clippy::match_bool)]
#![deny(intra_doc_link_resolution_failure)]
#![deny(missing_debug_implementations)]
#![deny(missing_docs)]
#![deny(unsafe_code)]
/// Container of all of the gadgets of the library.
pub mod gadgets;
/// Container with all of the range-check related gadgets.
pub use gadgets::scalar_gadgets::range as RangeGadgets;
/// Container with all of the pure-Scalar gadgets.
pub use gadgets::scalar_gadgets::scalar as ScalarGadgets;
/// Re-export of GadgetErrors.
pub use gadgets::GadgetErrors;
