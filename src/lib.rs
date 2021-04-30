// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (c) DUSK NETWORK. All rights reserved.

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
// We need to perform bitshifting sometimes.
#![allow(clippy::suspicious_arithmetic_impl)]
// We have cases where we just have `Variables` which don't have any descriptive name.
#![allow(clippy::many_single_char_names)]
#![deny(missing_debug_implementations)]
#![deny(missing_docs)]
#![deny(unsafe_code)]
#![no_std]

extern crate alloc;

pub(crate) mod allocated_scalar;
pub mod errors;
pub mod range;
pub mod scalar;

pub use crate::errors::Error;
pub use allocated_scalar::AllocatedScalar;
pub use range as RangeGadgets;
pub use scalar as ScalarGadgets;
