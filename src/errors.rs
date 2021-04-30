// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (c) DUSK NETWORK. All rights reserved.

//! Gadget Errors Module.
//!
//! Includes the definitions of all of the possible errors that the gadgets
//! might encounter with toghether with it's display message implementations.

/// Represents an error during the execution of one of the library gagets.
#[derive(Debug)]
pub enum Error {
    /// Error returned when we try to compute the inverse of a number which is
    /// non-QR (doesn't have an inverse inside of the field)
    NonExistingInverse,
}
