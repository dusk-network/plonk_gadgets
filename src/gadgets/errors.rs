use thiserror::Error;

/// Represents an error during the execution of one of the library gagets.
#[derive(Error, Debug)]
pub enum GadgetErrors {
    /// Error returned when we try to compute the inverse of a number which is
    /// non-QR (doesn't have an inverse inside of the field)
    #[error("error on the computation of an inverse")]
    NonExistingInverse,
}
