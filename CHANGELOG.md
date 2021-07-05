# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [v0.6.0] - 06-07-21

### Add
- Add `std` feature to crate [#42](https://github.com/dusk-network/plonk_gadgets/issues/42)

### Change
- Change crate to be `no_std` by default [#42](https://github.com/dusk-network/plonk_gadgets/issues/42)
- Update `rand` from `v0.7` to `v0.8` [#42](https://github.com/dusk-network/plonk_gadgets/issues/42)

### Remove
- Remove `anyhow` and `thiserror`. [#39](https://github.com/dusk-network/plonk_gadgets/issues/39)
- Remove `rand_core` from dev-deps [#42](https://github.com/dusk-network/plonk_gadgets/issues/42)

## [v0.5.0] - 13-01-21

### Change

- Update `dusk-plonk` to `v0.5.0`

## [v0.4.3] - 10-11-20

### Change

- Update `BlsScalar` instance backend

## [0.4.2] - 02-11-20

### Changed

- Bumped dusk-plonk to v0.3.3

## [0.4.1] - 01-11-20

### Changed

- Bumped dusk-plonk to v0.3.2

## [0.4.0] - 05-10-20

### Changed

- Updated dusk-plonk to v0.3.1

## [0.3.0] - 29-09-20

### Changed

- dusk_plonk version changed to latest (v0.2.11)

## [0.2.1] - 28-08-20

### Changed

- dusk_plonk version changed to latest (v0.2.8)

## [0.2.0] - 19-08-20

### Added

- `maybe_equal` gadget
- `AllocatedScalar` helper structure.
- Lib & code docs.

### Changed

- Integration tests moved to the `tests` folder.
- Changed the external API the crate exposes.
- Rangeproof-related functions (#17)

## [0.1.0] - 17-08-20

### Added

- Range gadgets taken from the `dusk-blindbid` library.
- `Inverse rangeproof` gadget.

### Changed

- Conditional selection gadgets have been upadated.
- Switched to latest `dusk-plonk` version.

### Removed

- Removed all of the ECC-related modules.
