# Plonk Gadgets
![Build Status](https://github.com/dusk-network/plonk/workflows/Continuous%20integration/badge.svg)
[![Repository](https://img.shields.io/badge/github-plonk-blueviolet?logo=github)](https://github.com/dusk-network/plonk)
[![Documentation](https://img.shields.io/badge/docs-plonk-blue?logo=rust)](https://docs.rs/plonk/)


This library cointains the gadgets that the Dusk-Network protocol needs to build it's ZK-Circuits.
The library **contains generic gadgets** which are used across Dusk's tech stack, all of the other
gadgets used which depend on foreign types are placed on the libraries where this types are defined.


## WARNING
This implementation is not audited. Use under your own responsability.

## Content
This library provides:

- Scalar gadgets: `is_non-zero`, `maybe_equals`, `conditionally_select_one`, `conditionally_select_zero`.
- Range gadgets: `range_check`, `max_bound`.


## Acknowledgements

- Conditional selection gadgets and `AllocatedScalar` structure have been taken from the ZCash sapling
circuits and translated to the Plonk Constraint System language.

## Licensing

This code is licensed under Mozilla Public License Version 2.0 (MPL-2.0). Please see [LICENSE](https://github.com/dusk-network/plonk_gadgets/blob/master/LICENSE) for further info.

## About

Implementation designed by the [dusk](https://dusk.network) team.

## Contributing
- If you want to contribute to this repository/project please, check [CONTRIBUTING.md](https://github.com/dusk-network/plonk_gadgets/blob/master/CONTRIBUTING.md)
- If you want to report a bug or request a new feature addition, please open an issue on this repository.