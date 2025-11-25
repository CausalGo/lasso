# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2025-11-25

### Added
- Elastic Net regularization with Alpha parameter (L1+L2 mix)
- K-fold cross-validation for automatic lambda selection
- Model serialization (Save/Load to JSON)
- Input validation for NaN/Inf values
- Comprehensive benchmarks
- Project infrastructure (CI/CD, linting, documentation)

### Changed
- Refactored to sequential coordinate descent (better cache locality)
- Optimized matrix access with RawMatrix() direct slice operations
- Reduced per-iteration allocations with predictInto()
- Lambda now scaled by nSamples for sklearn compatibility
- Fit() returns error instead of panic
- Updated to Go 1.25

### Fixed
- Population variance (n) instead of sample variance (n-1) for sklearn consistency

## [0.1.0] - 2025-06-08

### Added
- Initial LASSO implementation with parallel coordinate descent
- Feature standardization
- Early stopping
- Training history tracking
- R², MSE, MAE metrics

[Unreleased]: https://github.com/causalgo/lasso/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/causalgo/lasso/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/causalgo/lasso/releases/tag/v0.1.0
