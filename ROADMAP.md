# LASSO Regression Library - Development Roadmap

> **Strategic Approach**: Build production-ready ML library with sklearn compatibility

**Last Updated**: 2025-11-25 | **Current Version**: v0.2.0 (develop) | **Target**: v1.0.0 stable

---

## Vision

Build a **production-ready, pure Go LASSO/Elastic Net regression library** with comprehensive regularization options, cross-validation support, and sklearn-compatible API.

### Key Advantages

- **Pure Go Implementation**
  - No CGO dependencies
  - Cross-platform (Linux, macOS, Windows)
  - Easy deployment and integration

- **sklearn Compatibility**
  - Lambda scaled by nSamples (sklearn convention)
  - Population variance normalization
  - Familiar API patterns

- **Performance Optimized**
  - Sequential coordinate descent (cache-friendly)
  - RawMatrix() for direct slice access
  - Minimal allocations per iteration

- **Production Features**
  - Elastic Net (L1+L2 regularization)
  - K-fold cross-validation
  - Model serialization (Save/Load)
  - Input validation (NaN/Inf detection)

---

## Version Strategy

### Philosophy: MVP → Feature Complete → Community Feedback → Stable

```
v0.1.0 (Initial) → Basic LASSO with coordinate descent
         ↓
v0.2.0 (develop) → Elastic Net + CV + Serialization + Optimizations
         ↓ (testing & feedback)
v0.3.0 → Context support + additional regularization methods
         ↓ (API stabilization)
v1.0.0-rc.1 → Feature freeze, API locked
         ↓ (community feedback, 2+ months testing)
v1.0.0 STABLE → Production release
```

**Important Notes**:
- **v1.0.0** requires community feedback and API freeze
- Pre-1.0 versions may have API changes
- Semantic versioning strictly followed

---

## Current Status (v0.2.0 - develop)

### What's New in v0.2.0

**Algorithm Improvements** (100%):
- Lambda scaling by nSamples (sklearn compatibility)
- Sequential coordinate descent (removed fake parallelism)
- RawMatrix() optimization for gonum matrices
- Reduced per-iteration allocations with predictInto()
- Population variance (n) instead of sample variance (n-1)

**New Features** (100%):
- Elastic Net regularization with Alpha parameter (0=Ridge, 1=LASSO)
- K-fold cross-validation for automatic lambda selection
- Model serialization (Save/Load to JSON)
- Input validation for NaN/Inf values
- Error returns instead of panic

**Quality Metrics** (v0.2.0):
- Test coverage: 91.9%
- Tests: 46 passing (100%)
- Linter: 0 errors, 0 warnings
- Race detector: 0 races detected
- CI/CD: All platforms GREEN

**Known Limitations** (documented):
- Single-threaded execution (sequential coordinate descent)
- JSON serialization only (no binary format)
- No sparse matrix support

---

## Development Phases

### **Phase 1: v0.1.0 - Initial Implementation** COMPLETE

**Goal**: Basic LASSO regression with coordinate descent

**Deliverables**:
1. Parallel coordinate descent algorithm
2. Feature standardization
3. Early stopping
4. Training history tracking
5. R², MSE, MAE metrics
6. Basic tests

**Status**: RELEASED

---

### **Phase 2: v0.2.0 - Production Quality** COMPLETE

**Goal**: sklearn compatibility and production features

**Deliverables**:
1. Lambda scaling fix (Critical)
2. Sequential coordinate descent (Performance)
3. RawMatrix() optimization (Performance)
4. Reduced allocations (Performance)
5. Error handling (Robustness)
6. Elastic Net regularization (Feature)
7. Cross-validation (Feature)
8. Model serialization (Feature)
9. Numerical stability (Robustness)

**Status**: IN DEVELOP BRANCH

---

### **Phase 3: v0.3.0 - Advanced Features** PLANNED

**Goal**: Extended functionality

**Planned Features**:
1. Context support (cancellable operations)
2. Weighted samples
3. Custom loss functions
4. Warm start for incremental fitting
5. Feature importance scores
6. Regularization path (lambda sequence)

**Duration**: 2-4 weeks

---

### **Phase 4: v0.4.0 - Performance & Scalability** PLANNED

**Goal**: Handle large datasets efficiently

**Planned Features**:
1. Sparse matrix support (gonum/sparse)
2. Memory-mapped data for large datasets
3. Mini-batch coordinate descent
4. GPU acceleration (optional, via OpenCL)
5. Distributed training (optional)

**Duration**: 1-2 months

---

### **Phase 5: v1.0.0 - Production Stable** PLANNED

**Goal**: Production-ready library

**Requirements**:
- Stable for 2+ months
- No critical bugs
- Community feedback positive
- Test coverage >80%
- Documentation complete
- Benchmarks documented

**Guarantees**:
- API stability (no breaking changes in v1.x.x)
- Long-term support
- Semantic versioning

---

## Feature Support Roadmap

| Feature | v0.1.0 | v0.2.0 | v0.3.0 | v1.0.0 |
|---------|--------|--------|--------|--------|
| **LASSO regression** | ✅ | ✅ | ✅ | ✅ |
| **Feature standardization** | ✅ | ✅ | ✅ | ✅ |
| **Early stopping** | ✅ | ✅ | ✅ | ✅ |
| **Training history** | ✅ | ✅ | ✅ | ✅ |
| **R², MSE, MAE metrics** | ✅ | ✅ | ✅ | ✅ |
| **Elastic Net (L1+L2)** | ❌ | ✅ | ✅ | ✅ |
| **Cross-validation** | ❌ | ✅ | ✅ | ✅ |
| **Model serialization** | ❌ | ✅ | ✅ | ✅ |
| **Input validation** | ❌ | ✅ | ✅ | ✅ |
| **sklearn lambda scaling** | ❌ | ✅ | ✅ | ✅ |
| **Context support** | ❌ | ❌ | ✅ | ✅ |
| **Weighted samples** | ❌ | ❌ | ✅ | ✅ |
| **Sparse matrices** | ❌ | ❌ | ❌ | ✅ |
| **Regularization path** | ❌ | ❌ | ✅ | ✅ |

**Legend**:
- ✅ Implemented
- ❌ Not implemented

---

## Current Focus (v0.2.0 → Release)

### Immediate Priorities

**Current Status**: v0.2.0 in develop branch

**Planned Work**:
1. **Testing**
   - Additional edge case tests
   - Performance benchmarks documentation
   - Comparison with sklearn results

2. **Documentation**
   - API reference updates
   - Usage examples
   - Performance tips

3. **Release Preparation**
   - Final code review
   - CHANGELOG update
   - Version tag

---

## Dependencies

**Required**:
- Go 1.25+
- gonum.org/v1/gonum v0.15.1 (matrix operations)

**Development**:
- golangci-lint v2.5+ (code quality)
- GitHub Actions (CI/CD)

**Testing**:
- Python sklearn (for comparison testing)

---

## Development Approach

**Algorithm Design**:
- Sequential coordinate descent for cache efficiency
- Soft thresholding for L1 regularization
- Elastic Net mixing for L1+L2 combination

**Testing Strategy**:
- Unit tests for all components
- Integration tests (fit → predict → evaluate)
- Comparison tests against sklearn
- Performance benchmarks
- Target: >70% coverage

**Quality Assurance**:
- golangci-lint with ML-specific configuration
- Comprehensive CI/CD (Linux, macOS, Windows)
- Pre-release check script
- Code review

---

## Support

**Documentation**:
- README.md - Project overview and quick start
- CONTRIBUTING.md - Development guide
- CHANGELOG.md - Release history
- ROADMAP.md - This file

**Community**:
- GitHub Issues - Bug reports and feature requests
- Repository: https://github.com/causalgo/lasso

---

## Out of Scope

The following features are **not planned**:

- ❌ Classification (use logistic regression libraries)
- ❌ Deep learning integration
- ❌ Automatic feature selection beyond L1
- ❌ Bayesian LASSO (different algorithm family)
- ❌ Group LASSO (may be separate library)

---

*Version 1.0*
*Current: v0.2.0 (develop) | Next: v0.3.0 | Target: v1.0.0*
