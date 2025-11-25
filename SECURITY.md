# Security Policy

## Supported Versions

LASSO Regression Library is currently in active development. We provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| 0.1.x   | :x:                |
| < 0.1.0 | :x:                |

Future stable releases (v1.0+) will follow semantic versioning with LTS support.

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability in LASSO Regression Library, please report it responsibly.

### How to Report

**DO NOT** open a public GitHub issue for security vulnerabilities.

Instead, please report security issues by:

1. **Private Security Advisory** (preferred):
   https://github.com/CausalGo/lasso/security/advisories/new

2. **Email** to maintainers:
   Create a private GitHub issue or contact via discussions

### What to Include

Please include the following information in your report:

- **Description** of the vulnerability
- **Steps to reproduce** the issue
- **Affected versions** (which versions are impacted)
- **Potential impact** (DoS, information disclosure, etc.)
- **Suggested fix** (if you have one)
- **Your contact information** (for follow-up questions)

### Response Timeline

- **Initial Response**: Within 48-72 hours
- **Triage & Assessment**: Within 1 week
- **Fix & Disclosure**: Coordinated with reporter

We aim to:
1. Acknowledge receipt within 72 hours
2. Provide an initial assessment within 1 week
3. Work with you on a coordinated disclosure timeline
4. Credit you in the security advisory (unless you prefer to remain anonymous)

---

## Security Considerations for ML Libraries

LASSO Regression Library processes numerical data for machine learning. This introduces specific security considerations.

### 1. Input Validation

**Risk**: Malicious or malformed input data can cause crashes or unexpected behavior.

**Attack Vectors**:
- NaN/Inf values in input matrices
- Extremely large dimension arrays (memory exhaustion)
- Mismatched dimensions between X and y

**Mitigation in Library**:
- ✅ NaN/Inf detection in input data
- ✅ Dimension validation before processing
- ✅ Error returns instead of panics

**User Recommendations**:
```go
// ✅ GOOD - Library validates inputs
model := lasso.New()
err := model.Fit(X, y)
if err != nil {
    // Handle error - could be validation failure
    log.Printf("Fit failed: %v", err)
    return err
}

// ✅ GOOD - Pre-validate your data
if containsNaN(X) || containsNaN(y) {
    return errors.New("input contains NaN values")
}
```

### 2. Numerical Stability

**Risk**: Numerical operations can overflow, underflow, or produce incorrect results.

**Attack Vectors**:
- Extremely large coefficient values
- Division by zero in standardization
- Floating-point precision issues

**Mitigation**:
- ✅ Safe division checks (variance != 0)
- ✅ Coefficient bounds checking
- ✅ Population variance calculation (sklearn compatible)

**Current Protections**:
```go
// Safe standardization
if variance > 0 {
    standardized = (value - mean) / sqrt(variance)
}

// Soft thresholding with safe bounds
if math.Abs(rho) > lambda {
    coef = sign(rho) * (math.Abs(rho) - lambda)
}
```

### 3. Model Serialization

**Risk**: Loading untrusted model files could be exploited.

**Attack Vectors**:
- Malformed JSON in Save/Load
- Extremely large serialized models (memory exhaustion)
- Type confusion in deserialization

**Mitigation**:
- ✅ JSON encoding (standard library, well-tested)
- ✅ Type validation during Load
- ✅ Dimension consistency checks

**User Recommendations**:
```go
// ❌ BAD - Don't load untrusted model files without validation
model := &lasso.LASSO{}
err := model.Load(untrustedFile)

// ✅ GOOD - Validate file source and size
fileInfo, err := os.Stat(filename)
if err != nil || fileInfo.Size() > maxModelSize {
    return errors.New("invalid model file")
}

model := &lasso.LASSO{}
err = model.Load(trustedFile)
if err != nil {
    return err
}

// Validate loaded model makes sense
if model.Lambda < 0 || model.Alpha < 0 || model.Alpha > 1 {
    return errors.New("invalid model parameters")
}
```

### 4. Resource Exhaustion

**Risk**: Large datasets or many iterations can exhaust memory/CPU.

**Attack Vectors**:
- Very high MaxIterations setting
- Large input matrices
- Many cross-validation folds

**Mitigation**:
- ✅ Early stopping (convergence detection)
- ✅ Configurable iteration limits
- ✅ Per-iteration memory efficiency (predictInto)

**User Recommendations**:
```go
// ✅ GOOD - Set reasonable limits
model := lasso.New(
    lasso.WithMaxIterations(1000),  // Not 1000000
    lasso.WithTolerance(1e-4),       // Enable early stopping
)

// ✅ GOOD - Validate data size before processing
rows, cols := X.Dims()
if rows > maxRows || cols > maxCols {
    return errors.New("dataset too large")
}
```

### 5. Dependency Security

**Risk**: Vulnerabilities in dependencies could affect the library.

**Dependencies**:
- `gonum.org/v1/gonum v0.15.1` - Matrix operations

**Mitigation**:
- Using stable, well-maintained dependencies
- Regular dependency updates
- No CGO dependencies (pure Go)

---

## Security Best Practices for Users

### Input Validation

Always validate input data:

```go
// Validate matrix dimensions
rows, cols := X.Dims()
if rows == 0 || cols == 0 {
    return errors.New("empty input matrix")
}

// Validate y dimensions match X
if y.Len() != rows {
    return errors.New("dimension mismatch")
}

// Check for NaN/Inf (library does this, but you can pre-check)
for i := 0; i < rows; i++ {
    for j := 0; j < cols; j++ {
        v := X.At(i, j)
        if math.IsNaN(v) || math.IsInf(v, 0) {
            return fmt.Errorf("invalid value at (%d, %d)", i, j)
        }
    }
}
```

### Resource Limits

Set appropriate limits:

```go
// Set iteration limits
model := lasso.New(
    lasso.WithMaxIterations(1000),
    lasso.WithTolerance(1e-4),
)

// Set cross-validation limits
result, err := lasso.CrossValidate(X, y,
    lasso.WithKFolds(5),  // Not 100
    lasso.WithLambdas(lambdas[:10]),  // Reasonable lambda count
)
```

### Error Handling

Always check errors:

```go
// ❌ BAD - Ignoring errors
model.Fit(X, y)
predictions := model.Predict(XTest)

// ✅ GOOD - Proper error handling
err := model.Fit(X, y)
if err != nil {
    return fmt.Errorf("fit failed: %w", err)
}

predictions := model.Predict(XTest)
if predictions == nil {
    return errors.New("prediction failed")
}
```

---

## Security Testing

### Current Testing

- ✅ Unit tests with edge cases (empty data, NaN values)
- ✅ Input validation tests
- ✅ Cross-validation tests
- ✅ Linting with security-focused linters
- ✅ Race detector (0 data races)

### Planned for v1.0

- Fuzzing with go-fuzz
- Static analysis with gosec
- SAST scanning in CI

---

## Security Contact

- **GitHub Security Advisory**: https://github.com/CausalGo/lasso/security/advisories/new
- **Public Issues** (for non-sensitive bugs): https://github.com/CausalGo/lasso/issues

---

**Thank you for helping keep LASSO Regression Library secure!**

*Security is a priority. We continuously improve our security posture with each release.*
