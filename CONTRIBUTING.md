# Contributing to LASSO Regression Library

Thank you for considering contributing to the LASSO Regression Library! This document outlines the development workflow and guidelines.

## Git Workflow (Git-Flow)

This project uses Git-Flow branching model for development.

### Branch Structure

```
main                 # Production-ready code (tagged releases)
  └─ develop         # Integration branch for next release
       ├─ feature/*  # New features
       ├─ bugfix/*   # Bug fixes
       └─ hotfix/*   # Critical fixes from main
```

### Branch Purposes

- **main**: Production-ready code. Only releases are merged here.
- **develop**: Active development branch. All features merge here first.
- **feature/\***: New features. Branch from `develop`, merge back to `develop`.
- **bugfix/\***: Bug fixes. Branch from `develop`, merge back to `develop`.
- **hotfix/\***: Critical production fixes. Branch from `main`, merge to both `main` and `develop`.

### Workflow Commands

#### Starting a New Feature

```bash
# Create feature branch from develop
git checkout develop
git pull origin develop
git checkout -b feature/my-new-feature

# Work on your feature...
git add .
git commit -m "feat: add my new feature"

# When done, merge back to develop
git checkout develop
git merge --squash feature/my-new-feature  # Squash merge for clean history
git commit -m "feat: my new feature (squashed)"
git branch -d feature/my-new-feature
git push origin develop
```

#### Fixing a Bug

```bash
# Create bugfix branch from develop
git checkout develop
git pull origin develop
git checkout -b bugfix/fix-issue-123

# Fix the bug...
git add .
git commit -m "fix: resolve issue #123"

# Merge back to develop
git checkout develop
git merge --squash bugfix/fix-issue-123  # Squash merge for clean history
git commit -m "fix: resolve issue #123 (squashed)"
git branch -d bugfix/fix-issue-123
git push origin develop
```

#### Creating a Release

```bash
# Create release branch from develop
git checkout develop
git pull origin develop
git checkout -b release/v1.0.0

# Update version numbers, CHANGELOG, etc.
git add .
git commit -m "chore: prepare release v1.0.0"

# Merge to main and tag
git checkout main
git merge --no-ff release/v1.0.0
git tag -a v1.0.0 -m "Release v1.0.0"

# Merge back to develop
git checkout develop
git merge --no-ff release/v1.0.0

# Delete release branch
git branch -d release/v1.0.0

# Push everything
git push origin main develop --tags
```

#### Hotfix (Critical Production Bug)

```bash
# Create hotfix branch from main
git checkout main
git pull origin main
git checkout -b hotfix/critical-bug

# Fix the bug...
git add .
git commit -m "fix: critical production bug"

# Merge to main and tag
git checkout main
git merge --no-ff hotfix/critical-bug
git tag -a v1.0.1 -m "Hotfix v1.0.1"

# Merge to develop
git checkout develop
git merge --no-ff hotfix/critical-bug

# Delete hotfix branch
git branch -d hotfix/critical-bug

# Push everything
git push origin main develop --tags
```

## Commit Message Guidelines

Follow [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, etc.)
- **refactor**: Code refactoring
- **test**: Adding or updating tests
- **chore**: Maintenance tasks (build, dependencies, etc.)
- **perf**: Performance improvements

### Examples

```bash
feat: add elastic net regularization with alpha parameter
fix: correct lambda scaling for sklearn compatibility
docs: update README with cross-validation examples
refactor: optimize coordinate descent with RawMatrix
test: add benchmarks for parallel vs sequential descent
chore: update go.mod dependencies
```

## Code Quality Standards

### Before Committing

1. **Format code**:
   ```bash
   go fmt ./...
   ```

2. **Run linter**:
   ```bash
   golangci-lint run
   ```

3. **Run tests**:
   ```bash
   go test -v -race ./...
   ```

4. **All-in-one** (pre-release check):
   ```bash
   ./scripts/pre-release-check.sh
   ```

### Pull Request Requirements

- [ ] Code is formatted (`go fmt ./...`)
- [ ] Linter passes (`golangci-lint run`)
- [ ] All tests pass (`go test -v -race ./...`)
- [ ] New code has tests (minimum 70% coverage)
- [ ] Documentation updated (if applicable)
- [ ] Commit messages follow conventions
- [ ] No sensitive data (credentials, tokens, etc.)

## Development Setup

### Prerequisites

- Go 1.25 or later
- golangci-lint

### Install Dependencies

```bash
# Clone repository
git clone https://github.com/causalgo/lasso.git
cd lasso

# Download dependencies
go mod download

# Install golangci-lint (if not installed)
go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
```

### Running Tests

```bash
# Run all tests
go test -v ./...

# Run with race detector
go test -v -race ./...

# Run with coverage
go test -v -coverprofile=coverage.txt -covermode=atomic ./...

# Run benchmarks
go test -bench=. -run=^Benchmark -benchmem ./...
```

### Running Linter

```bash
# Run linter
golangci-lint run

# Run with timeout (for large codebases)
golangci-lint run --timeout=5m
```

## Project Structure

```
lasso/
├── .claude/              # AI development configuration (private)
├── .github/              # GitHub Actions, CODEOWNERS
│   └── workflows/       # CI/CD workflows
├── docs/                 # Documentation (private dev docs)
├── scripts/              # Automation scripts
│   └── pre-release-check.sh
├── cmd/                  # Command-line utilities
│   └── example/         # Example program
├── lasso.go             # Core LASSO/Elastic Net algorithm
├── lasso_test.go        # Unit tests
├── cv_test.go           # Cross-validation tests
├── .golangci.yml        # Linter configuration
├── go.mod               # Go module definition
├── go.sum               # Dependency checksums
├── LICENSE              # MIT License
├── README.md            # Main documentation
├── CHANGELOG.md         # Release history
├── CONTRIBUTING.md      # This file
└── CODE_OF_CONDUCT.md   # Community guidelines
```

## Adding New Features

1. Check if issue exists, if not create one
2. Discuss approach in the issue
3. Create feature branch from `develop`
4. Implement feature with tests
5. Update documentation
6. Run quality checks (`./scripts/pre-release-check.sh`)
7. Create pull request to `develop`
8. Wait for code review
9. Address feedback
10. Merge when approved

## Code Style Guidelines

### General Principles

- Follow Go conventions and idioms
- Write self-documenting code
- Add comments for complex mathematical operations
- Keep functions small and focused
- Use meaningful variable names

### Naming Conventions

- **Public types/functions**: `PascalCase` (e.g., `Fit`, `Predict`)
- **Private types/functions**: `camelCase` (e.g., `coordinateDescent`)
- **Constants**: `PascalCase` (e.g., `DefaultLambda`, `DefaultMaxIterations`)
- **Test functions**: `Test*` (e.g., `TestLassoFit`)

### Error Handling

- Always check and handle errors
- Use descriptive error messages
- Return errors immediately, don't wrap unnecessarily
- Validate inputs before processing (NaN/Inf checks)

### Testing

- Use table-driven tests when appropriate
- Test both success and error cases
- Test edge cases (empty data, NaN values, etc.)
- Include benchmarks for performance-critical code
- Target minimum 70% coverage

## ML Algorithm Implementation Patterns

### Numerical Stability

```go
// Always validate inputs
if hasNaNOrInf(X) || hasNaNOrInf(y) {
    return fmt.Errorf("input contains NaN or Inf values")
}

// Use safe divisions
if denominator != 0 {
    result = numerator / denominator
}
```

### Coordinate Descent

```go
// Sequential coordinate descent for better cache locality
for j := 0; j < nFeatures; j++ {
    // Update single coefficient
    rho := computeRho(X, residuals, j)
    coef[j] = softThreshold(rho, lambda)
}
```

### Convergence Checking

```go
// Check for convergence
maxCoefChange := 0.0
for j := 0; j < nFeatures; j++ {
    change := math.Abs(newCoef[j] - oldCoef[j])
    if change > maxCoefChange {
        maxCoefChange = change
    }
}

if maxCoefChange < tolerance {
    break // Converged
}
```

## Getting Help

- Check existing issues and discussions
- Read `.claude/CLAUDE.md` for architecture insights
- Review existing tests for usage examples
- Ask questions in GitHub Issues

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to the LASSO Regression Library!**
