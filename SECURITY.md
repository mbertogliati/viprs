# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.x.x   | ✅ Latest release only |

## Reporting a Vulnerability

If you discover a security vulnerability in viprs, please report it responsibly:

1. **Do NOT** open a public issue
2. Email: **security@bertogliati.dev** (or use [GitHub's private vulnerability reporting](https://github.com/mbertogliati/viprs/security/advisories/new))
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Impact assessment
   - Suggested fix (if any)

## Response Timeline

- **Acknowledgment**: within 48 hours
- **Initial assessment**: within 7 days
- **Fix or mitigation**: within 30 days for critical issues

## Scope

Security issues in viprs include:
- Memory safety violations (buffer overflows, use-after-free)
- Denial of service via crafted input images
- Unsafe code that violates documented invariants
- Dependency vulnerabilities that affect viprs users

## Recognition

Contributors who report valid security issues will be credited in the release notes (unless they prefer to remain anonymous).
