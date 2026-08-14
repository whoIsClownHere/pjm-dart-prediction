# Security Policy

## Supported version

Security updates are applied to the latest commit on the `main` branch. Historical commits and research outputs are not maintained as separate supported releases.

## Reporting a vulnerability

Do not open a public issue for:

- exposed API keys, tokens, or other credentials;
- dependency or workflow vulnerabilities that could execute untrusted code;
- data exfiltration or unsafe handling of local files;
- a reproducible path to compromise a contributor's environment.

Use GitHub's **Report a vulnerability** option for this repository when it is available. If private vulnerability reporting is unavailable, open a minimal issue titled `Security contact request` without technical details or secrets; the maintainer will arrange a private channel through GitHub.

Include only the information needed to reproduce and assess the problem:

- affected file or dependency;
- impact;
- minimal reproduction steps;
- suggested mitigation, if known.

Please allow up to seven days for an initial response. This project does not currently operate a bug-bounty program.

## Credentials

The project reads `GRIDSTATUS_API_KEY` from a local environment file. Never commit a real key. If a credential is exposed, revoke and rotate it immediately, then remove it from Git history before publishing any replacement.

## Research and model issues

Data-quality problems, point-in-time leakage, incorrect market assumptions, and disagreements with backtest methodology are important but are not security vulnerabilities. Report them with the **Research / reproducibility issue** template.
