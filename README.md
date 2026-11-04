# VentureBridge

VentureBridge is an entrepreneurship education simulation system built with Flask and an LLM API. It provides learner profiles, recommendations for entrepreneurial directions, ongoing venture scenarios, decision feedback, simulation history, and user data export and deletion.

This repository contains application source code, page templates, static assets, runtime requirements, built-in event rules, and a separate method-validation package with benchmark inputs, comparison code, and saved results. `venturebridge/data/blueprints.json` contains static rule definitions, with no user records or experimental results. The application creates a local database on first startup.

## Requirements and Installation

Python 3.11 or later is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Local Configuration

Configure the service through environment variables or a locally created `.env` file in the project root. No credentials file is included. Credential fields below are intentionally blank; configure them locally before use.

```dotenv
APP_ENV=development
APP_SECRET_KEY=
APP_BUILD_VERSION=local-development
V_API_KEY=
V_API_BASE_URL=
V_MODEL=
V_API_VERSION=
```

- `APP_SECRET_KEY`: A randomly generated session secret. Generate one locally with `python -c "import secrets; print(secrets.token_hex(32))"`.
- `V_API_KEY`: The model service credential. Set it only in your local environment or deployment platform.
- `V_API_BASE_URL`: The HTTPS API base URL of a service compatible with OpenAI Chat Completions.
- `V_MODEL`: A model name supported by your provider.
- `V_API_VERSION`: The model or API version supplied by your provider. If omitted, the health check reports an audit warning.
- `APP_BUILD_VERSION`: An identifier for the deployed application build.

Optional settings include `DATABASE_PATH`, `LLM_TIMEOUT_SECONDS`, `LLM_MAX_RETRIES`, `LLM_TEMPERATURE`, and `LLM_MAX_TOKENS`. By default, the database is stored locally at `instance/venturebridge.sqlite3`.

## Running the Application

```bash
python run.py
```

Open <http://127.0.0.1:5000>. The database schema is initialized automatically. Without model service configuration, the home page, transparency page, and `/health` remain available; operations that require the model return an explicit configuration error.

The production WSGI entry point is `wsgi:app`. For deployment, set `APP_ENV=production`, provide a random session secret of at least 32 characters, set `SESSION_COOKIE_SECURE=true`, and serve the application over HTTPS. Production simulations also require a valid local `ANCHOR_PLAN_PATH` and a matching `ANCHOR_PLAN_EXPECTED_HASH`. Simulation delivery remains blocked until a valid plan is configured. Deployment operators must keep plan files and runtime data private.

## Project Structure

```text
run.py                       Local development entry point
wsgi.py                      WSGI entry point
openai_messages_client.py    Model API client
requirements.txt             Runtime dependencies
venturebridge/               Application logic
  data/blueprints.json       Built-in event rule definitions
  static/                    CSS and JavaScript
  templates/                 Page templates
```

## Method Validation

See [the method-validation guide](method_validation/README.md) for dataset preparation, comparison methods, dependencies, and result inspection. The data availability statement is in [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md); source and scope are documented in [PROVENANCE.md](method_validation/PROVENANCE.md).

## License

See [LICENSE](LICENSE) for the repository's MIT license.

## Repository Scope

`.gitignore` uses a file allowlist so that only the explicitly listed system files are included in standard Git staging. Update the allowlist when adding new application source files. Local `.env` files, databases, caches, logs, and exports are excluded from the allowlist.

This system is intended for educational simulation. Its feedback describes process evidence and does not constitute a validated ability score or real-world business advice.
