VentureBridge Website
=====================

English
-------

VentureBridge is a Flask-based entrepreneurship education website. It provides
learner profiles, editable recommendations, text-based venture simulations,
evidence-oriented feedback, transparent data controls, and local SQLite
persistence. The language shown by the website is English.

System requirements
-------------------

- Python 3.11 or later
- An OpenAI-compatible API endpoint for features that use an LLM

Project structure
-----------------

- run.py: local development entry point
- wsgi.py: WSGI entry point for deployment
- openai_messages_client.py: OpenAI-compatible HTTP client
- venturebridge/: backend package
- venturebridge/templates/: HTML frontend templates
- venturebridge/static/: CSS and JavaScript frontend assets
- venturebridge/data/blueprints.json: simulation blueprint data
- requirements.txt: Python dependencies

Quick start
-----------

Run the following commands from this directory:

  python -m venv venv
  source venv/bin/activate
  python -m pip install -r requirements.txt
  export APP_SECRET_KEY="replace-with-a-long-random-value"
  export V_API_KEY="your-api-key"
  export V_API_BASE_URL="https://run.v3.cm/v1"
  export V_MODEL="qwen-plus"
  export APP_BUILD_VERSION="local-development"
  python run.py

On Windows PowerShell, activate the environment with:

  venv\Scripts\Activate.ps1

Set environment variables with $env:NAME="value" before running python run.py.

Open http://127.0.0.1:5000 in a browser. The application creates its SQLite
database automatically in the instance directory. Public pages can open without
an API key, but LLM-based actions require valid API settings.

Useful environment variables
----------------------------

- APP_ENV: development or production; defaults to development
- APP_SECRET_KEY: Flask session secret; use at least 32 characters in production
- APP_BUILD_VERSION: deployment/build identifier
- DATABASE_PATH: SQLite database path
- SESSION_COOKIE_SECURE: set to true for production HTTPS deployments
- V_API_KEY: server-side API key
- V_API_BASE_URL: HTTPS base URL of the OpenAI-compatible API
- V_MODEL: model name
- V_API_VERSION: optional immutable model or API revision
- LLM_TIMEOUT_SECONDS: request timeout; defaults to 90
- LLM_MAX_RETRIES: retry count; defaults to 2
- LLM_TEMPERATURE: generation temperature; defaults to 0.2
- LLM_MAX_TOKENS: maximum generated tokens; defaults to 2400
- ANCHOR_PLAN_PATH: path to a registered common-anchor plan
- ANCHOR_PLAN_EXPECTED_HASH: registered SHA-256 hash of that plan
- ANCHOR_PLAN_REQUIRED: whether simulation delivery requires the plan

Deployment
----------

For a production deployment, provide all secrets through the host environment
or a secret manager. Do not place real credentials in source files. Serve the
application behind HTTPS and start it with a WSGI server, for example:

  APP_ENV=production SESSION_COOKIE_SECURE=true ANCHOR_PLAN_REQUIRED=true \
    gunicorn --workers 2 --bind 127.0.0.1:8000 wsgi:app

Production mode requires a non-placeholder APP_SECRET_KEY of at least 32
characters. When ANCHOR_PLAN_REQUIRED is true, simulation delivery remains
closed until ANCHOR_PLAN_PATH and ANCHOR_PLAN_EXPECTED_HASH are both configured.

Basic checks
------------

  python -m compileall -q venturebridge run.py wsgi.py openai_messages_client.py
  node --check venturebridge/static/app.js
  python -m flask --app wsgi:app verify-audit



中文
----

VentureBridge 是一个基于 Flask 的创业教育网站，包含学习者画像、可编辑的方向
推荐、文本创业模拟、证据导向反馈、透明的数据控制和本地 SQLite 数据存储。网站
界面使用英文。

系统要求
--------

- Python 3.11 或更高版本
- 使用大语言模型功能时，需要兼容 OpenAI 接口的 API 服务

项目结构
--------

- run.py：本地开发启动入口
- wsgi.py：部署使用的 WSGI 入口
- openai_messages_client.py：兼容 OpenAI 接口的 HTTP 客户端
- venturebridge/：后端代码包
- venturebridge/templates/：HTML 前端模板
- venturebridge/static/：CSS 和 JavaScript 前端资源
- venturebridge/data/blueprints.json：模拟蓝图数据
- requirements.txt：Python 依赖列表

快速启动
--------

请在当前目录执行：

  python -m venv venv
  source venv/bin/activate
  python -m pip install -r requirements.txt
  export APP_SECRET_KEY="请替换为足够长的随机值"
  export V_API_KEY="你的API密钥"
  export V_API_BASE_URL="https://run.v3.cm/v1"
  export V_MODEL="qwen-plus"
  export APP_BUILD_VERSION="local-development"
  python run.py

Windows PowerShell 激活虚拟环境的命令是：

  venv\Scripts\Activate.ps1

在 PowerShell 中，可先用 $env:变量名="值" 设置环境变量，再执行 python run.py。

浏览器访问 http://127.0.0.1:5000。程序会自动在 instance 目录创建 SQLite
数据库。没有配置 API 密钥时，公开页面仍可访问，但依赖大语言模型的操作需要
有效的 API 配置。

常用环境变量
------------

- APP_ENV：development 或 production，默认是 development
- APP_SECRET_KEY：Flask 会话密钥；生产环境至少使用 32 个字符
- APP_BUILD_VERSION：部署或构建版本标识
- DATABASE_PATH：SQLite 数据库路径
- SESSION_COOKIE_SECURE：生产 HTTPS 环境设为 true
- V_API_KEY：服务端 API 密钥
- V_API_BASE_URL：兼容 OpenAI 接口的 HTTPS 基础地址
- V_MODEL：模型名称
- V_API_VERSION：可选的固定模型或 API 版本标识
- ANCHOR_PLAN_PATH：已登记的共同锚计划路径
- ANCHOR_PLAN_EXPECTED_HASH：计划对应的 SHA-256 登记哈希
- ANCHOR_PLAN_REQUIRED：是否强制要求计划后才允许投放模拟

生产部署
--------

生产环境应通过服务器环境变量或密钥管理服务提供机密信息，不要把真实密钥写入
源码。网站应部署在 HTTPS 之后，并使用 WSGI 服务器启动，例如：

  APP_ENV=production SESSION_COOKIE_SECURE=true ANCHOR_PLAN_REQUIRED=true \
    gunicorn --workers 2 --bind 127.0.0.1:8000 wsgi:app

生产模式要求 APP_SECRET_KEY 不是示例占位值且至少有 32 个字符。当
ANCHOR_PLAN_REQUIRED 为 true 时，必须同时配置 ANCHOR_PLAN_PATH 和
ANCHOR_PLAN_EXPECTED_HASH，之后才能正式投放模拟。

基础检查
--------

  python -m compileall -q venturebridge run.py wsgi.py openai_messages_client.py
  node --check venturebridge/static/app.js
  python -m flask --app wsgi:app verify-audit

