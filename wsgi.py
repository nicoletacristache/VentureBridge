"""Production WSGI entry point for a deployment server."""

from venturebridge import create_app


app = create_app()

