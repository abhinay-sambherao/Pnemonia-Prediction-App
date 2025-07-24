# Production Notes

- Always use HTTPS in production.
- Serve static files (css, js, images) with a real web server (e.g., nginx, Apache), not Flask's built-in server.
- Use a WSGI server (e.g., Gunicorn, uWSGI) to run the Flask app in production.
- Set all secrets and sensitive config values via environment variables (see .env.example).
- Regularly update dependencies to patch vulnerabilities.
- Backup your database regularly and use a production-grade DB for real deployments.
- Sanitize logs and avoid logging PII (personally identifiable information).
- Enforce file upload size and type restrictions.
- Use Flask-Limiter and Flask-Talisman for security and rate limiting.
