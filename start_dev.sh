#!/bin/bash
# Starts Redis (if not already running), Celery worker, and Django dev server.
# Usage: ./start_dev.sh
redis-server --daemonize yes 2>/dev/null

celery -A deepskin_backend worker --loglevel=info &
CELERY_PID=$!

trap "kill $CELERY_PID 2>/dev/null" EXIT
echo "Celery worker started (PID: $CELERY_PID)"
echo "Starting Django dev server..."
python manage.py runserver