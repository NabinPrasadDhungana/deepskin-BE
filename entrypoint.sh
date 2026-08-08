#!/bin/sh
set -e

echo "===> Triggering Django Database Migrations..."
python manage.py migrate --noinput

echo "===> Asset Aggregation Processing..."
python manage.py collectstatic --noinput

echo "===> Spinning Up Celery Worker in the Background..."
# --pool=solo is critical for cloud containers to optimize memory footprints
# and avoid fork multiplication per host core.
celery -A deepskin_backend worker --loglevel=info --pool=solo &

echo "===> Instantiating Production HTTP Gunicorn WSGI Node..."
# The 'exec' command ensures Gunicorn claims PID 1, keeping the container alive
# and routing traffic.
exec gunicorn deepskin_backend.wsgi:application \
    --bind 0.0.0.0:7860 \
    --workers 2 \
    --timeout 120 \
    --log-level info