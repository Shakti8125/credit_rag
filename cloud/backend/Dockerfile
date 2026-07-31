# Cloud tier container image.
#
# BUILD FROM THE PROJECT ROOT so shared/ is inside the build context:
#   docker build -f cloud/backend/Dockerfile -t creditrag-backend .
#
# Run locally:
#   docker run --env-file .env -p 8000:8000 creditrag-backend
#
# For AWS Lambda deployment, app.main exposes `handler` (Mangum); swap the
# base image for public.ecr.aws/lambda/python and set CMD to app.main.handler.

FROM python:3.11-slim

WORKDIR /srv

# Install dependencies first for layer caching
COPY cloud/backend/requirements.txt /srv/requirements.txt
RUN pip install --no-cache-dir -r /srv/requirements.txt

# shared/ sits next to app/ so `import shared` and `import app` both resolve
# from /srv without any sys.path manipulation.
COPY shared/ /srv/shared/
COPY cloud/backend/app/ /srv/app/

# Pre-download models at build time so cold starts don't pay the download
# (cross-encoder ~90MB, MiniLM ~80MB).
RUN python -c "from sentence_transformers import CrossEncoder, SentenceTransformer; \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
    SentenceTransformer('all-MiniLM-L6-v2')"

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
