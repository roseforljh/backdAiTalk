# ---- Builder Stage ----
FROM python:3.9-slim-buster as builder

ENV PYTHONUNBUFFERED 1

WORKDIR /app

# Install build-time dependencies if any
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential

# Install Python dependencies
COPY ../requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir --user -r requirements.txt

# ---- Final Stage ----
FROM python:3.9-slim-buster as final

ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app
ENV PATH=$APP_HOME/.local/bin:$PATH
ENV TEMP_UPLOAD_DIR=${APP_HOME}/temp_document_uploads

WORKDIR ${APP_HOME}

# Create a non-root user and group
RUN groupadd -r appgroup && useradd --no-log-init -r -g appgroup appuser

# Create the temporary upload directory and set permissions
RUN mkdir -p ${TEMP_UPLOAD_DIR} && \
    chown -R appuser:appgroup ${TEMP_UPLOAD_DIR} && \
    chmod -R 750 ${TEMP_UPLOAD_DIR}

# Copy installed packages from the builder stage
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code
# We do not copy the .env file into the image.
# Configuration should be passed via environment variables at runtime.
COPY ../eztalk_proxy ./eztalk_proxy
COPY ../run.py .

# Copy the entrypoint script and make it executable
COPY ./deployment/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && \
    chown appuser:appgroup /entrypoint.sh

# Ensure the .local directory is owned by the appuser
RUN chown -R appuser:appgroup /home/appuser/.local

# Expose the default port the application runs on.
# This can be overridden by setting the PORT environment variable.
EXPOSE 7860

# Switch to the non-root user
USER appuser

# Set the entrypoint and default command
ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "run:app", "--host", "0.0.0.0", "--port", "7860"]