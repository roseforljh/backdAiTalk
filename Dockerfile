FROM python:3.9-slim

ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app
ENV TEMP_UPLOAD_DIR_NAME temp_document_uploads

WORKDIR ${APP_HOME}




COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


RUN mkdir -p ${TEMP_UPLOAD_DIR_NAME} && \
    chmod 755 ${TEMP_UPLOAD_DIR_NAME}



COPY ./eztalk_proxy ${APP_HOME}/eztalk_proxy



ENV PORT 7860
EXPOSE ${PORT}




CMD ["uvicorn", "eztalk_proxy.main:app", "--host", "0.0.0.0", "--port", "7860"]