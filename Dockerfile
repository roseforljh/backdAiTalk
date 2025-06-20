FROM python:3.9-slim-buster

ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app
ENV TEMP_UPLOAD_DIR name temp_document_uploads


RUN mkdir -p ${APP_HOME}/temp_document_uploads

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . ${APP_HOME}/ 


WORKDIR ${APP_HOME}

CMD ["uvicorn", "eztalk\_proxy.main:app", "--host", "0.0.0.0", "--port", "8880"]