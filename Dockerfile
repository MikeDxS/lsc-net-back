FROM python:3.8
WORKDIR /work/src/app
COPY . .
EXPOSE 8000
RUN pip install -r requirements.txt
CMD ["uvicorn", "--port", "8000", "--host", "0.0.0.0", "server:app"]