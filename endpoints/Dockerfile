FROM python:3.10.13

WORKDIR /app

COPY . .
RUN pip install -U pip
RUN pip install -r requirements.txt

EXPOSE 80

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
