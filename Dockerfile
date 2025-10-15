FROM python:3.10-slim-buster

WORKDIR /app

COPY . /app

RUN pip install -r requirements.tx

EXPOSE 5000

CMD [ "python3", 'app.py' ]