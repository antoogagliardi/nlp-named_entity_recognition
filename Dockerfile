FROM python:3.8.12-slim

WORKDIR /home/app

# install requirements

COPY requirements.txt .
RUN pip install -r requirements.txt

# copy model

COPY model model

# copy code

COPY docker ner
ENV PYTHONPATH ner

# standard cmd

CMD [ "python", "ner/app.py" ]
