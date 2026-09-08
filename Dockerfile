#taking base image
FROM python 3.13.11-slim-buster 

WORKDIR /app
COPY ./app

#updating packages
RUN apt update-y && apt install awscli -y


RUN pip install -r requirements.txt

#running
CMD ["python3","app.py"]

