FROM python:3.8

WORKDIR /project

COPY requirements.txt ./requirements.txt

RUN pip3 install -r requirements.txt

EXPOSE 8501

COPY /app  /project/app
COPY /data /project/data
COPY /src /project/src
COPY /models /project/models
COPY run.sh /project/app/run.sh

RUN chmod a+x /project/app/run.sh

WORKDIR /project/app

CMD ["./run.sh"]