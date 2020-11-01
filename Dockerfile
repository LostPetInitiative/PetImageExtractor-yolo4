FROM ubuntu

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y python3.8 python3-pip && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY yolo4 /app
COPY requirements.txt /app/requirements.txt

RUN pip3 install -r requirements.txt

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests \
    -y libglib2.0-0 libsm6 libxrender1 libxtst6 libxi6 && \
    rm -rf /var/lib/apt/lists/*

ENV KAFKA_URL=kafka-cluster.kashtanka:9092
ENV INPUT_QUEUE=DistinctPhotosPetCards
ENV OUTPUT_QUEUE=DetectedPets

CMD python3.8 -u detect.py