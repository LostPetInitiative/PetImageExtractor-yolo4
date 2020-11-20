FROM lostpetinitiative/tensorflow-2-no-avx-cpu:2.3.0

WORKDIR /app
COPY requirements.txt /app/requirements.txt

RUN pip3 install -r requirements.txt

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests \
    -y libglib2.0-0 libsm6 libxrender1 libxtst6 libxi6 && \
    rm -rf /var/lib/apt/lists/*

COPY yolo4 /app

ENV KAFKA_URL=kafka-cluster.kashtanka:9092
ENV INPUT_QUEUE=kashtanka_distinct_photos_pet_cards
ENV OUTPUT_QUEUE=kashtanka_detected_pets

CMD python3.6 -u detect.py