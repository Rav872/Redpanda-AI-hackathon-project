import datetime
import logging
import os
import pickle
import time
import webbrowser
from flask import Flask, Response, json, request, jsonify, render_template, session
from producer import Producer
from kafka import KafkaConsumer
import threading
from flask_sse import sse
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import signal
import sys
import uuid
from flask_session import Session
import redis
import uuid

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL")
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
app = Flask(__name__)
app.config["REDIS_URL"] = REDIS_URL
app.register_blueprint(sse, url_prefix='/stream')


app.config['SESSION_TYPE'] = 'redis'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_REDIS'] = redis.from_url(REDIS_URL)
app.secret_key = os.getenv('SECRET_KEY', default='BAD_SECRET_KEY')
Session(app)

# Load the model
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

request_topic = "prediction_topic"
result_topic = "prediction_result_topic"

producer = Producer(request_topic)
prediction_consumer = None

is_running = True
threads = []

@app.route('/')
def home():
    sid = uuid.uuid4()
    session["sid"] = str(sid)
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    req_body = request.json
    data = []
    if req_body is not None:
        data = [float(req_body[k]) for k in req_body]

    producer.send(data)
    return jsonify({"status": "Data sent for processing"})

def request_consume():
    global is_running
    print("--->Consuming process started<----")
    kfkconsumer = KafkaConsumer(
        bootstrap_servers=[KAFKA_BOOTSTRAP_SERVERS],
        group_id="prediction_group",
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        consumer_timeout_ms=1000,
        value_deserializer=lambda m: json.loads(m.decode('ascii'))
    )

    result_producer = Producer(result_topic)

    kfkconsumer.subscribe([request_topic])
    try:
        while is_running:
            message = kfkconsumer.poll(100)
            if not message:
                time.sleep(1)
                continue

            for topic_partition, messages in message.items():
                for msg in messages:
                    request_data = msg.value
                    session_id = request_data['session_id']
                    data = request_data['data']

                    final_input = scalar.transform(np.array(data).reshape(1,-1))
                    prediction = regmodel.predict(final_input)[0]
                    
                    output = {
                        "session_id": session_id,
                        "prediction": prediction
                    }

                    result_producer.send(output)
                    print(f"Prediction made for session {session_id}: {prediction}")

            kfkconsumer.commit()
    except Exception as e:
        print(f"Error occurred consuming: {str(e)}")
    finally:
        print("Closing consumer")
        kfkconsumer.close()


def initialize_prediction_consumer():
    global prediction_consumer
    if prediction_consumer is None:
        prediction_consumer = KafkaConsumer(
            result_topic,
            bootstrap_servers=[KAFKA_BOOTSTRAP_SERVERS],
            auto_offset_reset='latest',
            enable_auto_commit=True,
            group_id='prediction_result_group',
            consumer_timeout_ms=500,
            value_deserializer=lambda m: json.loads(m.decode('ascii'))
        )


def prediction_response():
    global prediction_consumer
    
    try:
        if prediction_consumer is None:
            initialize_prediction_consumer()

        messages = prediction_consumer.poll(timeout_ms=100)

        if messages:
            with app.app_context():
                for topic_partition, msgs in messages.items():
                    for msg in msgs:
                        data = msg.value

                        session_id = data.get('session_id')
                        if session_id:
                            sse.publish(data, type='prediction')

    except Exception as e:
        print(f"Error in prediction_response: {str(e)}")
        prediction_consumer = None


def scheduler_thread():
    scheduler = BackgroundScheduler(daemon=True)
    scheduler.add_job(
        prediction_response,
        'interval',
        seconds=2,  
        misfire_grace_time=1, 
        coalesce=True,
        max_instances=1,
        id='prediction_job'
    )
    scheduler.start()

    while is_running:
        time.sleep(1)
    
    scheduler.shutdown()

def start_background_threads():
    consumer_thread = threading.Thread(target=request_consume, daemon=True)
    consumer_thread.start()
    threads.append(consumer_thread)

    sched_thread = threading.Thread(target=scheduler_thread, daemon=True)
    sched_thread.start()
    threads.append(sched_thread)

def signal_handler(signum, frame):
    global is_running
    print("Stopping threads...")
    is_running = False

    for thread in threads:
        thread.join()
    sys.exit(0)

if __name__ == '__main__':
    print("DEBUG: In starting of the app.py")
    threading.Thread(target=request_consume, daemon=True).start()
    threading.Thread(target=scheduler_thread, daemon=True).start()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    start_background_threads()

    app.run(debug=False, use_reloader=False, host="0.0.0.0", port=5001)
    
    if prediction_consumer:
        prediction_consumer.close()
    producer.clean()
