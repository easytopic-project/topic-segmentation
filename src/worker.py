from data_structures import Shot
import numpy as np
from scipy.io import wavfile
import re
from sys import argv
import os, sys
import glob
import evaluate_method
import multiprocessing
import time
import random
import json
sys.path.insert(0, 'document_similarity/')
from document_similarity import DocSim
from gensim.models.keyedvectors import KeyedVectors
from genetic_algorithm import GA
from aubio import source
from aubio import pitch as pt
import pika
import time
import os
import multiprocessing
import json
import logging
import ast
import threading
import functools
from files_ms_client import upload, download


import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

stopwords = None
googlenews_model_path = '/word2vec/GoogleNews-vectors-negative300.bin'
stopwords_path = "src/document_similarity/data/stopwords_en.txt"
docSim = None
with open(stopwords_path, 'r') as fh:
    stopwords = fh.read().split(",")
model = KeyedVectors.load_word2vec_format(googlenews_model_path, binary=True, limit=1000000)
docSim = DocSim.DocSim(model, stopwords=stopwords)


class Summary:
    def __init__(self, video_path):
        self.video_path = video_path
        self.video_file = None
        self.chunks_path = self.video_path + "chunks/"
        self.n_chunks = len(glob.glob(self.chunks_path+ "chunk*"))
        self.chunks = []
        self.video_length = 0


    '''Method that create a audio chunk object passing the extracted features'''
    def createShots(self, i, pause, ocr_on, time,end_time,  docSim, prosodic_file):
        pitch = 0
        volume = 0
        try:
            with open(prosodic_file) as f:
                data = json.load(f)
                pitch = float(data[str(i)][0])
                volume = float(data[str(i)][1])

        except FileNotFoundError:
            print('Prosodic features not found')

        s = Shot(i, pitch, volume, pause, [], init_time=time, end_time=end_time)

        s.extractTranscriptAndConcepts(self.video_path, ocr_on, docSim=docSim)

        return s

LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -30s %(funcName) '
              '-35s %(lineno) -5d: %(message)s')
LOGGER = logging.getLogger(__name__)

FILES_SERVER = os.environ.get("FILES_SERVER", "localhost:3001") 
QUEUE_SERVER_HOST, QUEUE_SERVER_PORT = os.environ.get("QUEUE_SERVER", "localhost:5672").split(":")
Q_IN = os.environ.get("INPUT_QUEUE_NAME", "topic_segmentation_in")
Q_OUT = os.environ.get("OUTPUT_QUEUE_NAME", "topic_segmentation_out")

def callback(channel, method, properties, body, args):

    (connection, threads) = args
    delivery_tag = method.delivery_tag
    t = threading.Thread(target=do_work, args=(connection, channel, delivery_tag, body))
    t.start()
    threads.append(t)


def do_work(connection, channel, delivery_tag, body):
    try:
        print(" [x] Received %r" % body, flush=True)
        args = json.loads(body) 
        llf = download(args['llf']['name'], url="http://" + FILES_SERVER, buffer=True)
        asr = download(args['asr']['name'], url="http://" + FILES_SERVER, buffer=True)

        chunks = []
        low_features_dict = ast.literal_eval(llf.decode('utf-8'))
        asr_dict = ast.literal_eval(asr.decode('utf-8'))
        print(low_features_dict, flush=True)
        print(asr_dict, flush=True)
        for k, v in low_features_dict.items():
            s = Shot(k, low_features_dict[k]['pitch'], low_features_dict[k]['volume'],
                     low_features_dict[k]['pause'], [], init_time=low_features_dict[k]['init_time'], end_time=0)
            s.extractTranscriptAndConcepts(asr_dict[k], False, docSim=docSim)
            chunks.append(s)
        # print(result['low_level_features'], flush=True)
        # print(result['asr'], flush=True)
        chunks = [s for s in chunks if s.valid_vector]
        if len(chunks) < 2:
            boundaries = [0]
        else:
            '''calls the genetic algorithm'''
            ga = GA.GeneticAlgorithm(population_size=100, constructiveHeuristic_percent=0.3, mutation_rate=0.05,
                                     cross_over_rate=0.4, docSim=docSim, shots=chunks,
                                     n_chunks=len(chunks), generations=500, local_search_percent=0.3,
                                     video_length=100, stopwords=stopwords, ocr_on=False)
            boundaries = ga.run()
        #print(chunks, flush=True)
        print(boundaries, flush=True)
        topics = {}
        topics["topics"] = boundaries
        payload = bytes(str(topics), encoding='utf-8')

        uploaded = upload(payload, url="http://" + FILES_SERVER, buffer=True, mime='text/json')
        message = {
                **args,
                'topic-segmentation-output': uploaded
                }
        connection_out = pika.BlockingConnection(
            pika.ConnectionParameters(host=QUEUE_SERVER_HOST, port=QUEUE_SERVER_PORT))
        channel2 = connection_out.channel()

        channel2.queue_declare(queue=Q_OUT, durable=True)
        channel2.basic_publish(
            exchange='', routing_key=Q_OUT, body=json.dumps(message))


    except Exception as e:
        # print(e, flush=True)
        print('Connection Error %s' % e, flush=True)
        
    print(" [x] Done", flush=True)
    cb = functools.partial(ack_message, channel, delivery_tag)
    connection.add_callback_threadsafe(cb)

def ack_message(channel, delivery_tag):
    """Note that `channel` must be the same pika channel instance via which
    the message being ACKed was retrieved (AMQP protocol constraint).
    """
    if channel.is_open:
        channel.basic_ack(delivery_tag)
    else:
        # Channel is already closed, so we can't ACK this message;
        # log and/or do something that makes sense for your app in this case.
        pass

def consume():
    logging.info('[x] start consuming')
    success = False
    while not success:
        try:
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(host=QUEUE_SERVER_HOST, port=QUEUE_SERVER_PORT, heartbeat=5))
            channel = connection.channel()
            success = True
        except:
            time.sleep(30)

            pass


    channel.queue_declare(queue=Q_IN, durable=True)
    channel.queue_declare(queue=Q_OUT, durable=True)
    print(' [*] Waiting for messages. To exit press CTRL+C', flush=True)
    channel.basic_qos(prefetch_count=1)

    threads = []
    on_message_callback = functools.partial(callback, args=(connection, threads))
    channel.basic_consume(queue=Q_IN, on_message_callback=on_message_callback)
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        channel.stop_consuming()

    # Wait for all to complete
    for thread in threads:
        thread.join()

    connection.close()

consume()
