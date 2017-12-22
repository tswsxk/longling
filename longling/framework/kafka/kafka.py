#!/usr/bin/env python
#coding: utf-8

import logging
import socket
import struct
import time

import fcntl
from pyutil.program import metrics2 as metrics
from pyutil.program.conf import Conf

import kafka


class KafkaProxy(object):

    def __init__(self, topic, codec='snappy', buffer_size=None, partitions=None, \
            consumer_group=None, conf=None, debug=False, timeout=120, \
            connect_timeout=1, connect_retry=10, retry_num=3, key_hash=False, \
            cluster_name=None, req_acks=1, socket_buffer_size=8192, \
            is_get_partition_info=False, retry_partition=False, partitioner=None, \
            auto_commit=True):
        '''
        topic:                  the topic name
        codec:                  the transmission compression algorithm, snappy by default
        buffer_size:            Initial number of bytes to tell kafka we have available.
                                This will double as needed. default 4K.
        partitions:             An optional list of partitions to consume the data from
        consumer_group:         A name for this consumer, used for offset storage and must be unique
        conf:                   The configuration file path. '/opt/tiger/ss_conf/ss/kafka.conf' by default
        debug:                  Currently unused
        timeout:                Socket timeout in seconds, it also used by the consumer to
                                indicate that how much time (in seconds) to wait for a message in
                                the iterator before exiting.
        connect_timeout:        Broker connection timeout in seconds
        connect_retry:          retry times to connect a broker
        retry_num:              retry times to produce a message. NOTE: this is only used in sync mode
        key_hash:               whether use the `hash by key` to choose a partition, false by default
        cluster_name:           The kafka cluster name, None means choose the appropriate cluster intelligently.
                                NOTE: you must be sure that the topic is globally unique if you let the cluster_name None.
        req_acks:               A value indicating the acknowledgements that the server must receive before responding to the request
                                0 means do not wait for the acknowledgements at all
                                1 means only wait for the leader's acknowledgement
                                -1 means wait all isr's acknowledgements
                                1 by default
        socket_buffer_size:     Socket buffer size by default
        is_get_partition_info:  Whether get partition meta information
        retry_partition:        Whether send the message to another partition again
                                when the chosen partition is something wrong(e.g. timedout or broke down).
        partitioner:            A partitioner class that will be used to get the partition to
                                send the message to. Must be derived from Partitioner
        auto_commit:            whether enable auto commit offset
        '''
        self.conf = conf
        self.topic = topic
        self.codec = codec
        self.consumer_group = consumer_group
        self.buffer_size = buffer_size
        self.socket_buffer_size = socket_buffer_size
        self.debug = debug
        self.key_hash = key_hash
        self.kafka_client = None
        if cluster_name is None:
            from kafka_router import get_cluster
            cluster_name = get_cluster(topic)
        self.original_cluster_name = cluster_name
        self.original_topic = topic
        self.cluster_name = cluster_name
        self.redirect_version = -1
        self.producer = None
        self.consumer = None
        self.timeout = timeout  # 执行超时时间
        self.connect_timeout = connect_timeout  # 连接超时时间
        self.connect_retry = connect_retry      # 连接重试最大次数
        self.retry_num = retry_num    # 发送数据失败重试最大次数
        self.req_acks = req_acks
        self.is_read_only = None
        self.is_get_partition_info = is_get_partition_info
        self.auto_commit = auto_commit
        if timeout == None:
            self.timeout = 120
        self.partition_int = -1
        if partitions == None: #血淋淋的教训， 之前使用if not partitions, 无法正确处理partitions=0的情况
            self.partitions = None
        else:
            if not isinstance(partitions, list):
                self.partitions = []
                self.partitions.append(int(partitions))
                self.partition_int = int(partitions)
            else:
                self.partitions = partitions
        self.retry_partition = retry_partition
        self.partitioner = partitioner
        self.METRICS_PREFIX='inf.kafka.client.' + self.cluster_name
        self.tagkv = {"topic": self.topic}
        self.consumer_tagkv = {'topic': self.topic}
        if self.consumer_group:
            self.consumer_tagkv['consumer_group'] = self.consumer_group
            metrics.define_tagkv('consumer_group', [self.consumer_group, ])
        else:
            self.consumer_tagkv['consumer_group'] = 'None'
            metrics.define_tagkv('consumer_group', ['None', ])
        self.client_id_suffix = '|kafka-python|%s|%s-%s' % (self._get_ip_address(), time.strftime('%Y-%m-%d-%H-%M-%S'),
                                                            ("%.9f" % time.time()).split('.')[1])
        self.client_id = 'produce-group' + self.client_id_suffix
        metrics.define_tagkv('topic', [self.topic, ])
        metrics.define_counter('send.success', prefix=self.METRICS_PREFIX) # it means number of succeeded messages to produce
        metrics.define_counter('send.fail', prefix=self.METRICS_PREFIX) # it means number of failed messages to produce
        metrics.define_timer('send.time', prefix=self.METRICS_PREFIX, units='millisecond')
        metrics.define_counter('recv.success', prefix=self.METRICS_PREFIX) # it means success times of calling fetch method
        metrics.define_counter('recv.fail', prefix=self.METRICS_PREFIX) # it means failed times of calling fetch method
        metrics.define_timer('recv.time', prefix=self.METRICS_PREFIX, units='millisecond')

    def __del__(self):
        self._close_client()

    def get_kafka_client(self):
        return self._check_kafka_client(read_only=True)

    def _close_client(self):
        try:
            if self.consumer and self.auto_commit:
                self.consumer.commit()
            if self.kafka_client:
                self.kafka_client.close()
        except Exception:
            logging.exception("close client exception")
        self.producer = None
        self.consumer = None
        self.kafka_client = None

    def _check_kafka_client(self, read_only=False):
        if self.is_read_only is None:
            self.is_read_only = read_only
        if self.is_read_only != read_only:
            # going to present error log only, rather than break contemporary behavior
            logging.error("Single kafka_proxy object cannot be used as both producer and consumer")
            # and, reset redirect state
            self.redirect_version = -1
            self.cluster_name = self.original_cluster_name
            self.topic = self.original_topic

        from kafka_router import get_redirection
        redirection = get_redirection(self.original_cluster_name, self.original_topic, read_only)
        if redirection is not None:
            if 'cluster_name' in redirection:
                # remember UTF8 convertion
                self.cluster_name = str(redirection['cluster_name'])
            if 'topic' in redirection:
                self.topic = str(redirection['topic'])
            version = redirection['version']
            if version > self.redirect_version:
                # update config and close client
                self.kafka_client = None
                self.producer = None
                self.consumer = None
                self.redirect_version = version
                logging.info("Using redirection (version %s): %s, %s" % (self.redirect_version, self.cluster_name, self.topic))

        if self.kafka_client:
            return self.kafka_client
        if self.conf:
            conf = self.conf
        else:
            conf = Conf('/opt/tiger/ss_conf/ss/kafka.conf')

        broker_list = []
        valid_cluster = conf.get_values("valid_cluster")
        if self.cluster_name not in valid_cluster:
            logging.error("not support cluster name %s", self.cluster_name)
            return None
        broker_list = conf.get_values(self.cluster_name)
        if not broker_list:
            logging.error("no available kafka broker found.")
            return None

        try:
            self.kafka_client = kafka.KafkaClient(host=broker_list, buffersize=self.socket_buffer_size,
                                                  timeout=self.timeout, connect_timeout=self.connect_timeout, connect_retry=self.connect_retry, client_id=self.client_id)
        except Exception:
            logging.exception('connect to kafka broker %s failed.', broker_list)
            self.kafka_client = None
        finally:
            return self.kafka_client

    def __create_producer(self, kafka_client):
        if self.key_hash:
            self.producer = kafka.KeyedProducer(kafka_client, partitioner=self.partitioner, codec=self.codec, req_acks=self.req_acks, retry_num=self.retry_num, retry_partition=self.retry_partition)
        else:
            self.producer = kafka.SimpleProducer(kafka_client, codec=self.codec, req_acks=self.req_acks, retry_num=self.retry_num, retry_partition=self.retry_partition)

    @staticmethod
    def _get_ip_address():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            ip_address = socket.inet_ntoa(fcntl.ioctl(
                s.fileno(),
                0x8915,  # SIOCGIFADDR
                struct.pack('256s', 'eth0')
            )[20:24])
        except Exception, err:
            logging.exception("get ip address error: %s", err)
            ip_address = "unknown-host"
        finally:
            s.close()
        return ip_address.replace('.', '-')

    def _emit_send_fail_metric(self, length):
        metrics.emit_counter('send.fail', length, self.METRICS_PREFIX, self.tagkv)

    def _emit_send_success_metric(self, length, tm):
        metrics.emit_counter('send.success', length, self.METRICS_PREFIX, self.tagkv)
        metrics.emit_timer('send.time', long(tm * 1000), self.METRICS_PREFIX, self.tagkv) # convert to milliseconds

    def _emit_recv_fail_metric(self):
        metrics.emit_counter('recv.fail', 1, self.METRICS_PREFIX, self.consumer_tagkv)

    def _emit_recv_success_metric(self, tm):
        metrics.emit_counter('recv.success', 1, self.METRICS_PREFIX, self.consumer_tagkv)
        metrics.emit_timer('recv.time', long(tm * 1000), self.METRICS_PREFIX, self.consumer_tagkv) # convert to milliseconds

    def write_msgs(self, msgs):
        t0 = time.time()
        kafka_client = self._check_kafka_client(read_only=False)
        if not kafka_client:
            logging.warn("no available kafka client ....")
            self._emit_send_fail_metric(len(msgs))
            return
        if not self.producer:
            self.__create_producer(kafka_client)
        try:
            if not self.key_hash:
                if isinstance(msgs, list):
                    self.producer.send_messages(self.topic, *msgs)
                else:
                    self.producer.send_messages(self.topic, msgs)
            else:
                self.producer.send_key_messages(self.topic, msgs) #msgs格式必须是list， [(key, msg), (key, msg)]
            t1 = time.time()
            self._emit_send_success_metric(len(msgs), t1 - t0)
        except Exception:
            self._emit_send_fail_metric(len(msgs))
            logging.exception('key_hash %s, save msgs to kafka failed....', self.key_hash)
            raise ValueError("kafka seems some error, please try again")

    def create_consumer(self, kafka_client):
        if "|" in self.consumer_group:
            raise ValueError("consumer group should not contain |")
        self.client_id = self.consumer_group + self.client_id_suffix
        kafka_client.client_id = self.client_id
        if self.buffer_size:
            self.consumer = kafka.SimpleConsumer(kafka_client, str(self.consumer_group), str(self.topic), buffer_size=self.buffer_size, partitions=self.partitions, auto_commit=self.auto_commit)
        else:
            self.consumer = kafka.SimpleConsumer(kafka_client, str(self.consumer_group), str(self.topic), partitions=self.partitions, auto_commit=self.auto_commit)
        if self.is_get_partition_info:
            self.consumer.provide_partition_info()

    def fetch_msgs_with_offset(self, count, block=True, timeout=0.1):
        msgs = self.__fetch_msgs(count=count, block=block, timeout=timeout)
        if msgs:
            if self.is_get_partition_info:
                return [(msg.offset, msg.message.value,partition) for partition,msg in msgs]
            else:
                return [(msg.offset, msg.message.value) for msg in msgs]
        else:
            return None

    def fetch_msgs_with_meta(self, count, block=True, timeout=0.1):
        msgs = self.__fetch_msgs(count=count, block=block, timeout=timeout)
        if msgs:
            if self.is_get_partition_info:
                return [(msg.offset, msg.message.value, partition, self.topic) for partition,msg in msgs]
            else:
                return [(msg.offset, msg.message.value, self.topic) for msg in msgs]
        else:
            return None

    def fetch_msgs(self, count=1, block=True, timeout=0.1):
        msgs = self.__fetch_msgs(count=count, block=block, timeout=timeout)
        if msgs:
            if self.is_get_partition_info:
                return [msg.message.value for partition,msg in msgs]
            else:
                return [msg.message.value for msg in msgs]
        else:
            return None

    def steal_msgs(self, partition, offset, buffersize):
        return None

    def __fetch_msgs(self, count=1, block=True, timeout=0.1):
        t0 = time.time()
        no_result = None
        kafka_client = self._check_kafka_client(read_only=True)
        if not kafka_client:
            self._emit_recv_fail_metric()
            return no_result
        if not self.consumer:
            if not self.consumer_group:
                raise ValueError("kafka proxy must set consumer group when fetch")
            try:
                self.create_consumer(kafka_client)
            except Exception:
                logging.exception("create simple consumer for kafka failed!")
                self._emit_recv_fail_metric()
                return no_result
        msgs = no_result
        try:
            msgs = self.consumer.get_messages(count, block, timeout)
            t1 = time.time()
            self._emit_recv_success_metric(t1 - t0)
        except Exception:
            self._emit_recv_fail_metric()
            logging.exception('fetch msgs from kafka failed.')
        return msgs

    def commit(self):
        '''
        By default, the kafka client commits the offset automatically.
        Through this method, user can also commit the offset manually.
        But auto-commit and manual-commit are not in conflict.
        '''
        kafka_client = self._check_kafka_client(read_only=True)
        if not kafka_client:
            return False
        if not self.consumer:
            if not self.consumer_group:
                raise ValueError("kafka proxy must set consumer group when fetch")
            try:
                self.create_consumer(kafka_client)
            except Exception:
                logging.exception("create simple consumer for kafka failed!")
                return False
        return self.consumer.commit()

    def set_consumer_offset(self, offset, whence, force=False):
        if self.consumer:
            self.consumer.seek(offset, whence)
            self.consumer.commit(force=force)
        else:
            if not self.consumer_group:
                logging.exception("kafka proxy must set consumer group when set_consumer_offset")
                return
            try:
                kafka_client = self._check_kafka_client(read_only=True)
                if kafka_client:
                    self.create_consumer(kafka_client)
                    self.consumer.seek(offset, whence)
                    self.consumer.commit(force=force)
            except Exception:
                logging.exception("kafka comsumer set offset failed.")

