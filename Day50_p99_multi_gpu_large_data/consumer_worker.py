from bisect import bisect_left
from dataclasses import dataclass
from statistics import mean
from typing import List, Optional, Dict
from math import ceil
import random

##Worker##
class Histogram:
    def __init__(self, upper_bucket_limits):
        self.upper_bucket_limits = upper_bucket_limits
        self.counts = [0] * len(upper_bucket_limits)
        self.total_req = 0
        self.sum = 0.0
        self.min_val = float("inf")
        self.max_val = float("-inf")
    
    def add(self, latency_value:int, count:int = 1):
        idx = bisect_left(self.upper_bucket_limits, latency_value)   
        if(idx == len(self.counts)):
            idx = idx-1
        self.counts[idx] += count
        self.total_req += count
        self.sum += (count*latency_value)
        self.min_val = min(self.min_val, latency_value)
        self.max_val = max(self.max_val, latency_value)

    def merge(self, other:"Histogram"):
        assert self.upper_bucket_limits == other.upper_bucket_limits
        for i,c in enumerate(other.counts):
            self.counts[i] += c
        self.total_req += other.total_req
        self.sum += other.sum
        self.min_val = min(self.min_val, other.min_val)
        self.max_val = max(self.max_val, other.max_val)

    def percentile(self, p:int) -> Optional[int]:
        if self.total == 0:
            return None
        cummulative = 0
        p_inst = ceil(p/100.0*self.total)
        for i, c in enumerate(self.counts):
            cummulative += c
            if(cummulative > p_inst):
                return self.upper_bucket_limits[i]
        
        return self.upper_bucket_limits[-1]

    def copy(self) -> "Histogram":
        h = Histogram(self.upper_bucket_limits)
        h.counts = self.counts
        h.sum = self.sum
        h.total_req = self.total_req
        h.min_val = self.min_val
        h.max_val = self.max_val
        return h

    def reset(self):
        for i in range(len(self.upper_bucket_limits)):
            self.counts[i] = 0
        self.total_req = 0
        self.sum = 0
        self.min_val = float("inf")
        self.max_val = float("-inf")

    def average(self) -> Optional[float]:
        if(self.total == 0):
            return None
        return self.sum/self.total

@dataclass
class HistogramPacket:
    worker_id :int
    metric_name : str
    timestamp_sec : int
    histogram : Histogram


class Worker:
    def __init__(self, worker_id:int, metric_name:str, upper_bucket_limits:List[int]):
        self.histogram = Histogram(upper_bucket_limits=upper_bucket_limits)
        self.worker_id = worker_id
        self.metric_name = metric_name
        self.upper_bucket_limits = upper_bucket_limits
        self.current_second = None
    
    def observe(self, current_time:int, latency_ms:int) -> List[HistogramPacket]:
        "Returns 0/1 packet. Packet returns when we move from one time stamp to another"
        ret_val: List[HistogramPacket] = []

        if self.current_second is None:
            self.current_second = current_time

        if current_time != self.current_second:
            ret_val.append(
                HistogramPacket(
                    worker_id=self.worker_id,
                    metric_name=self.metric_name,
                    timestamp_sec=self.current_second,
                    histogram=self.histogram.copy()
                )
            )

            self.current_second = current_time
            self.histogram = Histogram(self.upper_bucket_limits)

        self.histogram.add(latency_ms)
        return ret_val

    def flush(self):
        " Flush on request"
        if self.current_second is None or self.histogram.total_req == 0:
            return None

        ret_val = HistogramPacket(
            worker_id=self.worker_id,
            metric_name=self.metric_name,
            timestamp_sec=self.current_second,
            histogram=self.histogram.copy()
        )

        self.histogram.reset()

        return ret_val


##Co-ordinator Histogram##
class RollingHistogram:
    """
    window 300 sec, One slot per sec, Query p99
    """
    def __init__(self, window_sec:int , upper_bound_limit:List[float]):
        self.window_sec = window_sec
        self.slots = [Histogram(upper_bound_limit) for _ in range(window_sec)]
        self.timestamps = [0] * window_sec
        self.upper_bound_limit = upper_bound_limit

    def _compute_slot_idx(self, time_stamp):
        return time_stamp % self.window_sec

    def merge_packet(self, worker_packet:HistogramPacket):
        current_slot = self._compute_slot_idx(worker_packet.timestamp_sec)

        if(self.timestamps[current_slot] != worker_packet.timestamp_sec):
            self.slots[current_slot].reset()
            self.timestamps[current_slot] = worker_packet.timestamp_sec

        self.slots[current_slot].merge(worker_packet.histogram)

    def merge_window(self, now:int):
        start_time = now - self.window_sec
        ret_val = Histogram(self.upper_bound_limit)
        for cur_hist, cur_time_stamp in zip(self.slots, self.timestamps):
            if(cur_hist.total_req > 0) and (start_time < cur_time_stamp < now):
                ret_val.merge(cur_hist)

        return ret_val

    def percentile(self, p:int, now:int) -> Optional[int]:
        self.merge_window(now).percentile(p)

    def average(self, now:int):
        self.merge_window(now).average()
        
    def count(self, now:int) -> int:
        self.merge_window(now).total_req

class Aggregrator:
    def __init__(self, window_sec, upper_bound_limit):
        self.upper_bound_limit = upper_bound_limit
        self.window_sec = window_sec
        self.metrics:Dict[str, RollingHistogram] = {}

    def receive(self, incoming_packet:HistogramPacket):
        if incoming_packet.metric_name not in self.metrics.keys:
            self.metrics[incoming_packet.metric_name] = RollingHistogram(self.window_sec, self.upper_bound_limit)
        
        self.metrics[incoming_packet.metric_name].merge_packet(incoming_packet)

    def percentile(self, p, now, metric_name) -> Optional[int]:
        if metric_name not in self.metrics.keys:
            return None
        
        return self.metrics[metric_name].percentile(p, now)

    def average(self, now, metric_name):
        if metric_name not in self.metrics.keys:
            return None

        return self.metrics[metric_name].average(now)

    def count(self, now, metric_name):
        if metric_name not in self.metrics.keys:
            return None

        return self.metrics[metric_name].count(now)


#Bucket construction#
def bucket_const(min_val:int, max_val:int, factor:float) -> List[int]:
    ret_val = []
    val = min_val
    
    while val < max_val:
        ret_val.append(min_val)
        val *= factor

    ret_val.append(max_val)
    return ret_val

def simulate():
    upper_bucket_limit = bucket_const(1, 60000, 1.1)
    num_workers = 8
    window_seconds = 300
    metric_name = "e2e_latency_ms"

    workers_list = [Worker(i, "tpot", upper_bucket_limit) for i in range(num_workers)]
    coordinater = Aggregrator(window_seconds, upper_bucket_limit)

    #Simulate 10 mins of traffic

    start_sec = 1000
    end_sec = 1000+600

    for second in range(start_sec, end_sec):
        for worker in workers_list:
            rps = random.randint(500,1500)
            for _ in range(rps):
                latency_ms = random.lognormvariate(mu=4.5, sigma=0.5)
                if random.random() < 0.01:
                    latency_ms *= random.uniform(5,20)

                packets = worker.observe(second, latency_ms)

                for packet in packets:
                    coordinater.recieve(packet)

        if second%60 == 0:
            p50 = coordinater.percentile(50, second, metric_name)
            p95 = coordinater.percentile(95, second, metric_name)
            p99 = coordinater.percentile(99, second, metric_name)
            avg = coordinater.average(second, metric_name)
            count = coordinater.count(second, metric_name)

            print(
                f"sec : {second}",
                f"count : {count}",
                f"avg  : {avg}",
                f"p50 : {p50}",
                f"p95 : {p95}",
                f"p99 : {p99}",
            )

    for worker in workers_list:
        packet = worker.flush()
        if packet is not None:
            coordinater.receive(packet)

    now = end_sec - 1
    p50 = coordinater.percentile(50, now, metric_name)
    p95 = coordinater.percentile(95, now, metric_name)
    p99 = coordinater.percentile(99, now, metric_name)
    avg = coordinater.average(now, metric_name)
    count = coordinater.count(now, metric_name)

    print(
        f"F sec : {second}",
        f"F count : {count}",
        f"F avg  : {avg}",
        f"F p50 : {p50}",
        f"F p95 : {p95}",
        f"F p99 : {p99}",
    )
    
if __name__ == "__main__":
    simulate()

    
