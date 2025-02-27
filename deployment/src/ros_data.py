import rclpy
from rclpy.time import Time

class ROSData:
    def __init__(self, timeout: int = 3, queue_size: int = 1, name: str = ""):
        self.timout = timeout
        self.last_time_received = rclpy.clock.Clock().now()
        self.queue_size = queue_size
        self.data = None
        self.name = name
        self.phantom = False

    def get(self):
        return self.data

    def set(self, data):
        now = rclpy.clock.Clock().now()
        time_waited = now.nanoseconds - self.last_time_received.nanoseconds

        if self.queue_size == 1:
            self.data = data
        else:
            if self.data is None or time_waited > self.timout: # reset queue if timeout
                self.data = []
            if len(self.data) == self.queue_size:
                self.data.pop(0)
            self.data.append(data)
        self.last_time_received = now

    def is_valid(self, verbose: bool = False):
        # now = rclpy.clock.Clock().now()
        # time_waited = now.nanoseconds - self.last_time_received.nanoseconds
        # valid =  time_waited < self.timout
        # if self.queue_size > 1:
        #     valid = valid and len(self.data) == self.queue_size
        # if verbose and not valid:
        #     print(f"Not receiving {self.name} data for {time_waited} seconds (timeout: {self.timout} seconds)")
        # return valid
        return True