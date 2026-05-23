class Receiver:
    def __init__(self):

        self.last_update_time = 0

    def receive(self, packets, current_time):

        ack_count = len(packets)

        # AoI update
        if ack_count > 0:
            newest_packet = packets[-1]

            self.last_update_time = newest_packet.created_time

        aoi = current_time - self.last_update_time

        return {"ack_count": ack_count, "aoi": aoi}
