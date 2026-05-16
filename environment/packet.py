class Packet:
    def __init__(self, seq, size=1):

        self.seq = seq
        self.size = size

        self.dropped = False
