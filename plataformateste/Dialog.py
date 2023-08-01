class Dialog:

    def __init__(self, id):
        self.id = id
        self.turns = []

    def add_turn(self, speaker, utterance, encoding):
        self.turns.append(Turn(speaker, utterance, encoding))

class Turn:

    def __init__(self, speaker, utterance, encoding):
        self.speaker = speaker
        self.utterance = utterance
        self.encoding = encoding