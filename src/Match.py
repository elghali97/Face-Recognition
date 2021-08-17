class Match:
    def __init__(self, left, top, width, height, probability):
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        self.probability = probability
        self.nbVotes = 1

    def right(self):
        return self.left + self.width

    def bottom(self):
        return self.top + self.height

    def center(self):
        return [self.left + self.width / 2, self.top + self.height / 2]

    def __eq__(self, other):
        return self.left == other.left and \
               self.top == other.top and \
               self.width == other.width and \
               self.height == other.height

    def __hash__(self):
        return hash((self.left, self.top, self.width, self.height))
