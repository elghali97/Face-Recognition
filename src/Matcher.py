from threading import Thread

import torchvision
import importlib

from src.Match import Match
import torch
import time
import multiprocessing


class MatcherThread(Thread):
    """Thread chargé simplement d'afficher une lettre dans la console."""

    def __init__(self, tasks, model):

        Thread.__init__(self)

        class_name = model[model.rfind('.') + 1:]
        if class_name == 'pt':
            class_name = 'Net'
        module = importlib.import_module('src.Net.' + class_name)
        self.net_class = getattr(module, class_name)
        self.net = self.net_class()
        self.net.load_state_dict(torch.load("../models/" + model))
        self.net.eval()
        self.tasks = tasks
        self.matches = []

    def is_face(self, sampleImage):
        tensor = self.net_class.transform(sampleImage)
        result_net = self.net(tensor.reshape(1, 1, 36, 36))
        predicted = result_net.detach().numpy()
        result = predicted.item(1)

        return result

    def run(self):

        for task in self.tasks:

            sampleImage, sampleSize, originalSize, threshold, topOffset, leftOffset, resizedWidth, resizedHeight = task

            is_face_probabilty = self.is_face(sampleImage)

            if is_face_probabilty > threshold:
                topOffsetOriginalSized = round(originalSize[1] * topOffset / resizedHeight)
                leftOffsetOriginalSized = round(originalSize[0] * leftOffset / resizedWidth)

                sampleHeightOriginalSized = round(originalSize[1] * sampleSize[1] / resizedHeight)
                sampleWidthOriginalSized = round(originalSize[0] * sampleSize[0] / resizedWidth)

                self.matches.append(Match(leftOffsetOriginalSized, topOffsetOriginalSized, sampleWidthOriginalSized,
                                          sampleHeightOriginalSized, is_face_probabilty))


class Matcher:

    def __init__(self, image, sampleSize, offset, threshold, model):

        originalSize = image.size

        resizedHeight = originalSize[1] - ((originalSize[1] - sampleSize[1]) % offset[1])

        self.matches = []

        tasks = []

        print("Work is preparing...")

        while resizedHeight >= sampleSize[1]:

            resizedWidth = round(originalSize[0] * resizedHeight / originalSize[1])

            resizedImage = torchvision.transforms.Resize((resizedHeight, resizedWidth))(image)

            topOffset = 0
            while (topOffset + sampleSize[1]) < resizedHeight:

                leftOffset = 0
                while (leftOffset + sampleSize[0]) < resizedWidth:
                    sampleImage = torchvision.transforms.functional.crop(resizedImage, topOffset, leftOffset,
                                                                         sampleSize[1],
                                                                         sampleSize[0])

                    tasks.append((sampleImage, sampleSize, originalSize, threshold, topOffset, leftOffset, resizedWidth,
                                  resizedHeight))

                    leftOffset += offset[0]

                topOffset += offset[1]

            resizedHeight -= offset[1]

        # Dbg

        number_of_worker = multiprocessing.cpu_count()

        threads = []
        threadsTasks = []

        for t in range(number_of_worker):
            threadsTasks.append([])

        currentThreadIndex = 0
        while len(tasks) > 0:
            threadsTasks[currentThreadIndex].append(tasks.pop())
            currentThreadIndex = (currentThreadIndex + 1) % number_of_worker

        start = time.time()

        for t in range(number_of_worker):
            thread = MatcherThread(threadsTasks[t], model)
            threads.append(thread)
            thread.start()

        print("Work is progressing...")
        print(str(number_of_worker) + " worker(s)")

        for t in range(number_of_worker):
            threads[t].join()
            for match in threads[t].matches:
                self.matches.append(match)

        end = time.time()

        duration = end - start

        print("Duration in seconds : " + str(int(duration)))
