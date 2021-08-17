from math import hypot

class MagnetCluster:

    def extract(matches, epsilon, minVotes):

        matchesCluster = {}

        nextClusterIndex = 0

        for match in matches:
            matchesCluster[id(match)] = nextClusterIndex
            nextClusterIndex += 1

        merge_occured = True
        while merge_occured:
            merge_occured = False

            transformationMap = {}

            for match in matches:
                for targetMatch in matches:

                    matchCluster = matchesCluster[id(match)]
                    targetMatchCluster = matchesCluster[id(targetMatch)]

                    if id(match) != id(targetMatch) and matchCluster != targetMatchCluster:

                        matchCenter = match.center()
                        targetMatchCenter = targetMatch.center()

                        dist = hypot(targetMatchCenter[0] - matchCenter[0], targetMatchCenter[1] - matchCenter[1])

                        if dist <= epsilon:
                            if matchCluster < targetMatchCluster:
                                transformationMap[targetMatchCluster] = matchCluster
                            else:
                                transformationMap[matchCluster] = targetMatchCluster

                            merge_occured = True


            # Transitivity

            # Merge

            for match in matches:
                matchCluster = matchesCluster[id(match)]
                if matchCluster in transformationMap:
                    matchesCluster[id(match)] = transformationMap[matchCluster]

        # Clusters
        clusters = {}

        for match in matches:
            clusterIndex = matchesCluster[id(match)]

            if not(clusterIndex in clusters):
                clusters[clusterIndex] = []

            clusters[clusterIndex].append(match)

        # Best matches

        bestMatches = []

        for clusterIndex in clusters:

            clusterMatches = clusters[clusterIndex]

            if len(clusterMatches) > 0:

                bestMatch = clusterMatches[0]

                for match in clusterMatches:
                    if match.probability > bestMatch.probability:
                        bestMatch = match

                bestMatch.nbVotes = len(clusterMatches)
                if(bestMatch.nbVotes >= minVotes):
                    bestMatches.append(bestMatch)

        return bestMatches
