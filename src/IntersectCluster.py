
class SquareFilter:

    def extract(matches):

        bestMatches = []

        matchIndex = 0

        for match in matches:

            bestMatchIndex = 0

            nonEmptyIntersectedBestMatchIndexes = []

            for bestMatch in bestMatches:

                # If the intersection is not empty
                if bestMatch.right() > match.left and \
                        match.right() > bestMatch.left and \
                        bestMatch.bottom() > match.top and \
                        match.bottom() > bestMatch.top:
                    nonEmptyIntersectedBestMatchIndexes.append(bestMatchIndex)

                bestMatchIndex += 1

            if len(nonEmptyIntersectedBestMatchIndexes) == 0:
                bestMatches.append(match)
            else:

                betterProbability = False

                for bestMatchIndex in nonEmptyIntersectedBestMatchIndexes:
                    if match.probability > bestMatches[bestMatchIndex].probability:
                        betterProbability = True
                        break

                if betterProbability:
                    for bestMatchIndex in nonEmptyIntersectedBestMatchIndexes:
                        bestMatches[bestMatchIndex] = match

            matchIndex += 1

        bestMatches = list(set(bestMatches))  # Remove duplicates

        return bestMatches