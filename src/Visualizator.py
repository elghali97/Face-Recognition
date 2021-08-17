import torchvision

from PIL import Image, ImageDraw

from src.MagnetCluster import MagnetCluster
from src.Matcher import Matcher
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Find face on picture.')
    parser.add_argument('-f','--filename', metavar='FILENAME', type=str, default="assemblee.jpg",
                        help='File name')
    parser.add_argument('-p','--path', metavar='PATH', type=str, default="samples/",
                        help='Path of the file')
    parser.add_argument('-mh', '--maxheight', metavar='MAXHEIGHT', type=int, default=1000,
                        help='Picture is resized to a max height (0 for unlimited height)')
    parser.add_argument('-t', '--threshold', metavar='THRESHOLD', type=float, default=0.99,
                        help='Threshold for face detection')
    parser.add_argument('-o', '--offset', metavar='OFFSET', type=int, default=10,
                        help='Offset of the window')
    parser.add_argument('-md', '--min-dist', metavar='MIN-DIST', type=int, default=20,
                        help='Minimum distance for clustering window')
    parser.add_argument('-mv', '--min-votes', metavar='MIN-VOTES', type=int, default=1,
                        help='Number minimum of vote for face detection')
    parser.add_argument('-m', '--model', metavar='MODEL', type=str, default="model.pt",
                        help='Name of the model (in "models" folder)')

    args = parser.parse_args()

    image = Image.open("../"+ args.path + args.filename)

    if args.maxheight != 0 and image.size[1] > args.maxheight:
        preprocessedWidth = int(args.maxheight * image.size[0] / image.size[1])
        image = torchvision.transforms.Resize((args.maxheight, preprocessedWidth))(image)

    matcher = Matcher(image, sampleSize=(36, 36), offset=(args.offset, args.offset), threshold=args.threshold, model=args.model)

    matches = matcher.matches

    bestMatches = MagnetCluster.extract(matches, args.min_dist, args.min_votes)

    print(str(len(bestMatches)) + " best matches found")

    # Return a color between red 0%, yellow 50% and green 100%
    def compute_color(probability):
        if probability < 0.5:
            q = probability / 0.50
            r = round(0xDB + q * (0xFB - 0xDB))
            g = round(0x28 + q * (0xBD - 0x28))
            b = round(0x28 + q * (0x08 - 0x28))
        else:
            q = (probability - 0.50) / 0.75
            r = round(0xFB + q * (0x21 - 0xFB))
            g = round(0xBD + q * (0xBA - 0xBD))
            b = round(0x08 + q * (0x45 - 0x08))
        return r, g, b, 192


    # Draw best matches

    image = image.convert('RGBA')
    layer = Image.new('RGBA', image.size, (255, 255, 255, 0))

    draw = ImageDraw.Draw(layer)

    for bestMatch in bestMatches:
        normalizedProbability = bestMatch.probability
        color = compute_color(normalizedProbability)
        # color = (0,0,0)
        width = 2

        draw.line((bestMatch.left, bestMatch.top, bestMatch.right(), bestMatch.top), fill=color, width=width)  # top
        draw.line((bestMatch.left, bestMatch.bottom(), bestMatch.right(), bestMatch.bottom()), fill=color,
                  width=width)  # bottom
        draw.line((bestMatch.left, bestMatch.top, bestMatch.left, bestMatch.bottom()), fill=color, width=width)  # left
        draw.line((bestMatch.right(), bestMatch.top, bestMatch.right(), bestMatch.bottom()), fill=color,
                  width=width)  # right
        draw.line((bestMatch.left, bestMatch.top, bestMatch.right() + 1, bestMatch.top), fill=color, width=width)  # text bg

        x = round((bestMatch.left + bestMatch.right()) / 2)
        y = round((bestMatch.top + bestMatch.bottom()) / 2)

        draw.line((x, y, x+1, y+1), fill=color, width=width)  # center point

        draw.line((bestMatch.left, bestMatch.top + 5, bestMatch.right() + 1, bestMatch.top + 5), fill=color,
                  width=14)  # top

        draw.text((bestMatch.left + 2, bestMatch.top), str(bestMatch.nbVotes) + " - " + str(normalizedProbability)[0:7], (255, 255, 255, 192))

    out = Image.alpha_composite(image, layer)

    out.show()

    out = out.convert('RGB')
    out.save("../results/result_" + args.filename)