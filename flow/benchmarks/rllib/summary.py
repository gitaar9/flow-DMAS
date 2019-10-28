import sys

print("Summary of {}".format(sys.argv[1]))
with open(sys.argv[1], 'r') as f:
    lines = f.readlines()[:100]
    print("Averages are over {} episodes".format(len(lines)))
    percs = [0, 0, 0, 0, 0, 0]
    #0 2 4 5
    for line in lines:
        words = line.split()
        percs[0] += float(words[0])
        percs[1] += float(words[2])
        percs[2] += float(words[4])
        percs[3] += float(words[5].replace('%', ''))
        percs[4] += float(words[6])
        percs[5] += float(words[7])
    percs = map(lambda a: a / len(lines), percs)
    print("{:.2f} + {:.2f} = {:.2f} {:.2f}% {:.2f} {:.2f}".format(*percs))
