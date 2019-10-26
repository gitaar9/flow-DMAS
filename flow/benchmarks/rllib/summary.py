import sys

print("Summary of {}".format(sys.argv[1]))
with open(sys.argv[1], 'r') as f:
    lines = f.readlines()
    percs = [0, 0, 0, 0]
    #0 2 4 5
    for line in lines:
        words = line.split()
        percs[0] += float(words[0])
        percs[1] += float(words[2])
        percs[2] += float(words[4])
        percs[3] += float(words[-1].replace('%', ''))
    percs = map(lambda a: a / len(lines), percs)
    print("{:.2f} + {:.2f} = {:.2f} {:.2f}%".format(*percs))
