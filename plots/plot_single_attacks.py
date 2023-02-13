import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib


labels = ['done2', 'done1', 'empty']
# done2, done1, empty
fgsm = [0.625,1,0.0625]
fgsm_round = [1,1,0]
fgsm_floor = [0,0,1]
deepfool = [0.125,1,0.875]
deepfool_round = [0.75,1,0.25]
deepfool_floor = [0,0,1]

barWidth = 0.15

r1 = np.arange(len(fgsm))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]
r6 = [x + barWidth for x in r5]

plt.bar(r1, fgsm, width=barWidth, label='fgsm', zorder=2)
plt.bar(r2, fgsm_round, width=barWidth,label='fgsm_round', zorder=2)
plt.bar(r3, fgsm_floor, width=barWidth, label='fgsm_floor', zorder=2)
plt.bar(r4, deepfool, width=barWidth, label='deepfool', zorder=2)
plt.bar(r5, deepfool_round, width=barWidth, label='deepfool_round', zorder=2)
plt.bar(r6, deepfool_floor, width=barWidth, label='deepfool_floor', zorder=2)

plt.xlabel('bar')
plt.xticks([r + (barWidth*2) for r in range(len(fgsm))], labels)
plt.ylabel('value')
plt.legend(bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0.)
plt.grid(zorder=1)
plt.tight_layout()
plt.show()




# Show graphic
tikzplotlib.save("plots/single_attacks.tex")
