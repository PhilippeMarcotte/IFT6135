import matplotlib.pyplot as plt
import numpy as np
from Q1_training import training_loop
import Losses


x = -1

JSD = []
WSD = []
x = np.linspace(-1,1,21)

for i in range(21):
    print("Progress:  {:.2f} %".format((i+1)*100/21))
    lossJSD, _ = training_loop(Losses.JSDLoss(), x[i])
    LossWST, _ = training_loop(Losses.WassersteinLoss(10), x[i])
    JSD.append(np.abs(lossJSD.item()))
    WSD.append(np.abs(LossWST.item()))


plt.plot(x,JSD)
plt.plot(x,WSD)
plt.xlabel("$\phi$")
plt.ylabel("Distance metric")
plt.legend(["JSD","Wasserstein"])
plt.savefig("Distance_vs_phi.png")
plt.show()
