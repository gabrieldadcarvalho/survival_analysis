import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
import json


def roc(fpr, sensibility):
    """
    Plota a curva ROC suavizada com interpolação e destaca o melhor ponto baseado em distância ao ponto (0, 1).
    """

    fpr = np.array(fpr)
    sensibility = np.array(sensibility)

    distance = np.sqrt(fpr**2 + (1 - sensibility) ** 2)
    bestIndex = np.argmin(distance)
    bestFpr = fpr[bestIndex]
    bestSensibility = sensibility[bestIndex]

    # Plot
    plt.plot([0, 1], [0, 1], "k--", label="Stochastic Model")
    plt.scatter(
        fpr, sensibility, color="white", edgecolors="black", label="Original Points"
    )
    plt.scatter(bestFpr, bestSensibility, color="red", label="Best Point")
    plt.text(
        bestFpr - 0.03,
        bestSensibility + 0.02,
        f"({bestFpr:.2f}, {bestSensibility:.2f})",
        fontsize=6,
        color="black",
    )
    plt.xlabel("False Positive Rate")
    plt.ylabel("Sensibility")
    plt.title("ROC Curve")
    plt.grid(True)
    plt.legend()
    plt.show()


with open(
    "/home/gabrieldadcarvalho/github/survival_analysis/roc/creatine_transplant_rejection.json"
) as f:
    data = json.load(f)

roc = roc(data["FPR"]["values"], data["sensitivity"]["values"])
