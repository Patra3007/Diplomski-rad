import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# === OVDJE UNESI SVOJE VRIJEDNOSTI ===
TP = 5700969
FP = 314410
FN = 401401
TN = 234735220

# Kreiraj matricu zabune (redoslijed: [[TN, FP], [FN, TP]])
cm = np.array([[TP, FP],
               [FN, TN]])

# Normalizacija po ukupnom broju uzoraka
cm_norm = cm.astype('float') / cm.sum() * 100  # u %

# Crtanje
plt.figure(figsize=(4,4), dpi=150)
sns.heatmap(
    cm_norm,
    annot=cm,          # prikaži originalne brojeve u poljima
    fmt="d",           # format za anotacije
    cmap="Blues",
    xticklabels=["Pozitivno", "Negativno"],
    yticklabels=["Pozitivno", "Negativno"],
    cbar=True,
    square=True
)

plt.xlabel("Predikcija")
plt.ylabel("Stvarna slika")
plt.title("Modalna maska, podjela 5")

plt.tight_layout()
plt.savefig("modal5.png")
plt.close()
