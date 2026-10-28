"""Build the synthetic table examples/data/glm_regression.csv for examples/regression.toml.

Forty made-up subjects (sub-01 to sub-40; 20 in group OSA, 20 in group control) with logAHI, age
and sex, and band power in E1 to E256. Power rises with logAHI on E129 and its neighbours only;
elsewhere it is noise plus a small age effect. Nothing here comes from a real recording. The
table serves the regression (predictor logAHI, covariates age and sex) and the ANCOVA (group
column group, same covariates), from a configuration file or in the window's Table file mode.

    python examples/make_regression_example.py
    snpm run examples/regression.toml
"""

from pathlib import Path

import numpy as np
import pandas as pd

from hdeeg_snpm.montage import get_montage

OUT = Path(__file__).parent / "data" / "glm_regression.csv"

rng = np.random.default_rng(20261027)
m = get_montage("egi")
centre = m.labels.index("E129")
planted = [m.labels[centre]] + [m.labels[i] for i in m.neighbors[centre] if i >= 0]

n = 40
group = np.array(["OSA"] * 20 + ["control"] * 20)
logAHI = np.where(group == "OSA", rng.normal(1.4, 0.3, n), rng.normal(0.5, 0.3, n)).round(3)
age = rng.normal(50, 10, n).round(1)
sex = rng.choice(["F", "M"], n)
chans = [f"E{i}" for i in range(1, 257)]
power = rng.normal(10, 1, (n, 256)) + 0.02 * (age - 50)[:, None]
for lab in planted:
    power[:, chans.index(lab)] += 1.5 * (logAHI - logAHI.mean())

df = pd.DataFrame(power.round(4), columns=chans)
df.insert(0, "sex", sex)
df.insert(0, "age", age)
df.insert(0, "logAHI", logAHI)
df.insert(0, "group", group)
df.insert(0, "Subject", [f"sub-{i:02d}" for i in range(1, n + 1)])

if __name__ == "__main__":
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({n} subjects); effect planted on {', '.join(planted)}")
