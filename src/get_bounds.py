import pandas as pd
import numpy as np

#### FEHLER ERFOLGRICH REPRODUZIERT ####

def get_bounds_for_hri_tolerances():

    df = pd.read_csv("../data/final/hri_filtering.csv").drop(columns=["Unnamed: 0"], errors='ignore')

    df_mean = df.mean()
    df_std = df.std()

    #### assume following way:
    # untere Grenze: mean - 1 * std.
    # obere Grenze: mean + 1 * std.

    tol_1_unten = df_mean["hri_O_0.21"] - (1 * df_std["hri_O_0.21"])
    tol_1_oben = df_mean["hri_O_0.21"] + (1 * df_std["hri_O_0.21"])
    tol_2_unten = df_mean["hri_O_0.31"] - (1 * df_std["hri_O_0.31"])
    tol_2_oben = df_mean["hri_O_0.31"] + (1 * df_std["hri_O_0.31"])
    tol_3_unten = df_mean["hri_O_0.42"] - (1 * df_std["hri_O_0.42"])
    tol_3_oben = df_mean["hri_O_0.42"] + (1  * df_std["hri_O_0.42"])

    print("stds: ", df_std["hri_O_0.21"], df_std["hri_O_0.31"], df_std["hri_O_0.42"])

    return tol_1_unten, tol_1_oben, tol_2_unten, tol_2_oben,tol_3_unten, tol_3_oben


if __name__ == "__main__":
    tol_1_unten, tol_1_oben, tol_2_unten, tol_2_oben,tol_3_unten, tol_3_oben = get_bounds_for_hri_tolerances()
    print(tol_1_unten, tol_1_oben, tol_2_unten, tol_2_oben,tol_3_unten, tol_3_oben)
