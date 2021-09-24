import matplotlib.pyplot as plt
import matplotlib as mlp

import plotly.express as px

from model_dataset import ModelGenerator

def earth_model(model, keys=[]):

    A=1



if __name__ == "__main__":
    import model_dataset as md
    [nz, nx] = [100, 100]
    Model = ModelGenerator(nx, nz, 1, 1)
    model = Model.circle({"vp":2500}, {"vp": 3000}, [50, 50], 10)

    earth_model(model)
