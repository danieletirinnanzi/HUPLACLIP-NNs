# import PyYAML as yaml


def load_config(path):
    with open(path, "r") as stream:
        config = yaml.safe_load(stream)
    return config


def get_model(model_name):
    match model_name:
        case "MLP":
            model = MLP()
        case _:
            model = None
    return model


def train_model(model_name):
    model = get_model(model_name)
    model.train()
    return model
