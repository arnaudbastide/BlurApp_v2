import argparse, dataclasses
from config import Config

def parse() -> Config:
    parser = argparse.ArgumentParser()
    for f in dataclasses.fields(Config):
        parser.add_argument(f"--{f.name.replace('_', '-')}", type=f.type, default=f.default)
    return Config(**vars(parser.parse_args()))
