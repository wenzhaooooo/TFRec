from data import Splitter
import configparser


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("split_data.ini")
    splitter_info = dict(config.items("split"))
    splitter = Splitter(splitter_info)
    splitter.split()
