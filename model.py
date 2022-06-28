from math import sqrt
from pathlib import Path

from commons import PUB, eprint


class Model:
    def __init__(self):
        self.N = None  # number of buses
        self.Q = None  # capacity of buses (number of people each bus can carry)
        self.customers = []
        self.alpha = 1  # route cost parameter
        self.beta = 1  # customer satisfaction parameter
        self.omega = 1  # distance/time to cost parameter

    def parse(self, filename: str):
        def is_key_value(line: str, key: str):
            return line.startswith(f"{key}=")

        def get_value_from_key_value(line: str):
            return line.split("=")[1]

        try:
            with Path(filename).open() as f:
                SECTION_HEADER = 0
                SECTION_CUSTOMERS = 1
                SECTION_ALPHA = 2
                SECTION_BETA = 3
                SECTION_OMEGA = 4

                section = SECTION_HEADER

                for line in f:
                    line = line.strip()
                    if not line:
                        # skip empty lines
                        continue
                    if line.startswith("#"):
                        # skip comment
                        continue

                    # new section
                    if line == "CUSTOMERS":
                        section = SECTION_CUSTOMERS
                        continue

                    if line == "ALPHA":
                        section = SECTION_ALPHA
                        continue

                    if line == "BETA":
                        section = SECTION_ALPHA
                        continue

                    if line == "OMEGA":
                        section = SECTION_ALPHA
                        continue

                    # header
                    if section == SECTION_HEADER:
                        if is_key_value(line, "N"):
                            self.N = int(get_value_from_key_value(line))
                            continue
                        if is_key_value(line, "Q"):
                            self.Q = int(get_value_from_key_value(line))
                            continue

                    # customers
                    if section == SECTION_CUSTOMERS:
                        x, y, a, = line.split(" ")
                        self.customers.append((int(x), int(y), int(a)))
                        continue

                    # alpha
                    if section == SECTION_ALPHA:
                        self.alpha = int(get_value_from_key_value(line))
                        continue

                    # beta
                    if section == SECTION_BETA:
                        self.beta = int(get_value_from_key_value(line))
                        continue

                    # omega
                    if section == SECTION_OMEGA:
                        self.omega = int(get_value_from_key_value(line))
                        continue
            return True
        except Exception as e:
            eprint(f"ERROR: failed to parse file: {e}")
            return False

    def distance(self, v1, v2):
        xi, yi = (self.customers[v1 - 1][0], self.customers[v1 - 1][1]) if v1 != PUB else (0, 0)
        xj, yj = (self.customers[v2 - 1][0], self.customers[v2 - 1][1]) if v2 != PUB else (0, 0)
        return sqrt((xi - xj) ** 2 + (yi - yj) ** 2)

    def time(self, v1, v2):
        return self.distance(v1, v2)