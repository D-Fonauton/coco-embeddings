from abc import ABC, abstractmethod


class Calculation(ABC):
    def __init__(self):
        super().__init__()


    @abstractmethod
    def calculate():
        pass


    def save():
        pass