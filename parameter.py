from typing import Any


class ModelParameters:
    def __init__(self):
        
        self.Step = 0  # Distance between days
        self.PastLength = 0  # Number of days in the past used to predict
        self.FutureLength = 0  # Number of predicted days in the future
        self.TrainingFeature = [] # Training feature names
        self.TargetFeature = [] # Target feature names
        self.InputShape = [None, None] # Model input
        self.OutputShape = [None] # Model output
        self.DistinFeature = None  # Prepare data for each province

    def __setattr__(self, __name: str, __value: Any) -> None:
        # Past parameter
        if __name == "PastLength" and type(__value) == int and __value >= 0:
            super().__setattr__(__name, __value)
            if "InputShape" not in self.__dict__:
                super().__setattr__("InputShape", [None, None])
            self.InputShape[0] = __value

        # Step parameter
        if __name == "Step" and type(__value) == int and __value >= 0:
            super().__setattr__(__name, __value)

        # FutureLength parameter
        if __name == "FutureLength" and type(__value) == int and __value >= 0:
            super().__setattr__(__name, __value)
            if "OutputShape" not in self.__dict__:
                super().__setattr__("OutputShape", [None])
            if "TargetFeature" not in self.__dict__:
                super().__setattr__("TargetFeature", [])
            self.OutputShape[0] = len(self.TargetFeature) * __value

        # TrainingFeature parameter
        if __name == "TrainingFeature" and type(__value) == list and len(__value) > 0:
            super().__setattr__(__name, __value)
            if "InputShape" not in self.__dict__:
                super().__setattr__("InputShape", [None, None])
            self.InputShape[1] = len(__value)

        # TargetFeature parameter
        if __name == "TargetFeature" and type(__value) == list and len(__value) > 0 and len(__value) <= len(self.TrainingFeature):
            super().__setattr__(__name, __value)
            if "OutputShape" not in self.__dict__:
                super().__setattr__("OutputShape", [None])
            if "FutureLength" not in self.__dict__:
                super().__setattr__("FutureLength", 0)
            self.OutputShape[0] = self.FutureLength * len(__value)

        # DistinFeature parameter
        if __name == "DistinFeature" and type(__value) == str:
            super().__setattr__(__name, __value)
