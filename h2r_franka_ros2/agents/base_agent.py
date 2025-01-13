class base_agent:
    def __init__(self, device):
        self.device = device
    
    def predict_action(self, image: Image.Image, instruction:str):
        NotImplementedError()