from gpt2_medium import Trainer
from labml import experiment
import torch


print(torch.__version__)
print(torch.version.cuda)

experiment.create(name="lora_gpt2")

print(torch.cuda.is_available())

print(torch.cuda.device_count())

trainer = Trainer()

experiment.configs(trainer)

trainer.initialize()

with experiment.start():
    trainer.run()