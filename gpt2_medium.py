import torch
from labml.utils.download import download_file
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoModelForCausalLM

from labml import lab, monit, tracker
from labml.configs import BaseConfigs, option
from labml_nn.lora.gpt2 import GPTModel  # Custom implementation for LoRA


class Trainer(BaseConfigs):
    """
    ## Trainer configurations and the training loop
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # GPT-2 configurations
    layer_norm_epsilon: float = 1e-5
    d_model: int = 1024  # Hidden size
    n_layers: int = 24  # Number of layers for GPT-2 Medium
    n_heads: int = 16  # Number of attention heads
    n_positions: int = 1024  # Context length
    vocab_size: int = 50257  # Vocabulary size for GPT-2

    # Training configs
    epochs: int = 10
    batch_size: int = 7  # Smaller batch size due to reduced GPU memory
    learning_rate: float = 1e-4
    context_len: int = 256  # Modify context length as needed

    # LoRA rank
    lora_r: int = 32

    # Dataset
    text: TensorDataset = "tiny_shakespeare"

    # Load the pre-trained GPT-2 Medium model
    model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
    tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
    # Optimizer
    optimizer: torch.optim.Adam
    # Cross entropy loss
    loss_func = torch.nn.CrossEntropyLoss()
    # Dataloader
    data_loader: DataLoader

    def _load_pretrained_weights(self):
        """
        ### Load pre-trained [GPT-2 from huggingface](https://huggingface.co/openai-community/gpt2)
        """

        # Load the huggingface model and get the parameters
        hf_model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
        state_dict = hf_model.state_dict()

        # Transformer embedding and prediction layer parameter mapping (`hf: ours`)
        mapping = {
            'transformer.wte.weight': 'token_embedding.weight',
            'transformer.wpe.weight': 'position_embedding.weight',
            'transformer.ln_f.weight': 'final_norm.weight',
            'transformer.ln_f.bias': 'final_norm.bias',
            'lm_head.weight': 'lm_head.weight'
        }

        # Mapping (`hf: ours`) of decoder layers for GPT-2 Medium
        for i in range(24):  # Cambiado a 24 para GPT-2 Medium
            mapping[f'transformer.h.{i}.ln_1.weight'] = f'blocks.{i}.attn_norm.weight'
            mapping[f'transformer.h.{i}.ln_1.bias'] = f'blocks.{i}.attn_norm.bias'
            mapping[f'transformer.h.{i}.attn.c_attn.weight'] = f'blocks.{i}.attn.qkv_projection.weight'
            mapping[f'transformer.h.{i}.attn.c_attn.bias'] = f'blocks.{i}.attn.qkv_projection.bias'
            mapping[f'transformer.h.{i}.attn.c_proj.weight'] = f'blocks.{i}.attn.output_projection.weight'
            mapping[f'transformer.h.{i}.attn.c_proj.bias'] = f'blocks.{i}.attn.output_projection.bias'
            mapping[f'transformer.h.{i}.ln_2.weight'] = f'blocks.{i}.ffn_norm.weight'
            mapping[f'transformer.h.{i}.ln_2.bias'] = f'blocks.{i}.ffn_norm.bias'
            mapping[f'transformer.h.{i}.mlp.c_fc.weight'] = f'blocks.{i}.ffn.linear_in.weight'
            mapping[f'transformer.h.{i}.mlp.c_fc.bias'] = f'blocks.{i}.ffn.linear_in.bias'
            mapping[f'transformer.h.{i}.mlp.c_proj.weight'] = f'blocks.{i}.ffn.linear_out.weight'
            mapping[f'transformer.h.{i}.mlp.c_proj.bias'] = f'blocks.{i}.ffn.linear_out.bias'

        # Move the parameters based on mapping
        new_state_dict = {}
        for old_key, new_key in mapping.items():
            if old_key in state_dict:
                new_state_dict[new_key] = state_dict[old_key]

        # GPT-2 hugging face uses 1D Convolution layers. We need to transpose those weights since we use linear layers
        convo_layers = ([f'blocks.{i}.ffn.linear_in.weight' for i in range(24)] +
                        [f'blocks.{i}.ffn.linear_out.weight' for i in range(24)] +
                        [f'blocks.{i}.attn.qkv_projection.weight' for i in range(24)] +
                        [f'blocks.{i}.attn.output_projection.weight' for i in range(24)])

        for layer in convo_layers:
            new_state_dict[layer] = torch.transpose(new_state_dict[layer], 0, 1)

        # Load our model. We use `strict = False` because the state does not have LoRA weights
        missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)

        # Make sure that only LoRA weights are not loaded
        assert all('lora' in key for key in missing_keys)
        assert not unexpected_keys

    def initialize(self):
        """
        ### Initialize the model, optimizer, and dataloader
        """
        # Initialize the model (adjusted for LoRA if applicable)
        self.model = GPTModel(
            layer_norm_epsilon=self.layer_norm_epsilon,
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            n_positions=self.n_positions,
            vocab_size=self.vocab_size,
            r=self.lora_r,
        )
        self.model.to(self.device)
        # Load pre-trained weights
        self._load_pretrained_weights()

        # Initialize the optimizer
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)

        # Initialize the dataloader
        self.data_loader = DataLoader(self.text, batch_size=self.batch_size, shuffle=True)

    def run(self):
        """
        ### Training loop
        """

        for _ in monit.loop(self.epochs):
            # `inputs` has shape `[batch_size, seq_len]`
            for (inputs,) in monit.iterate('Train', self.data_loader):
                # Move `inputs` to device
                inputs = inputs.to(self.device)
                # Call the model, excluding the last token
                logits = self.model(inputs[:, :-1])
                # Compute loss
                loss = self.loss_func(logits.reshape(-1, logits.shape[-1]), inputs[:, 1:].reshape(-1))

                # Zero gradients, backpropagate, and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Log the loss
                tracker.save({'loss': loss})
                tracker.add_global_step()
            #
            tracker.new_line()


@option(Trainer.text)
def tiny_shakespeare(c: Trainer):
    """
    ### Tiny Shakespeare dataset
    """
    path = lab.get_data_path() / 'tiny_shakespeare.txt'
    if not path.exists():
        download_file("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", path)
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    tokens = c.tokenizer.encode(text)
    num_batches = len(tokens) // (c.batch_size * c.context_len)
    tokens = tokens[:num_batches * c.batch_size * c.context_len]
    input_ids = torch.tensor(tokens).view(-1, c.context_len)
    return TensorDataset(input_ids)
