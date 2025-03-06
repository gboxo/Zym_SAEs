import torch
from torch.utils.data import DataLoader, TensorDataset
from transformer_lens.hook_points import HookedRootModule
from datasets import Dataset, load_from_disk
import tqdm

class ActivationsStore:
    def __init__(
        self,
        model: HookedRootModule,
        cfg: dict,
        batch_size=1024
    ):
        self.model = model
        
        self.dataset = iter(load_from_disk(cfg["dataset_path"]))
        self.hook_point = cfg["hook_point"]
        self.context_size = min(cfg["seq_len"], model.cfg.n_ctx)
        self.model_batch_size = cfg["model_batch_size"]
        self.device = cfg["device"]
        self.num_batches_in_buffer = cfg["num_batches_in_buffer"]
        self.tokens_column = self._get_tokens_column()
        self.cfg = cfg
        self.tokenizer = model.tokenizer
        
        # Add tracking of position in dataset
        self.current_batch_idx = 0
        self.current_epoch = 0
        self.dataset_position = 0

    def _get_tokens_column(self):
        sample = next(self.dataset)
        if sum(sample["input_ids"]) < 1000:
            _

        if "tokens" in sample:
            return "tokens"
        elif "input_ids" in sample:
            return "input_ids"
        elif "text" in sample:
            return "text"
        else:
            raise ValueError("Dataset must have a 'tokens', 'input_ids', or 'text' column.")

    def get_batch_tokens(self):
        all_tokens = []
        while len(all_tokens) < self.model_batch_size * self.context_size:
            batch = next(self.dataset)
            if self.tokens_column == "text":
                tokens = self.model.to_tokens(batch["text"], truncate=True, move_to_device=True, prepend_bos=True).squeeze(0)
            else:
                tokens = batch[self.tokens_column]
            if sum(tokens) < 1000:
                continue

            all_tokens.extend(tokens)
        token_tensor = torch.tensor(all_tokens, dtype=torch.long, device=self.device)[:self.model_batch_size * self.context_size]
        return token_tensor.view(self.model_batch_size, self.context_size)

    def get_activations(self, batch_tokens: torch.Tensor):
        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                batch_tokens,
                names_filter=[self.hook_point],
                stop_at_layer=self.cfg["layer"] +1,
            )
        return cache[self.hook_point]

    def _fill_buffer(self):
        all_activations = []
        for _ in range(self.num_batches_in_buffer):
            batch_tokens = self.get_batch_tokens()
            activations = self.get_activations(batch_tokens).reshape(-1, self.cfg["act_size"])
            all_activations.append(activations)
        return torch.cat(all_activations, dim=0)

    def _get_dataloader(self):
        return DataLoader(TensorDataset(self.activation_buffer), batch_size=self.cfg["batch_size"], shuffle=True)

    def next_batch(self):
        try:
            return next(self.dataloader_iter)[0]
        except (StopIteration, AttributeError):
            self.activation_buffer = self._fill_buffer()
            self.dataloader = self._get_dataloader()
            self.dataloader_iter = iter(self.dataloader)
            return next(self.dataloader_iter)[0]

    def update_position(self, batch_idx, epoch):
        """Update the current position in the dataset"""
        self.current_batch_idx = batch_idx
        self.current_epoch = epoch
        self.dataset_position = epoch * 1024 + batch_idx
        
    def get_position(self):
        """Get the current position in the dataset"""
        return self.current_batch_idx, self.current_epoch, self.dataset_position


