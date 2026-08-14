from __future__ import annotations
from typing import Optional, Union, Any
import logging

try:
    from datasets import load_dataset, IterableDataset, Dataset
except ImportError:
    IterableDataset = type("IterableDataset", (), {})
    Dataset = type("Dataset", (), {})
    load_dataset = None

from .sources import DatasetSource, get_dataset_source

logger = logging.getLogger(__name__)

class VasudhaDatasetLoader:
    """Streams datasets without full downloads, useful for constrained environments like Colab."""
    def __init__(
        self,
        source: Union[DatasetSource, str],
        buffer_size: int = 10000,
        seed: int = 42,
        num_workers: int = 0,
    ):
        if load_dataset is None:
            raise ImportError("The 'datasets' package is required. Install it with pip install datasets.")
        if isinstance(source, str):
            self.source = get_dataset_source(source)
        else:
            self.source = source
        self.buffer_size = buffer_size
        self.seed = seed
        self.num_workers = num_workers

    def load(self) -> Any:
        """Load dataset in streaming mode."""
        mode = "streaming" if self.source.streaming else "local/materialized"
        logger.info(f"Loading dataset {self.source.name} in {mode} mode.")
        load_args = [self.source.name]
        if getattr(self.source, "config", None):
            load_args.append(self.source.config)
        dataset = load_dataset(
            *load_args,
            split=self.source.split,
            streaming=self.source.streaming,
            trust_remote_code=self.source.trust_remote_code
        )
        # buffer_size <= 0 disables shuffling. On a streaming dataset, shuffle also
        # shuffles shard order, so it reads from many shards at once — expensive on
        # a 120-shard source and pointless for a smoke test.
        if self.buffer_size > 0 and hasattr(dataset, "shuffle"):
            dataset = dataset.shuffle(seed=self.seed, buffer_size=self.buffer_size)
        return dataset

    def load_normalized(self) -> Any:
        """
        Load the dataset and reduce it to a single 'text' column.

        Sources in a mixture have incompatible schemas — OpenThoughts3 has
        'conversations', NuminaMath has 'messages', GSM8K has question/answer —
        and interleave_datasets requires matching features across all of them.
        Collapsing each source to one string column makes them interleavable and
        gives TRL the field it trains on.
        """
        from .chat_format import ChatFormatter

        # _format_qwen3 is pure string assembly, so no tokenizer is needed here.
        formatter = ChatFormatter(tokenizer=None, format="qwen3")
        source = self.source
        column = source.text_column
        fmt = source.format

        def to_text(example: dict) -> dict:
            value = example.get(column)
            try:
                if fmt == "sharegpt" and value:
                    return {"text": formatter.format_sharegpt(value)}
                if fmt == "messages" and value:
                    return {"text": formatter.format_messages(value)}
                if fmt == "qa":
                    question = example.get(column, "")
                    answer = example.get(getattr(source, "answer_column", "answer"), "")
                    if not (question and answer):
                        return {"text": ""}
                    return {"text": formatter.format_conversation([
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ])}
            except Exception as exc:
                logger.warning(f"Skipping malformed example in {source.name}: {exc}")
                return {"text": ""}
            return {"text": value if isinstance(value, str) else ""}

        dataset = self.load()

        # For streaming datasets, column_names may be None until features resolve.
        # Rather than peeking (which triggers a blocking network fetch), just map
        # and let the output schema be determined by the map function.  We pass
        # remove_columns only when the dataset already knows its columns.
        columns = getattr(dataset, "column_names", None)
        if columns:
            dataset = dataset.map(to_text, remove_columns=columns)
        else:
            # Streaming dataset without resolved features — map without removal,
            # then select only the 'text' column.
            dataset = dataset.map(to_text)
            dataset = dataset.select_columns(["text"])

        if hasattr(dataset, "filter"):
            dataset = dataset.filter(lambda ex: bool(ex.get("text")))
        return dataset

    def load_with_limit(self, max_samples: int) -> Any:
        """Load a limited number of samples (for eval/testing)."""
        logger.info(f"Loading {max_samples} samples from dataset {self.source.name}.")
        dataset = self.load()
        if hasattr(dataset, "take"):
            # It's an IterableDataset
            taken = dataset.take(max_samples)
            return Dataset.from_generator(lambda: (yield from taken))
        else:
            # If not streaming
            return dataset.select(range(min(len(dataset), max_samples)))

    def estimate_dataset_size(self) -> Optional[int]:
        """Try to get dataset size without full download."""
        try:
            from datasets import load_dataset_builder
            builder = load_dataset_builder(self.source.name, trust_remote_code=self.source.trust_remote_code)
            if builder.info.splits and self.source.split in builder.info.splits:
                return builder.info.splits[self.source.split].num_examples
        except Exception as e:
            logger.warning(f"Could not estimate size for {self.source.name}: {e}")
        return None
        
    def __repr__(self) -> str:
        return f"VasudhaDatasetLoader(source={self.source.name}, buffer_size={self.buffer_size})"
