"""Fixed XOR truth table and document-local, left-context token prediction."""

from dataclasses import dataclass
import random

import torch
from torch import Tensor
from torch.nn.functional import one_hot

Examples = tuple[Tensor, Tensor]
CLAUSE_LENGTH = 5


@dataclass(frozen=True)
class TaskData:
    train: Examples
    validation: Examples
    test: Examples
    classes: int


def xor() -> TaskData:
    x = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    table = x, torch.tensor([0, 1, 1, 0])
    return TaskData(table, table, table, 2)


def documents(seed: int, sizes: tuple[int, int, int], clauses: int) -> list[list[str]]:
    """Each clause names a direction followed by four consecutive ring symbols."""
    rng = random.Random(seed)
    symbols = "abcdef"
    if min(sizes) < 1 or clauses < 1 or sum(sizes) > (2 * len(symbols)) ** clauses:
        raise ValueError("Corpus requires enough distinct nonempty documents")
    seen: set[str] = set()
    splits = []
    for size in sizes:
        split = []
        while len(split) < size:
            words = []
            for _ in range(clauses):
                start = rng.randrange(len(symbols))
                direction = rng.choice((-1, 1))
                words.append("up" if direction == 1 else "down")
                words.extend(symbols[(start + direction * i) % len(symbols)] for i in range(CLAUSE_LENGTH - 1))
            doc = " ".join(words)
            if doc not in seen:
                seen.add(doc)
                split.append(doc)
        splits.append(split)
    return splits


def vocabulary(training_docs: list[str]) -> dict[str, int]:
    # Zero is reserved for out-of-vocabulary input and target tokens.
    return {word: i for i, word in enumerate(sorted({w for d in training_docs for w in d.split()}), 1)}


def examples(docs: list[str], vocab: dict[str, int], context: int) -> Examples:
    if context < 1:
        raise ValueError("Context length must be positive")
    windows = []
    for doc in docs:
        ids = torch.tensor([vocab.get(word, 0) for word in doc.split()], dtype=torch.long)
        if len(ids) > context:
            windows.append(ids.unfold(0, context + 1, 1))
    if not windows:
        raise ValueError("No document contains a complete prediction window")
    rows = torch.cat(windows)
    return one_hot(rows[:, :-1], len(vocab) + 1).flatten(1).float(), rows[:, -1]


def token_task(seed: int, sizes: tuple[int, int, int], clauses: int, context: int) -> TaskData:
    train, validation, test = documents(seed, sizes, clauses)
    vocab = vocabulary(train)
    return TaskData(*(examples(split, vocab, context) for split in (train, validation, test)),
                    classes=len(vocab) + 1)
