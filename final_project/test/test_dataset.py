"""Dataset and collate smoke tests."""

from final_project.src.config import ftad_split_path
from final_project.src.dataset import (
    EOTDataset,
    build_vocab_from_train,
    collate_bow,
    collate_ids,
    load_ftad_split,
)
from final_project.src.preprocess import parse_ftad_line


def test_parse_ftad_line() -> None:
    example = parse_ftad_line("agent:hi|user:hello\tokay then\t1\n")
    assert example.label == 1
    assert example.utterance == "okay then"


def test_dataset_and_collate() -> None:
    examples = load_ftad_split(ftad_split_path("train"))[:16]
    dataset = EOTDataset(examples, text_mode="normalized", use_context=True)
    assert len(dataset) == 16

    batch = [dataset[i] for i in range(4)]
    vocab = build_vocab_from_train(examples, text_mode="normalized", use_context=True)
    bow_batch = collate_bow(batch, vocab)
    assert bow_batch["features"].shape[0] == 4
    assert bow_batch["labels"].shape == (4,)

    id_batch = collate_ids(batch, vocab, max_len=32)
    assert id_batch["input_ids"].shape == (4, 32)
