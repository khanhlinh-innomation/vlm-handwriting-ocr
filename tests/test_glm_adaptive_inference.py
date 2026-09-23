from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

from vlm_handwriting.glm import predict_one, predict_one_detailed


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeGenerated(list):
    @property
    def shape(self):
        return (len(self),)

    def __getitem__(self, item):
        value = super().__getitem__(item)
        return FakeScalar(value) if isinstance(item, int) else FakeGenerated(value)


class FakeRow(FakeGenerated):
    pass


class FakeInputIds:
    shape = (1, 2)


class FakeInputs(dict):
    def to(self, _device):
        return self


class FakeProcessor:
    def apply_chat_template(self, *_args, **_kwargs):
        return FakeInputs(input_ids=FakeInputIds(), token_type_ids=object())

    def decode(self, generated, **_kwargs):
        return " ".join(str(item) for item in generated)


class FakeModel:
    def __init__(self, generated_sequences):
        self.generated_sequences = iter(generated_sequences)
        self.generation_config = SimpleNamespace(eos_token_id=9)
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        generated = next(self.generated_sequences)
        return [FakeRow([100, 101, *generated])]


class FakeTorch:
    cuda = SimpleNamespace(synchronize=lambda: None)

    @staticmethod
    def inference_mode():
        return nullcontext()


def arguments(model, generation):
    return dict(
        torch=FakeTorch(),
        processor=FakeProcessor(),
        model=model,
        device="cuda",
        image_path=Path("page.jpg"),
        prompt="Text Recognition:",
        generation=generation,
    )


def test_cap_without_eos_uses_configured_fallback():
    model = FakeModel([[1, 2, 3, 4], [7, 9]])
    prediction, _latency, metadata = predict_one_detailed(
        **arguments(
            model,
            {
                "max_new_tokens": 4,
                "do_sample": False,
                "repetition_penalty": 1.1,
                "fallback_no_repeat_ngram_size": 8,
            },
        )
    )
    assert prediction == "7 9"
    assert metadata["fallback_used"] is True
    assert metadata["primary"]["hit_max_new_tokens"] is True
    assert metadata["fallback"]["ended_with_eos"] is True
    assert "no_repeat_ngram_size" not in model.calls[0]
    assert model.calls[1]["no_repeat_ngram_size"] == 8


def test_eos_keeps_primary_and_legacy_return_shape():
    model = FakeModel([[7, 9]])
    prediction, latency = predict_one(
        **arguments(
            model,
            {
                "max_new_tokens": 4,
                "do_sample": False,
                "repetition_penalty": 1.1,
                "fallback_no_repeat_ngram_size": 8,
            },
        )
    )
    assert prediction == "7 9"
    assert latency >= 0
    assert len(model.calls) == 1
