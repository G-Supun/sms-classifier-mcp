
from src.classifier import SMSClassifier


def test_classifier_model_info():
    classifier = SMSClassifier(model_dir="model")
    info = classifier.model_info()

    assert info["model"] == "SMSSequenceCNN"
    assert info["parameters"] == 2002881
    assert info["max_sequence_length"] == 64


def test_classify_promotional_sms():
    classifier = SMSClassifier(model_dir="model")

    result = classifier.classify(
        "Congratulations! You have won a free prize. Click here now!"
    )

    assert result["category"] == "promotional"
    assert 0 <= result["confidence"] <= 1


def test_classify_transactional_sms():
    classifier = SMSClassifier(model_dir="model")

    result = classifier.classify(
        "Your bank transfer of Rs. 2500 was completed successfully."
    )

    assert result["category"] == "transactional"
    assert 0 <= result["confidence"] <= 1


def test_empty_message_is_rejected():
    classifier = SMSClassifier(model_dir="model")

    try:
        classifier.classify("")
        assert False, "Expected ValueError for empty SMS"
    except ValueError:
        pass


def test_batch_returns_one_result_per_message():
    classifier = SMSClassifier(model_dir="model")
    messages = [
        "Congratulations! Claim your free prize!",
        "Your bank transfer was completed.",
    ]

    results = classifier.classify_batch(messages)

    assert len(results) == len(messages)
    assert all("category" in result for result in results)
