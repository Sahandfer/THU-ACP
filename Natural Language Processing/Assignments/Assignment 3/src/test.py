def test(model, test_iterator):
    num_correct, num_total = 0, 0

    model.eval()
    for batch in test_iterator:
        x, l = batch.sentence
        y = batch.label
        output = model(x, l)

        preds = output.argmax(dim=1)
        num_correct += (preds == y).sum()
        num_total += len(batch)

    acc = num_correct * 100 / num_total

    print(f"\n Testing Accuracy: {acc}%\n")