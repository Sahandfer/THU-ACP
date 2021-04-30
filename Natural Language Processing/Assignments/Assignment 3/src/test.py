def test(model, test_iterator, use_improved=True):
    num_correct, num_total = 0, 0

    model.eval()
    for batch in test_iterator:
        if use_improved:
            output, y = model(batch)
        else:
            x, l = batch.sentence
            y = batch.label
            output = model(x, l)

        preds = output.argmax(dim=1)
        num_correct += (preds == y).sum()
        num_total += preds.size()[0]

    acc = num_correct * 100 / num_total

    print(f"\n Testing Accuracy: {acc}%\n")