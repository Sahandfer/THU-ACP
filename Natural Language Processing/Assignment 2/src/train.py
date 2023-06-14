import torch.optim as optim
from tqdm.auto import tqdm, trange
from torch.utils.data import DataLoader


def train(model, dataset, args):
    try:
        print("> Training")
        dl = DataLoader(
            dataset,
            collate_fn=dataset.collate_fn,
            batch_size=args.batch_size,
        )
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

        optimizer = optim.SparseAdam(model.parameters(), lr=args.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(dl))

        global_step = 0
        epochs_trained = 0

        # Loss
        train_loss = 0.0

        model.zero_grad()

        # Iterator
        train_iterator = trange(epochs_trained, args.num_train_epochs, desc="Epoch")

        for _ in train_iterator:
            epoch_iterator = tqdm(dl, desc="Steps")
            for _, batch in enumerate(epoch_iterator):
                model.train()
                word_pos, ctx, neg = batch
                word_pos = word_pos.to(args.device)
                ctx = ctx.to(args.device)
                neg = neg.to(args.device)
                loss = model(word_pos, ctx, neg)

                step_loss = loss.mean().item()
                train_loss += step_loss

                optimizer.zero_grad()

                loss.mean().backward()

                optimizer.step()
                scheduler.step()
                model.zero_grad()

                global_step += 1

                if global_step % args.log_loss == 0:
                    print(f"Loss at step {global_step} --> {train_loss/global_step}")

            model.save(args.output_dir, f"word_embedding_{args.embedding_size}")
            print(train_loss / global_step)
            epoch_iterator.close()

        print(">>> End of Training <<<")

    except KeyboardInterrupt:
        model.save(args.output_dir, f"word_embedding_{args.embedding_size}")
        print("\n > Shutdown")