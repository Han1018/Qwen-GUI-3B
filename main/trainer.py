import pdb
import time
import wandb
from data.data_utils import AverageMeter, ProgressMeter, Summary, dict_to_cuda

def train(
    train_loader,
    model,
    epoch,
    scheduler,
    writer,
    train_iter,
    args,
):

    """Main training loop."""
    batch_time = AverageMeter("Batch time (s)", ":6.3f")
    iter_time = AverageMeter("Iter time (s)", ":6.3f")
    epoch_time = AverageMeter("Epoch time (h)", ":6.3f")
    remain_time = AverageMeter("Remain time (h)", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    seq_len = AverageMeter("Seq Len", ":6.3f")
    ctx_len = AverageMeter("Ctx Len", ":6.3f")
    vis_len = AverageMeter("Vis Len", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")

    progress = ProgressMeter(
        args.steps_per_epoch,
        [
            batch_time,
            iter_time,
            epoch_time,
            remain_time,
            losses,
            seq_len,
            ctx_len,
            vis_len,
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()
    end = time.time()

    # for global_step in range(args.steps_per_epoch):
    for local_step in range(args.steps_per_epoch):
        global_step = local_step + epoch * args.steps_per_epoch
        for i in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            data_time.update(time.time() - end)
            input_dict = dict_to_cuda(input_dict)

            # incase of visual blind
            if input_dict["pixel_values"] is not None:
                if args.precision == "fp16":
                    input_dict["pixel_values"] = input_dict["pixel_values"].half()
                elif args.precision == "bf16":
                    input_dict["pixel_values"] = input_dict["pixel_values"].bfloat16()
                else:
                    input_dict["pixel_values"] = input_dict["pixel_values"].float()

            forward_dict = dict(
                pixel_values=input_dict["pixel_values"],
                input_ids=input_dict["input_ids"],
                labels=input_dict["labels"],
                output_hidden_states=True
                )

            forward_dict.update(image_grid_thw=input_dict["image_sizes"]) if "image_sizes" in input_dict else None
            forward_dict.update(patch_assign=input_dict["patch_assign"]) if "patch_assign" in input_dict else None
            forward_dict.update(patch_assign_len=input_dict["patch_assign_len"]) if "patch_assign_len" in input_dict else None
            forward_dict.update(patch_pos=input_dict["patch_pos"]) if "patch_pos" in input_dict else None
            forward_dict.update(select_mask=input_dict["select_mask"]) if "select_mask" in input_dict else None
            
            output_dict = model.forward(
                **forward_dict
            )
            loss = output_dict["loss"]
            losses.update(loss.item(), input_dict["input_ids"].size(0))
            model.backward(loss)
            model.step()

        # measure elapsed time
        batch_sec  = (time.time() - end)
        iter_sec = batch_sec / args.grad_accumulation_steps
        batch_time.update(batch_sec)
        iter_time.update(iter_sec)
        epoch_time.update(iter_sec * args.steps_per_epoch / 3600)
        remain_time.update( (iter_sec * (args.steps_per_epoch - local_step-1)) / 3600)
        end = time.time()
        seq_len.update(input_dict["input_ids"].size(1))
        if "ctx_len" in output_dict:
            ctx_len.update(output_dict["ctx_len"].float().mean())
            vis_len.update(output_dict["vis_len"].float().mean())

        if global_step % args.print_freq == 0:
            if args.distributed:
                batch_time.all_reduce()
                iter_time.all_reduce()
                epoch_time.all_reduce()
                remain_time.all_reduce()
                data_time.all_reduce()
                losses.all_reduce()
                seq_len.all_reduce()
                ctx_len.all_reduce()
                vis_len.all_reduce()

            if args.global_rank == 0:
                progress.display(global_step + 1)
                if not args.debug:
                    writer.add_scalar("train/loss", losses.avg, global_step)
                    writer.add_scalar(
                        "metrics/total_secs_per_batch", batch_time.avg, global_step
                    )
                    writer.add_scalar(
                        "metrics/total_secs_per_iter", iter_time.avg, global_step
                    )
                    writer.add_scalar(
                        "metrics/total_secs_per_epoch", epoch_time.avg, global_step
                    )
                    writer.add_scalar(
                        "metrics/data_secs_per_batch", data_time.avg, global_step
                    )

                    wandb.log({"epoch": epoch, 
                                "train loss": losses.avg,
                                "batch_time": batch_time.avg,
                                "iter_time": iter_time.avg,
                                "epoch_time": epoch_time.avg,
                                "seq_len (pre)": seq_len.avg,
                                "ctx_len (post)": ctx_len.avg,
                                "vis_len (post)": vis_len.avg,
                                },
                                step=global_step)

            batch_time.reset()
            epoch_time.reset()
            iter_time.reset()
            remain_time.reset()
            data_time.reset()
            losses.reset()
            seq_len.reset()
            ctx_len.reset()
            vis_len.reset()

        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if args.global_rank == 0 and not args.debug:
                writer.add_scalar("train/lr", curr_lr[0], global_step)

    return train_iter, global_step