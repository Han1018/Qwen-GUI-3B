import tqdm
import torch
from data.data_utils import AverageMeter, ProgressMeter, Summary, dict_to_cuda

# no test yet
def validate(val_loader, model_engine, processor, epoch, global_step, writer, args):
    model_engine.eval()

    metric = 0
    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict)
        if args.precision == "fp16":
            input_dict["pixel_values"] = input_dict["pixel_values"].half()
        elif args.precision == "bf16":
            input_dict["pixel_values"] = input_dict["pixel_values"].bfloat16()
        else:
            input_dict["pixel_values"] = input_dict["pixel_values"].float()

        with torch.no_grad():
            output_dict = model_engine(
                pixel_values=input_dict["pixel_values"],
                image_sizes=input_dict["image_sizes"],
                input_ids=input_dict["input_ids"],
                labels=input_dict["labels"],
                output_hidden_states=True,
            )
            metric += output_dict['loss'].item()
    # a navie way to calculate the metric by their loss
    return 1 / metric