import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, ConcatDataset
from transformers import AutoProcessor, AutoModelForImageTextToText
from accelerate import Accelerator
from accelerate.utils import set_seed, DummyOptim, DummyScheduler

from data import SimpleDataset, make_collate_fn
import time
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal Qwen3.5-Dense VLM SFT Training")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--save_checkpoint", action="store_true")
    parser.add_argument("--attn_implementation", type=str, default="sdpa",
                        help="注意力实现: sdpa/flash_attention_2/flash_attention_3")
    return parser.parse_args()


def setup_model(model_path, attn_implementation="sdpa"):
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )
    if not hasattr(model.config, "hidden_size") or model.config.hidden_size is None:
        for sub_name in ["text_config", "language_config"]:
            sub_cfg = getattr(model.config, sub_name, None)
            if sub_cfg is not None and hasattr(sub_cfg, "hidden_size"):
                model.config.hidden_size = sub_cfg.hidden_size
                break
    return model


def setup_processor(model_path):
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return processor


def main(args):
    set_seed(args.seed)
    accelerator = Accelerator()

    model = setup_model(args.model_path, attn_implementation=args.attn_implementation)
    processor = setup_processor(args.model_path)
    collate_fn = make_collate_fn(processor, args.max_seq_length)

    world_size = accelerator.num_processes
    micro_batch_size = accelerator.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"]

    base_dataset = SimpleDataset(args.data_path)
    min_samples = micro_batch_size * world_size
    k = (min_samples + len(base_dataset) - 1) // len(base_dataset)
    dataset = ConcatDataset([base_dataset] * k) if k > 1 else base_dataset

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=dist.get_rank(), shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    optimizer = DummyOptim(model.parameters())
    scheduler = DummyScheduler(optimizer)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    model.train()

    total_steps = len(dataloader) * args.epochs
    if accelerator.is_main_process:
        print(f"Training: epochs={args.epochs}, micro_batch_size={micro_batch_size}, "
              f"samples={len(dataset)}, steps={total_steps}")

    # hygon prof
    profile = True
    prof = None
    profile_dir = "./prof_deepspeed_qwen35_08b_mbs2gbs8"
    if profile:
        def trace_handler(p):
            from pathlib import Path
            Path(f"{profile_dir}").mkdir(parents=True, exist_ok=True)
            # if torch.distributed.get_rank() == 0:
            rank = accelerator.process_index
            if accelerator.is_main_process:
                print(p.key_averages(group_by_input_shape=True,
                                     group_by_stack_n=5).table(sort_by="self_cuda_time_total",
                                                               row_limit=-1,
                                                               max_src_column_width=100,
                                                               max_name_column_width=280,
                                                               max_shapes_column_width=200))

            p.export_chrome_trace(f"{profile_dir}/trace_rank{rank}_step{p.step_num}.json")

        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        prof = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=8 - 1,
                warmup=1,
                active=1,
                repeat=1,
            ),
            on_trace_ready=trace_handler,
            record_shapes=True,
            with_stack=True,
        )
        prof.start()

    global_step = 0
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for step, batch in enumerate(dataloader):
            start_time = time.time()
            batch = {k: v.to(accelerator.device, non_blocking=True) if hasattr(v, "to") else v for k, v in batch.items()}
            with accelerator.accumulate(model):
                loss = model(**batch, use_cache=False).loss
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            global_step += 1

            end_time = time.time()
            # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)) + f".{int((t % 1) * 1000):03d}")

            # hygon prof
            if profile is not None:
                prof.step()
            
            if accelerator.is_main_process:
                # cost_time = f"Step time: {end_time- start_time}"
                # cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                # loss_info = f"Step {global_step}/{total_steps} Loss: {loss.item():.6f}"
                cost_time = end_time - start_time
                cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                loss_info = f"Step {global_step}/{total_steps} Loss: {loss.item():.6f}"

                print(f"[{cur_time}] {loss_info} | Step time: {cost_time:.4f}s")

    # hygon prof
    if profile is not None:
        prof.stop()

    if args.save_checkpoint:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            os.makedirs(args.output_dir, exist_ok=True)
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(args.output_dir, state_dict=accelerator.get_state_dict(model))
            processor.save_pretrained(args.output_dir)
            print(f"Model saved to {args.output_dir}")

    accelerator.end_training()
    if accelerator.is_main_process:
        print("Training completed!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
