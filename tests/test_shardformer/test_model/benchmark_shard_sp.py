import math
import torch
import torch.distributed as dist
import colossalai
from colossalai.logging import disable_existing_loggers
from colossalai.testing import spawn
from tests.kit.model_zoo import model_zoo
import tests.test_shardformer.test_model.benchmark_shard_config as bench_config
from tests.test_shardformer.test_model.benchmark_utils import ColTimer
from tests.test_shardformer.test_model._utils import (
    build_model_from_hybrid_plugin,
)


def profile_shard(rank, model_fn, criterion, args, kwargs, file_path='./logs/profile_shard_sp', trials=10):
    if rank == 0:
        print("Profiling ...")

    torch.cuda.empty_cache()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        #schedule=torch.profiler.schedule(wait=1, warmup=1, active=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(file_path),
        profile_memory=True,
    ) as prof:

        for _ in range(trials):
            prof.step()
            model_output = model_fn(*args, **kwargs)
            model_loss = criterion(model_output)
            model_loss.backward()
            torch.cuda.synchronize()

    torch.cuda.memory._record_memory_history()

    for _ in range(trials):
        prof.step()
        model_output = model_fn(*args, **kwargs)
        model_loss = criterion(model_output)
        model_loss.backward()
        torch.cuda.synchronize()

    torch.cuda.memory._dump_snapshot("my_snapshot.pickle")


def bench_time_forward(rank, model_fn, args, kwargs):
    torch.cuda.synchronize()
    if rank == 0:
        print("Benching forward ...")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    timer = ColTimer(model_fn, args, kwargs, in_ms=False)

    elaps_time = timer.time()
    peak_memory = torch.cuda.max_memory_allocated()

    print(f"[{rank}] Memory: {peak_memory / 1024 / 1024 / 1024} GB")
    print(f"[{rank}] Time: {elaps_time}")


def bench_time_backward(rank, model_fn, criterion, args, kwargs, times=3):
    if rank == 0:
        print("Benching backward ...")

    elaps_time = 0
    
    for _ in range(times):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model_output = model_fn(*args, **kwargs)
        model_loss = criterion(model_output)

        timer = ColTimer(model_loss.backward, (), {}, in_ms=False, trials=1, warmup_trials=0)

        elaps_time += timer.time()
        peak_memory = torch.cuda.max_memory_allocated()

    print(f"[{rank}] Memory: {peak_memory / 1024 / 1024 / 1024} GB")
    print(f"[{rank}] Time: {elaps_time / times}")


def bench_time(rank, model_fn, criterion, args, kwargs):
    torch.cuda.synchronize()
    if rank == 0:
        print("Benching time ...")

    def func():
        model_output = model_fn(*args, **kwargs)
        model_loss = criterion(model_output)
        model_loss.backward()

    # forward
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    timer = ColTimer(func, (), {}, in_ms=False, trials=1,warmup_trials=0)

    elaps_time = timer.time()
    peak_memory = torch.cuda.max_memory_allocated()

    print(f"[{rank}] Memory: {peak_memory / 1024 / 1024 / 1024} GB")
    print(f"[{rank}] Time: {elaps_time}")


def bench_shard(rank):

    # load model and input
    sub_model_zoo = model_zoo.get_sub_registry("transformers_llama")

    for name, (model_fn, data_gen_fn, output_transform_fn, loss_fn, _) in sub_model_zoo.items():
        if name != "transformers_llama":
            continue
        org_model, org_optimizer, sharded_model, sharded_optimizer, criterion, booster = build_model_from_hybrid_plugin(
            model_fn, loss_fn, bench_config.shard_config
        )

        org_model.cuda()
        sharded_model.cuda()

        def _criterion(outputs, inputs):
            outputs = output_transform_fn(outputs)
            loss = criterion(outputs)
            return loss

        data = data_gen_fn()
        print(data)
        for k, v in data.items():
            size = list(v.shape)
            tg_size = [1] * len(size)
            tg_size[1] = 64 * 12
            data[k] = v.repeat(tg_size)
        for k, v in data.items():
            print(v.shape)

        if (
            booster.plugin.shard_config.enable_sequence_parallelism
            and booster.plugin.shard_config.sequence_parallelism_mode in ["1", "2"]
            and booster.plugin.tp_size != 0
        ):
            seq_len = data["input_ids"].shape[-1]
            lcm = booster.plugin.tp_size * seq_len // math.gcd(booster.plugin.tp_size, seq_len)
            times = lcm // seq_len
            input_shape = data["input_ids"].shape
            for k, v in data.items():
                if v.shape == input_shape:
                    data[k] = v.repeat((1,) * (v.dim() - 1) + (times,))

        shard_test_data = {}
        for k, v in data.items():
            if k == "labels":
                shard_test_data[k] = data[k].clone()
            elif k == "attention_mask":
                shard_test_data[k] = (
                    torch.chunk(data[k].clone(), booster.plugin.shard_config.sequence_parallel_size, dim=1)[dist.get_rank()]
                    if booster.plugin.shard_config.enable_sequence_parallelism
                    and booster.plugin.shard_config.sequence_parallelism_mode in ["2"]
                    else data[k].clone()
                )
            else:
                shard_test_data[k] = (
                    torch.chunk(data[k].clone(), booster.plugin.shard_config.sequence_parallel_size, dim=1)[dist.get_rank()]
                    if booster.plugin.shard_config.enable_sequence_parallelism
                    and booster.plugin.shard_config.sequence_parallelism_mode in ["2", "3"]
                    else data[k].clone()
                )
        unshard_test_data = {}
        for k, v in data.items():
            unshard_test_data[k] = data[k].clone()

        args = ()
        sharded_model.train()
        if booster.plugin.stage_manager is not None:
            raise RuntimeError('Benchmark_shard: do not support pipeline yet!')
            for k, v in shard_test_data.items():
                if torch.is_tensor(v) or "Tensor" in v.__class__.__name__:
                    new_shape = [1] * v.dim()
                    new_shape[0] = 4
                    shard_test_data[k] = v.to("cuda").repeat(*new_shape)

            data_iter = iter([shard_test_data])
            sharded_output = booster.execute_pipeline(
                data_iter, sharded_model, _criterion, sharded_optimizer, return_loss=True, return_outputs=True
            )
            sharded_loss = sharded_output["loss"]

        else:
            shard_test_data = {k: v.cuda() for k, v in shard_test_data.items()}
            #sharded_output = sharded_model(**shard_test_data)
            #sharded_loss = criterion(sharded_output)
            #sharded_optimizer.backward(sharded_loss)
            #bench_time_forward(rank, sharded_model, args, shard_test_data)
            #bench_time_backward(rank, sharded_model, criterion, args, shard_test_data)
            bench_time(rank, sharded_model, criterion, args, shard_test_data)

            if bench_config.profile is True:
                profile_shard(rank, sharded_model, criterion, args, shard_test_data, file_path='./logs/bench_sp_sharded_model')

        org_model.train()
        if booster.plugin.stage_manager is not None:
            for k, v in unshard_test_data.items():
                if torch.is_tensor(v) or "Tensor" in v.__class__.__name__:
                    new_shape = [1] * v.dim()
                    new_shape[0] = 4
                    unshard_test_data[k] = v.to("cuda").repeat(*new_shape)
        unshard_test_data = {k: v.cuda() for k, v in unshard_test_data.items()}
        #org_output = org_model(**unshard_test_data)
        #org_loss = criterion(org_output)
        #org_loss.backward()
        #bench_time_forward(rank, org_model, args, unshard_test_data)
        #bench_time_backward(rank, org_model, criterion, args, unshard_test_data)
        #bench_time(rank, org_model, criterion, args, unshard_test_data)

        #if bench_config.profile is True:
        #    profile_shard(rank, org_model, criterion, args, unshard_test_data, file_path='./logs/bench_sp_org_model')



def bench_launch(rank, world_size, port):
    disable_existing_loggers()
    colossalai.launch(config={}, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    bench_shard(rank)


if __name__ == '__main__':
    spawn(bench_launch, bench_config.workers) 
