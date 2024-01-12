import os
import time
import colossalai

class ColTimer:

    def __init__(self, func, args, kwargs, trials=3, warmup_trials=3, in_ms=True, device="cuda") -> None:
        assert func is not None

        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.warmup_trials = warmup_trials
        self.trials = trials
        self.in_ms = in_ms

        self.device = device
        self.res = None

    def time(self):
        if self.device == "cuda":
            return self.time_torch_cuda()
        elif self.device == "cpu":
            return self.time_cpu()
        return None

    def time_cpu(self):

        for _ in range(self.warmup_trials):
            self.func(*self.args, **self.kwargs)

        start_t = time.perf_counter()
        for _ in range(self.trials):
            res = self.func(*self.args, **self.kwargs)
        end_t = time.perf_counter()

        elapsed_time = (start_t - end_t) / self.trials

        if self.in_ms:
            return elapsed_time * 1000

        self.res = res

        return elapsed_time

    def time_torch_cuda(self):
        import torch

        start_evt = []
        end_evt = []
        for _ in range(0, self.trials):
            start_evt.append(torch.cuda.Event(enable_timing=True))
            end_evt.append(torch.cuda.Event(enable_timing=True))

        for trial_idx in range(0, self.trials + self.warmup_trials):
            evt_idx = trial_idx - self.warmup_trials

            if evt_idx >= 0:
                start_evt[evt_idx].record()

            self.func(*self.args, **self.kwargs)

            if evt_idx >= 0:
                end_evt[evt_idx].record()

        torch.cuda.synchronize()

        ops_elapsed_time = 0
        for evt_idx in range(0, self.trials):
            ops_elapsed_time += start_evt[evt_idx].elapsed_time(end_evt[evt_idx])
        ops_elapsed_time = ops_elapsed_time / self.trials

        if self.in_ms:
            return ops_elapsed_time
        return ops_elapsed_time / 1000

    def result(self):
        assert self.res is not None
        res = self.res
        self.res = None
        return res
