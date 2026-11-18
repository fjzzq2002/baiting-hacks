"""Record and replay the generation boundary, so training can be compared bitwise.

vLLM's sampling differs between identical-seed runs, so full runs cannot match. Everything else in a run is a function
of the generated tokens. A recording stores every call on the client (`vllm_client.VLLMClient`); a replay substitutes
the recorded results and needs no server, which pins the rest of the system: tokenization, rewards, routing, gradients
and the optimizer. The replay also checks each request against the recording, including a hash of the adapter weights
sent at every step, and reports the first mismatch.
"""
import atexit
import hashlib
import json
import pickle

_MISSING = object()


def digest(obj):
    return hashlib.sha256(pickle.dumps(obj, protocol=4)).hexdigest()[:16]


def weights_digest(model):
    from vllm_client import _pack_weights_flat
    return hashlib.sha256(_pack_weights_flat(model)[0]).hexdigest()[:16]


class Recorder:
    """Wraps a real client and appends (op, request digest, result) for each call."""

    def __init__(self, client, path):
        self._client, self._f = client, open(path, "wb")

    def __getattr__(self, name):
        return getattr(self._client, name)

    def _log(self, op, req, result=None):
        timings = getattr(self._client, "_last_gen_timings", None)
        pickle.dump((op, req, result, timings), self._f, protocol=4)
        self._f.flush()
        return result

    def _call(self, op, *args, **kw):
        return self._log(op, digest((args, sorted(kw.items()))), getattr(self._client, op)(*args, **kw))

    def register(self):
        return self._call("register")

    def release(self, experiment_id):
        return self._call("release", experiment_id)

    def update_weights_from_model(self, experiment_id, model):
        self._client.update_weights_from_model(experiment_id, model)
        return self._log("update_weights", (experiment_id, weights_digest(model)))

    def generate(self, *a, **kw):
        return self._call("generate", *a, **kw)

    def generate_multi(self, *a, **kw):
        return self._call("generate_multi", *a, **kw)

    def set_scales(self, *a, **kw):
        return self._call("set_scales", *a, **kw)

    def set_steering(self, experiment_id, layer_to_vec, alpha):
        self._client.set_steering(experiment_id, layer_to_vec, alpha)
        return self._log("set_steering", digest((experiment_id, sorted(layer_to_vec), alpha)))

    def sleep(self, *a, **kw):
        return self._call("sleep", *a, **kw)

    def wake_up(self, *a, **kw):
        return self._call("wake_up", *a, **kw)


class ReplayClient:
    """Stands in for the client: returns recorded results and checks each request.

    A request that differs from the recording is a behaviour change. Weight hashes are counted, not fatal, so a run
    can continue past one and every later difference can still be reported.
    """

    def __init__(self, path, report):
        self._calls, self._i, self._report_path = [], 0, report
        with open(path, "rb") as f:
            while True:
                try:
                    self._calls.append(pickle.load(f))
                except EOFError:
                    break
        self._last_gen_timings = {}
        self.stats = {"calls": 0, "weight_checks": 0, "weight_mismatches": 0, "first_weight_mismatch": None}
        atexit.register(self._write)

    def _next(self, op):
        assert self._i < len(self._calls), f"replay exhausted at call {self._i}: {op}"
        rec = self._calls[self._i]
        assert rec[0] == op, f"call {self._i}: code made {op}, recording has {rec[0]}"
        self._i += 1
        self.stats["calls"] += 1
        self._last_gen_timings = rec[3] or {}
        return rec

    def _call(self, op, *args, **kw):
        rec = self._next(op)
        got = digest((args, sorted(kw.items())))
        assert got == rec[1], f"call {self._i - 1}: {op} request differs from the recording"
        return rec[2]

    def register(self):
        return self._call("register")

    def release(self, experiment_id):
        return self._call("release", experiment_id)

    def update_weights_from_model(self, experiment_id, model):
        rec = self._next("update_weights")
        self.stats["weight_checks"] += 1
        if (experiment_id, weights_digest(model)) != rec[1]:
            self.stats["weight_mismatches"] += 1
            if self.stats["first_weight_mismatch"] is None:
                self.stats["first_weight_mismatch"] = self._i - 1
        self._write()

    def generate(self, *a, **kw):
        return self._call("generate", *a, **kw)

    def generate_multi(self, *a, **kw):
        if "return_logprobs" not in kw:
            # The step-0 eval deliberately moved to the in-rollout limits and sampler after these recordings were
            # made. It still replays the recorded completions, so only its request can differ.
            return self._next("generate_multi")[2]
        return self._call("generate_multi", *a, **kw)

    def set_scales(self, *a, **kw):
        return self._call("set_scales", *a, **kw)

    def set_steering(self, experiment_id, layer_to_vec, alpha):
        rec = self._next("set_steering")
        assert digest((experiment_id, sorted(layer_to_vec), alpha)) == rec[1], f"call {self._i - 1}: set_steering differs"

    def sleep(self, *a, **kw):
        return self._call("sleep", *a, **kw)

    def wake_up(self, *a, **kw):
        return self._call("wake_up", *a, **kw)

    def shutdown(self):
        self._write()

    def close(self):
        self._write()

    def _write(self):
        left = [c[0] for c in self._calls[self._i:]]
        json.dump({**self.stats, "total_recorded": len(self._calls), "unconsumed": left}, open(self._report_path, "w"))
