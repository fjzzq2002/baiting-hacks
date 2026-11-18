"""Run one golden scenario on the code in this checkout.

    python tests/golden/run.py <scenario> <out dir> [--seed 1] [--set key=value ...] [--record file | --replay file]
"""
import argparse
import ast
import copy
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RL = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [RL, HERE]
os.chdir(RL)


def main():
    from scenarios import LAZY_SCENARIOS, SCENARIOS
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", choices=sorted(SCENARIOS) + sorted(LAZY_SCENARIOS))
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--record", help="write the generation calls of this run to this file")
    ap.add_argument("--replay", help="take generation from this recording instead of vLLM")
    a = ap.parse_args()
    params = copy.deepcopy(SCENARIOS[a.scenario] if a.scenario in SCENARIOS else LAZY_SCENARIOS[a.scenario]())
    params.update(seed=a.seed, output_dir=a.out, run_name=f"golden_{a.scenario}_s{a.seed}")
    for kv in a.set:
        k, v = kv.split("=", 1)
        try:
            params[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            params[k] = v
    os.makedirs(a.out, exist_ok=True)
    import replay
    import vllm_client
    if a.record:
        real = vllm_client.VLLMClient
        vllm_client.VLLMClient = lambda addr: replay.Recorder(real(addr), a.record)
    if a.replay:
        vllm_client.VLLMClient = lambda addr: replay.ReplayClient(a.replay, os.path.join(a.out, "replay_report.json"))
        params.update(vllm_spawn=False, vllm_server="replay", eval_at_end=False)  # the recordings stop at the last step
    from train import train_main
    train_main(params)


if __name__ == "__main__":
    main()
