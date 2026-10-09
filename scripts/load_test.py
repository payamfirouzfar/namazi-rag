"""Send many questions to the running app at the same time and see how it copes.

    python -m scripts.load_test --users 10000 --concurrency 100

Every "user" asks one question taken from the eval file. At the end you get the speed,
the answer times and the number of failures. Use a test copy of the app (rate limit off,
its own database), see docs/SETUP.md.
"""
import argparse
import asyncio
import csv
import sys
import time
from pathlib import Path

import httpx

from scripts.tune import load_eval


async def one_user(client, url, question, results, wait):
    async with wait:  # at most --concurrency users at the same moment
        started = time.perf_counter()
        try:
            response = await client.post(f"{url}/v1/ask", json={"question": question})
            code = response.status_code
        except httpx.HTTPError as error:
            code = error.__class__.__name__  # for example ReadTimeout or ConnectError
        results.append((code, time.perf_counter() - started))


async def run(url, questions, concurrency, timeout):
    results, wait = [], asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency)
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        started = time.perf_counter()
        await asyncio.gather(*(one_user(client, url, q, results, wait) for q in questions))
        return results, time.perf_counter() - started


def percent(times, share):
    return times[min(len(times) - 1, int(len(times) * share))]


def save_row(path, name, users, concurrency, good, seconds):
    row = {"test": name, "users": users, "concurrency": concurrency, "answered": len(good), "seconds": round(seconds),
           "questions_per_second": round(users / seconds, 1), "median_s": round(percent(good, 0.5), 2),
           "p95_s": round(percent(good, 0.95), 2), "p99_s": round(percent(good, 0.99), 2)}
    new_file = not Path(path).exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    parser.add_argument("--users", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--eval", default="data/eval/eval_questions.jsonl")
    parser.add_argument("--name", default="", help="a label for this run")
    parser.add_argument("--save", default="", help="add one row with the result to this csv file")
    args = parser.parse_args()

    # English only, the same questions again and again if there are more users than questions
    pool = [i["question"] for i in load_eval(args.eval) if i["question"].isascii()]
    questions = [pool[n % len(pool)] for n in range(args.users)]
    print(f"{args.users} users, {args.concurrency} at the same time, {len(pool)} different questions")

    results, seconds = asyncio.run(run(args.url, questions, args.concurrency, args.timeout))
    good = sorted(t for code, t in results if code == 200)
    failed = [code for code, _ in results if code != 200]
    print(f"\nfinished in {seconds:.0f} s = {len(results) / seconds:.1f} questions per second")
    print(f"answered : {len(good)} of {len(results)}")
    for code in sorted(set(failed), key=str):
        print(f"failed   : {failed.count(code)} with {code}")
    if good:
        print(f"answer time: median {percent(good, 0.5):.2f} s, 95% under {percent(good, 0.95):.2f} s, "
              f"99% under {percent(good, 0.99):.2f} s, slowest {good[-1]:.2f} s")
    if args.save and good:
        save_row(args.save, args.name or f"{args.users} users", args.users, args.concurrency, good, seconds)
    return 0 if len(good) == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
