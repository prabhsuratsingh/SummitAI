import asyncio, aiohttp, time, json, psutil, threading, csv, os
from pathlib import Path
from statistics import mean, quantiles
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

API_BASE = "http://meetai:8000"
MEETING_ID = "bench001"
AUDIO_FILE = "sample.wav" 

RESULTS_DIR = Path("./benchmark")
RESULTS_DIR.mkdir(exist_ok=True)

SYSTEM_LOG = RESULTS_DIR / "system_metrics.csv"
LAT_LOG = RESULTS_DIR / "latencies.csv"

import aiohttp, asyncio, time

import platform, psutil, subprocess, sys
from pathlib import Path

def log_system_specs(output_path="bench_results/system_info.txt"):
    Path("bench_results").mkdir(exist_ok=True)
    info = []

    # --- CPU / RAM ---
    info.append(f"CPU: {platform.processor()}")
    info.append(f"Cores: {psutil.cpu_count(logical=True)}")
    info.append(f"RAM: {round(psutil.virtual_memory().total / 1e9, 2)} GB")

    # --- GPU ---
    try:
        gpu_info = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True
        ).strip()
    except Exception:
        gpu_info = "No GPU detected"
    info.append(f"GPU: {gpu_info}")

    # --- Docker limits ---
    try:
        mem_limit = int(Path("/sys/fs/cgroup/memory/memory.limit_in_bytes").read_text())
        cpu_quota = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text().strip()
        info.append(f"Container memory limit: {round(mem_limit / 1e9, 2)} GB")
        info.append(f"Container CPU quota: {cpu_quota}")
    except Exception:
        info.append("Container resource info not available")

    # --- OS / Python ---
    info.append(f"OS: {platform.platform()}")
    info.append(f"Python: {sys.version.split()[0]}")

    with open(output_path, "w") as f:
        f.write("\n".join(info))
    print("🧠 System specs logged →", output_path)


async def wait_for_server(api_base="http://meetai:8000", timeout=60):
    print("⏳ Waiting for MeetAI server to be ready...")
    start = time.time()
    async with aiohttp.ClientSession() as session:
        while time.time() - start < timeout:
            try:
                async with session.get(f"{api_base}/docs") as resp:
                    if resp.status == 200:
                        print("✅ MeetAI server is ready!")
                        return True
            except Exception:
                await asyncio.sleep(2)
        raise TimeoutError("❌ MeetAI server not ready after 60s")



# ---------------- SYSTEM METRIC LOGGER ---------------- #
def log_system_metrics(stop_event, interval=1.0):
    with open(SYSTEM_LOG, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "cpu%", "ram%", "disk_read_MB", "disk_write_MB"])
        start = time.time()
        prev = psutil.disk_io_counters()
        while not stop_event.is_set():
            now = psutil.disk_io_counters()
            elapsed = time.time() - start
            writer.writerow([
                round(elapsed, 1),
                psutil.cpu_percent(),
                psutil.virtual_memory().percent,
                (now.read_bytes - prev.read_bytes) / (1024 * 1024),
                (now.write_bytes - prev.write_bytes) / (1024 * 1024)
            ])
            f.flush()
            prev = now
            time.sleep(interval)


# ---------------- BENCHMARK FUNCTIONS ---------------- #
async def upload_chunk(session, file_path, meeting_id):
    with open(file_path, "rb") as f:
        data = aiohttp.FormData()
        data.add_field("meeting_id", meeting_id)
        data.add_field("chunk", f, filename="chunk_1.webm", content_type="audio/webm")
        t0 = time.perf_counter()
        async with session.post(f"{API_BASE}/ai/upload_chunk/", data=data) as resp:
            await resp.text()
        return time.perf_counter() - t0


async def finalize_meeting(session, meeting_id):
    """Measures internal breakdown from /ai/finalize/."""
    t0 = time.perf_counter()
    async with session.post(f"{API_BASE}/ai/finalize/", params={"meeting_id": meeting_id}) as resp:
        text = await resp.text()
        try:
            data = json.loads(text)
        except:
            data = {"error": text}
    total_t = time.perf_counter() - t0
    return total_t, data


# ---------------- BENCHMARK DRIVER ---------------- #
async def run_benchmark():
    log_system_specs()
    await wait_for_server(API_BASE)
    stop_event = threading.Event()
    sys_thread = threading.Thread(target=log_system_metrics, args=(stop_event,), daemon=True)
    sys_thread.start()

    async with aiohttp.ClientSession() as session:
        print("\n🚀 Starting benchmark...")
        upload_lat = []
        for i in tqdm(range(3), desc="Uploading"):
            t = await upload_chunk(session, AUDIO_FILE, MEETING_ID)
            upload_lat.append(t)

        print("📦 Upload done. Avg upload latency:", round(mean(upload_lat), 2), "s")

        print("🧠 Running /ai/finalize/...")
        finalize_t, data = await finalize_meeting(session, MEETING_ID)
        print(f"✅ Finalize latency: {finalize_t:.2f}s")

        with open(LAT_LOG, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["phase", "latency_s"])
            for t in upload_lat:
                writer.writerow(["upload", round(t, 3)])
            writer.writerow(["finalize_total", round(finalize_t, 3)])

        print("\n--- SAMPLE SUMMARY ---")
        print(data.get("summary", "")[:400])

    stop_event.set()
    sys_thread.join(timeout=2)
    print("\n📝 Benchmark complete. Logs saved to:", RESULTS_DIR)


# ---------------- VISUALIZATION ---------------- #
def visualize_results():
    lat_df = pd.read_csv(LAT_LOG)
    sys_df = pd.read_csv(SYSTEM_LOG)

    # 1️⃣ Latency bar chart
    plt.figure(figsize=(6,4))
    grouped = lat_df.groupby("phase")["latency_s"].mean()
    plt.bar(grouped.index, grouped.values)
    plt.ylabel("Latency (s)")
    plt.title("MeetAI Latency per Phase")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "latency_chart.png")
    plt.show()

    # 2️⃣ CPU/RAM over time
    plt.figure(figsize=(8,5))
    plt.plot(sys_df["time"], sys_df["cpu%"], label="CPU %")
    plt.plot(sys_df["time"], sys_df["ram%"], label="RAM %")
    plt.xlabel("Time (s)")
    plt.ylabel("Usage (%)")
    plt.title("System Resource Usage")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "system_usage.png")
    plt.show()

    print("\nCharts saved under:", RESULTS_DIR)


# ---------------- MAIN ---------------- #
if __name__ == "__main__":
    asyncio.run(run_benchmark())
    visualize_results()
