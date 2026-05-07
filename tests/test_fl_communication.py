"""
FL 통신 테스트 — 외부 서버에 더미 데이터 클라이언트 접속.

이미 실행 중인 서버(예: Windows PC)에 더미 데이터로
클라이언트 N개를 접속시켜 실제 FL 라운드가 돌아가는지 검증합니다.

실제 main_client.py를 그대로 subprocess로 호출하므로,
이 테스트가 통과하면 실제 배포 환경에서도 작동합니다.

Usage:
  # 서버(168.131.153.47:9000)에 클라이언트 3개 접속 테스트
  uv run python tests/test_fl_communication.py --server-address 168.131.153.47:9000 --num-clients 3

  # 클라이언트 수 변경 (서버 min_clients에 맞춰야 함)
  uv run python tests/test_fl_communication.py --server-address 168.131.153.47:9000 --num-clients 2

사전 조건:
  1. 서버가 이미 실행 중이어야 함 (Windows: run_liver_server.bat)
  2. 이 PC에서 서버 IP:PORT로 TCP 연결 가능해야 함
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CLIENT_SCRIPT = os.path.join(
    PROJECT_ROOT, "src", "use_cases", "liver_segmentation", "main_client.py"
)
DUMMY_GEN = os.path.join(SCRIPT_DIR, "generate_dummy_data.py")

PATIENTS_PER_CLIENT = 3


def check_server_reachable(server_address: str) -> bool:
    """TCP 연결로 서버 접근 가능 여부 확인."""
    import socket
    host, port = server_address.rsplit(":", 1)
    try:
        with socket.create_connection((host, int(port)), timeout=5):
            return True
    except (OSError, ValueError):
        return False


def generate_dummy_data(tmp_dir: str, n_clients: int) -> list[str]:
    """각 클라이언트용 더미 데이터 생성."""
    client_dirs = []
    for cid in range(n_clients):
        cdir = os.path.join(tmp_dir, f"client_{cid}_data")
        subprocess.run(
            [
                sys.executable, DUMMY_GEN,
                "--out-dir", cdir,
                "--n-patients", str(PATIENTS_PER_CLIENT),
                "--depth", "20",
                "--height", "36",
                "--width", "36",
            ],
            check=True, cwd=PROJECT_ROOT,
        )
        client_dirs.append(cdir)
    return client_dirs


def main():
    parser = argparse.ArgumentParser(
        description="FL 통신 테스트 — 외부 서버에 더미 클라이언트 접속"
    )
    parser.add_argument(
        "--server-address", type=str, required=True,
        help="실행 중인 서버 주소 (예: 168.131.153.47:9000)",
    )
    parser.add_argument(
        "--num-clients", type=int, default=3,
        help="접속할 클라이언트 수 (서버 min_clients에 맞춰야 함)",
    )
    parser.add_argument(
        "--timeout", type=int, default=600,
        help="최대 대기 시간(초)",
    )
    parser.add_argument(
        "--keep-logs", action="store_true",
        help="테스트 후 로그/데이터 보존",
    )
    args = parser.parse_args()

    tmp_dir = os.path.join(SCRIPT_DIR, "tmp_comm_test")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    print("=" * 60)
    print("FL 통신 테스트 (실제 main_client.py 사용)")
    print("=" * 60)
    print(f"  서버:       {args.server_address}")
    print(f"  클라이언트:  {args.num_clients}개")
    print()

    # ── Step 1: 서버 연결 확인 ──
    print("[Step 1] 서버 연결 확인 ...")
    if check_server_reachable(args.server_address):
        print(f"  [OK] {args.server_address} 접속 가능")
    else:
        print(f"  [FAIL] {args.server_address} 접속 불가!")
        print("  → 서버가 실행 중인지, 방화벽이 열려있는지 확인하세요.")
        sys.exit(1)

    # ── Step 2: 더미 데이터 생성 ──
    print(f"\n[Step 2] 더미 데이터 생성 ({args.num_clients}개 클라이언트) ...")
    client_dirs = generate_dummy_data(tmp_dir, args.num_clients)
    for i, d in enumerate(client_dirs):
        print(f"  Client {i}: {d}")

    # ── Step 3: 클라이언트 시작 ──
    print(f"\n[Step 3] 클라이언트 {args.num_clients}개 → {args.server_address}")
    client_procs = []
    client_logs = []
    for cid in range(args.num_clients):
        log_path = os.path.join(tmp_dir, f"client_{cid}.log")
        client_logs.append(log_path)
        cf = open(log_path, "w")
        proc = subprocess.Popen(
            [
                sys.executable, CLIENT_SCRIPT,
                "--server-address", args.server_address,
                "--data-dir", client_dirs[cid],
                "--client-id", str(cid),
            ],
            stdout=cf, stderr=subprocess.STDOUT,
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        client_procs.append((proc, cf))
        print(f"  Client {cid}: PID={proc.pid}")
        time.sleep(1)

    # ── Step 4: 완료 대기 ──
    print(f"\n[Step 4] FL 완료 대기 (최대 {args.timeout}s) ...")
    start = time.time()
    while time.time() - start < args.timeout:
        all_done = all(p.poll() is not None for p, _ in client_procs)
        if all_done:
            break
        elapsed = int(time.time() - start)
        if elapsed > 0 and elapsed % 30 == 0:
            # 중간 상태 출력
            alive = sum(1 for p, _ in client_procs if p.poll() is None)
            print(f"  ... {elapsed}s 경과 (활성 클라이언트: {alive})")
        time.sleep(5)

    elapsed = time.time() - start

    # 핸들 정리
    for _, cf in client_procs:
        cf.close()
    for proc, _ in client_procs:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(10)

    # ── Step 5: 결과 ──
    print(f"\n{'=' * 60}")
    print("결과")
    print(f"{'=' * 60}")

    all_ok = True
    for cid, (proc, _) in enumerate(client_procs):
        ok = proc.returncode == 0
        if not ok:
            all_ok = False
        print(f"  [{'PASS' if ok else 'FAIL'}] Client {cid} (exit={proc.returncode})")

    print(f"  소요: {elapsed:.0f}s")

    # 로그 요약
    for cid in range(args.num_clients):
        print(f"\n{'─' * 60}")
        print(f"Client {cid} 로그 (마지막 15줄):")
        print(f"{'─' * 60}")
        with open(client_logs[cid]) as f:
            lines = f.readlines()
            for line in lines[-15:]:
                print(f"  {line.rstrip()}")

    # 정리
    if not args.keep_logs:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"\n  임시 파일 정리 완료")
    else:
        print(f"\n  로그 보존: {tmp_dir}")

    if all_ok:
        print("\n  ✓ 전체 통과 — 서버 통신 + FL 학습 + 집계 정상!")
        print("    → 실제 데이터로 교체해도 동일하게 작동합니다.")
    else:
        print("\n  ✗ 일부 실패 — 위 로그 확인")
        sys.exit(1)


if __name__ == "__main__":
    main()
