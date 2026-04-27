#!/bin/bash
# ══════════════════════════════════════════════════════════════
#  FedMorph Liver Segmentation — Start Client
# ══════════════════════════════════════════════════════════════
#
#  Usage:
#    ./run_liver_client.sh CLIENT_ID [SERVER_ADDRESS]
#
#  Examples:
#    ./run_liver_client.sh 0
#    ./run_liver_client.sh 0 192.168.1.100:9000
#    ./run_liver_client.sh 1 192.168.1.100:9000
# ══════════════════════════════════════════════════════════════

USE_CASE="liver_segmentation"

if [ $# -eq 0 ]; then
    echo ""
    echo "  Usage: run_liver_client.sh CLIENT_ID [SERVER_ADDRESS]"
    echo ""
    echo "  CLIENT_ID     : 0, 1, or 2"
    echo "  SERVER_ADDRESS : e.g. 192.168.1.100:9000"
    echo ""
    exit 1
fi

CLIENT_ID=$1
SERVER_ADDR=$2

cd "$(dirname "$0")/.."

echo "============================================================"
echo " FedMorph Liver Segmentation Client $CLIENT_ID"
echo "============================================================"

if [ -z "$SERVER_ADDR" ]; then
    echo "Starting client $CLIENT_ID (server address from config) ..."
    uv run python "src/use_cases/$USE_CASE/main_client.py" --client-id "$CLIENT_ID"
else
    echo "Starting client $CLIENT_ID connecting to $SERVER_ADDR ..."
    uv run python "src/use_cases/$USE_CASE/main_client.py" --client-id "$CLIENT_ID" --server-address "$SERVER_ADDR"
fi
