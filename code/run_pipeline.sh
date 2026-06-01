#!/bin/bash
set -e # Stop on errors

# Ask R to check for the package. If missing, R will exit with status code 1.
if Rscript -e 'if (!requireNamespace("lixoftConnectors", quietly = TRUE)) quit(status=1)' >/dev/null 2>&1; then
    echo "✓ lixoftConnectors is installed and ready."
else
   MONOLIX_KEY="6361-2444-1485-2969"
    echo "Installing Monolix..."
    # 1. Copy the installer from the read-only /data folder to /tmp
    cp /data/monolixSuite2024R1 /tmp/monolixSuite2024R1

    # 2. Make the copy executable
    chmod +x /tmp/monolixSuite2024R1

    # 3. Run the binary directly (Notice 'bash' is removed from the front)
    /tmp/monolixSuite2024R1 in --am --al --c -t /opt/MonolixSuite

    echo "Installing lixoftConnectors..."
    # 4. Install the local R package
    Rscript -e "install.packages('/opt/MonolixSuite/MonolixSuite2024R1/connectors/lixoftConnectors.tar.gz', repos=NULL, type='source')"

    /opt/MonolixSuite/MonolixSuite2024R1/lib/licenseActivate --key $MONOLIX_KEY

    echo "--- MISSING LIBRARIES LIST ---"
    ldd /opt/MonolixSuite/MonolixSuite2024R1/lib/liblixoftConnectors.so | grep "not found" || true
    echo "------------------------------"
fi

echo "Running analysis..."
bash /code/run_everything.sh
