# Run the full BIRD dev matrix independently on RunPod

The existing vLLM server stays running. The application gets its own Linux
virtual environment; do not install these requirements into vLLM's environment.
Use persistent storage mounted at `/workspace`. A network volume survives Pod
termination; a Pod's local volume does not. Container storage is unsuitable for
the only copy of checkpoints. Budget space for two transfer archives, extracted
dev databases, the application environment, and results; `df -h /workspace`
shows available space. Training databases are not required for this evaluation.

## 1. Windows PowerShell: package and upload

Use the Pod Connect panel's **SSH over exposed TCP** hostname/IP and port.
The SSH gateway command is not an SCP endpoint. Use the private key already
authorized for this Pod. Replace the first two values and, if needed, key path.
The SSH private key stays on your PC. Do not disable SSH host-key verification.

```powershell
Set-Location 'C:\Users\Vikrant Gandotra\OneDrive\Desktop\mars_sql_project\mars_ablation'
$podAddress = 'REPLACE_WITH_SSH_TCP_HOST_OR_IP'
$podSshPort = 12345 # replace with the external SSH TCP port
$podKeyPath = Join-Path $env:USERPROFILE '.ssh\id_ed25519'
$codeArchive = Join-Path $env:TEMP 'mars-runpod-code.tar.gz'
$devArchive = Join-Path $env:TEMP 'mars-runpod-dev.tar.gz'
$hashFile = Join-Path $env:TEMP 'mars-transfer.sha256'

tar --exclude=./venv --exclude=./env --exclude=./.venv --exclude=./data --exclude=./results --exclude=./artifacts --exclude=./.git --exclude=./.dspy_cache --exclude=__pycache__ --exclude=./.env* --exclude=./.ssh --exclude=./.aws --exclude=./.codex --exclude=./.agents -czf $codeArchive .
if ($LASTEXITCODE -ne 0) { throw 'Code archive failed' }
tar -czf $devArchive data/dev.json data/dev_databases
if ($LASTEXITCODE -ne 0) { throw 'Development data archive failed' }

$codeHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $codeArchive).Hash.ToLowerInvariant()
$devHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $devArchive).Hash.ToLowerInvariant()
@("$codeHash  mars-runpod-code.tar.gz", "$devHash  mars-runpod-dev.tar.gz") | Set-Content -Encoding ascii -LiteralPath $hashFile

ssh -i $podKeyPath -p $podSshPort "root@$podAddress" 'mkdir -m 700 -p /workspace/mars_transfer_01'
if ($LASTEXITCODE -ne 0) { throw 'Remote directory creation failed' }
scp -i $podKeyPath -P $podSshPort $codeArchive $devArchive $hashFile "root@${podAddress}:/workspace/mars_transfer_01/"
if ($LASTEXITCODE -ne 0) { throw 'Archive upload failed' }
scp -i $podKeyPath -P $podSshPort .env "root@${podAddress}:/workspace/mars_transfer_01/.env"
if ($LASTEXITCODE -ne 0) { throw 'Environment upload failed' }
ssh -i $podKeyPath -p $podSshPort "root@$podAddress" 'chmod 600 /workspace/mars_transfer_01/.env'
ssh -i $podKeyPath -p $podSshPort "root@$podAddress"
```

Archive creation/upload can take time. Wait for each block to finish. The code
archive excludes credentials, Windows environments, training data, and old
outputs. Credentials transfer separately inside the encrypted SSH connection
into a private directory. The next commands run **inside the Linux SSH shell**.

## 2. Linux: extract into a new persistent project directory

Only do this extraction on first migration. Do not re-extract code over an
active or checkpointed production run. If the target already exists, inspect
it first or choose a fresh deployment directory consistently below.

```bash
set -e
umask 077
df -h /workspace
cd /workspace/mars_transfer_01
sha256sum -c mars-transfer.sha256
mkdir /workspace/mars_ablation_prod_01
tar -xzf mars-runpod-code.tar.gz -C /workspace/mars_ablation_prod_01
tar -xzf mars-runpod-dev.tar.gz -C /workspace/mars_ablation_prod_01
install -m 600 .env /workspace/mars_ablation_prod_01/.env
cd /workspace/mars_ablation_prod_01
sed -i 's/\r$//' run_full_ablation.sh
bash -n run_full_ablation.sh

apt-get update
apt-get install -y tmux python3-venv util-linux
python3 -c "import sys; assert sys.version_info >= (3,11), 'Python 3.11+ is required'"
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-runpod.txt
.venv/bin/python -m unittest discover -s tests
bash run_full_ablation.sh --dry-run results/full_dev_vllm_01
```

If the container's Python is below 3.11, install/use a compatible Python before
creating `.venv`. Keep the existing vLLM environment and server unchanged. The
copied `.env` keeps your working HTTPS model endpoint and dynamic authentication.
No traffic is routed through the local Windows computer.

## 3. Configure phone notifications (ntfy)

Install the ntfy app on your phone, or use its web app. Subscribe to a long,
unguessable topic. Public-topic names act like passwords: anyone knowing the
topic can subscribe or post. For access control, use a protected topic/account
or a self-hosted ntfy server; an access token is supported below.

Set the HTTPS topic URL without echoing it or any API key:

```bash
.venv/bin/python - <<'PY'
from getpass import getpass
from dotenv import set_key
set_key('.env', 'MARS_NOTIFY_URL', getpass('HTTPS ntfy topic URL: ').strip())
token = getpass('ntfy access token (Enter if not required): ').strip()
set_key('.env', 'MARS_NOTIFY_TOKEN', token)
PY
chmod 600 .env
```

The runner sends separate **MARS ablation FAILED** and **MARS ablation COMPLETED**
alerts. Messages contain status, configuration name, exit code and UTC time;
they do not contain SQL, database samples, endpoints or inference credentials.
Test both notifications, labelled `notification_test`, before launching:

```bash
.venv/bin/python -m src.matrix_status --root results/notification_test --status failed --configuration notification_test --exit-code 1
.venv/bin/python -m src.matrix_status --root results/notification_test --status completed --configuration notification_test
```

Wait for both test alerts on your phone. Notification failures are logged without
exposing credentials and never discard experiment results. Without notification
configuration you still have console logs and local status files, but no push.

## 4. Optional cloud micro-test, then launch detached

One small call verifies this new Linux environment before the multi-day run.
This is a new output directory, not a resume of Windows checkpoints.

```bash
.venv/bin/python run_ablation.py --dev-json data/dev.json --db-dir data/dev_databases --limit 3 --selection self_consistency --prompt-mode uncompiled_dspy --output results/cloud_micro_01/predict.json
tmux new-session -d -s mars_full -c /workspace/mars_ablation_prod_01 'bash run_full_ablation.sh results/full_dev_vllm_01'
tmux attach-session -t mars_full
```

Detach with **Ctrl+B**, release, then **D**. It is then safe to close SSH and
turn off your PC. Do not press Ctrl+C in the tmux session unless intentionally
interrupting the run. The matrix is sequential:

| Prompt mode | Selection | Candidates | Generator turns |
|---|---|---:|---:|
| uncompiled / compiled | self_consistency | 8 | 5 |
| uncompiled / compiled | dspy_verifier | 8 | 5 |
| uncompiled / compiled | self_consistency_verifier | 8 | 5 |
| uncompiled / compiled | greedy (one-candidate self_consistency) | 1 | 1 |

Every command sets `--limit 1534`. Greedy uses temperature 0 and top-p 1;
the other modes use temperature 0.8 and top-p 0.7, eight workers, seed 0.
The runner does not change the six/eight-condition scientific protocol.

## 5. Monitor, recover, and download

After reconnecting with SSH:

```bash
cd /workspace/mars_ablation_prod_01
cat results/full_dev_vllm_01/STATUS.json
tail -n 30 results/full_dev_vllm_01/matrix.log
tmux list-sessions
```

`STATUS.json` tracks the current phase. `FAILED.json` signals a failed attempt;
`COMPLETED.json` is written only after all eight reports passed validation.
Each configuration has a separate `run.log`, predictions, metrics and durable
question journal. A nonzero subprocess exit or report-validation failure stops
the matrix and sends a failure alert. Individual SQL-generation failures are
valid EX outcomes and remain recorded; they do not necessarily stop the batch.

tmux survives SSH disconnects, **not** Pod shutdown/reboot or termination.
A whole-container outage/SIGKILL cannot send an in-container alert; use an
external uptime/heartbeat monitor if that alert is required. Persistent storage
protects completed question records independently of tmux.

If the runner failed, inspect its log before restarting. With the SAME code,
environment, compiled artifacts, dataset, endpoint, and deployment/output paths,
rerun the SAME script/root; it automatically adds `--resume` where journals exist:

```bash
tmux new-session -d -s mars_resume -c /workspace/mars_ablation_prod_01 'bash run_full_ablation.sh results/full_dev_vllm_01'
```

Existing completed questions, including failures, are skipped. If source or
configuration changes are needed, use a fresh run root rather than mixing
protocols. After a reboot, restart vLLM with the same model/context/authentication
configuration before resuming the runner. Ensure the old runner has exited;
the matrix lock also prevents duplicate writers.

After completion, package the results on Linux:

```bash
cd /workspace/mars_ablation_prod_01
tar -czf /workspace/full_dev_vllm_01_results.tar.gz results/full_dev_vllm_01
sha256sum /workspace/full_dev_vllm_01_results.tar.gz
```

Back in Windows PowerShell (reuse your SSH variables):

```powershell
scp -i $podKeyPath -P $podSshPort "root@${podAddress}:/workspace/full_dev_vllm_01_results.tar.gz" .
Get-FileHash -Algorithm SHA256 .\full_dev_vllm_01_results.tar.gz
```

Compare the checksums and retain an off-Pod backup before terminating the Pod.
