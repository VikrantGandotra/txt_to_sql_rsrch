# Automatic Google Drive backups without touching the active ablation

This installs a SEPARATE backup process. Do not stop the experiment, modify its
source/settings, or restart the Pod. Nothing is backed up until the first
upload/check finishes with `VERIFIED`.

The helper reads `results/`, `artifacts/`, `src/` and selected root-level
compiled prompts, dependency files and runner code. It excludes `.env`, rclone
credentials, database files, virtual environments and lock files. Past outputs
that exist only on Windows require the one-time Windows upload in step 3.

## 1. Windows PowerShell: install rclone

```powershell
winget install --id Rclone.Rclone --exact
```

Close/reopen PowerShell so the command is available, then:

```powershell
rclone version
$driveConfig = Join-Path $env:USERPROFILE '.config\rclone\mars-drive.conf'
New-Item -ItemType Directory -Force -Path (Split-Path $driveConfig) | Out-Null
rclone --config $driveConfig config
```

Answer the interactive prompts:

1. `n`: New remote.
2. Name: `marsdrive`.
3. Storage type: `drive` (Google Drive, not Google Cloud Storage).
4. Client ID: Enter (blank); client secret: Enter (blank).
5. Scope: `drive.file` (files/folders created by rclone only; usually option 3).
6. Service-account file: Enter (blank).
7. Advanced configuration: `n`.
8. Use browser authentication: `y`. Sign in to the account with your 5 TB quota
   and approve rclone. If your organization blocks approval, obtain admin
   approval; do not bypass its policy.
9. Shared Drive: `n` for your personal My Drive (even if it is a school account).
10. Accept/save: `y`; quit: `q`.

Optional fields vary by version; leave other fields at their defaults. The
configuration contains a refresh token: never paste it into chat or commit it.
It will let the Pod refresh authorization without the Windows PC being online.

Verify the connection and create a private backup folder:

```powershell
rclone --config $driveConfig mkdir marsdrive:MARS-SQL-Backups
rclone --config $driveConfig lsd marsdrive:
```

Keep the Google Drive folder private. `drive.file` lets rclone see only files
created using this authorization; do not create/move backup folders manually
in the Drive UI and expect rclone to discover them.

## 2. Transfer the isolated helper and Drive authorization

Set SSH-over-exposed-TCP values from the CURRENT RunPod Connect tab. The values
below were shown earlier; replace them if the console now shows different ones.

```powershell
Set-Location 'C:\Users\Vikrant Gandotra\OneDrive\Desktop\mars_sql_project\mars_ablation'
$podAddress = '213.173.107.89'
$podSshPort = 17940
$podKeyPath = Join-Path $env:USERPROFILE '.ssh\id_ed25519'
$driveConfig = Join-Path $env:USERPROFILE '.config\rclone\mars-drive.conf'

ssh -i $podKeyPath -p $podSshPort "root@$podAddress" 'mkdir -p -m 700 /root/mars-backup'
if ($LASTEXITCODE -ne 0) { throw 'SSH setup failed' }
scp -i $podKeyPath -P $podSshPort $driveConfig .\runpod_results_backup.py "root@${podAddress}:/root/mars-backup/"
if ($LASTEXITCODE -ne 0) { throw 'Backup helper/config upload failed' }
ssh -i $podKeyPath -p $podSshPort "root@$podAddress" 'chmod 600 /root/mars-backup/mars-drive.conf'
```

The SSH private key stays on Windows. The experiment's `.env` is not transferred
or changed by this setup. The copied Google authorization is sensitive and stays
in a private directory outside the experiment and outside backup contents.

## 3. One-time backup of past Windows results

These commands only read the existing local results and copy them to a separate
Drive folder. They do not change the experiment running on RunPod.

```powershell
if (Test-Path .\results) {
    rclone --config $driveConfig copy .\results marsdrive:MARS-SQL-Backups/windows_past/results --checksum --progress
    if ($LASTEXITCODE -ne 0) { throw 'Windows results upload failed' }
    rclone --config $driveConfig check .\results marsdrive:MARS-SQL-Backups/windows_past/results --one-way
    if ($LASTEXITCODE -ne 0) { throw 'Windows results verification failed' }
}
if (Test-Path .\artifacts) {
    rclone --config $driveConfig copy .\artifacts marsdrive:MARS-SQL-Backups/windows_past/artifacts --checksum --progress
    if ($LASTEXITCODE -ne 0) { throw 'Windows artifacts upload failed' }
    rclone --config $driveConfig check .\artifacts marsdrive:MARS-SQL-Backups/windows_past/artifacts --one-way
    if ($LASTEXITCODE -ne 0) { throw 'Windows artifacts verification failed' }
}
```

The Windows outputs should be inactive while copying. Do not upload the entire
project/credential directory or the BIRD databases for this results backup.
You can do this step after the first verified Pod backup if time is urgent.

## 4. RunPod terminal: install backup tools and locate the live project

Open a NEW SSH or web terminal. Do not type into or interrupt the ablation's
active terminal. These packages do not modify the Python/vLLM environment.

```bash
apt-get update
apt-get install -y rclone tmux
rclone version
python3 --version
chmod 700 /root/mars-backup
chmod 600 /root/mars-backup/mars-drive.conf
rclone --config /root/mars-backup/mars-drive.conf lsd marsdrive:
```

The helper needs Python 3.11+. Use your application Python executable if
`python3` is older; no additional Python packages are required.

Locate the actual working directory of the currently running evaluator:

```bash
EVAL_PID=$(pgrep -of '[p]ython.*run_ablation.py')
if [ -n "$EVAL_PID" ]; then
    readlink -f "/proc/$EVAL_PID/cwd"
fi
find /workspace /vllm-workspace -type f -name predict.checkpoint.jsonl 2>/dev/null
```

Use the evaluator's working directory as `PROJECT`. For example, if the files
are under `/workspace/mars_ablation_prod_01/results/...`:

```bash
PROJECT=/workspace/mars_ablation_prod_01
test -f "$PROJECT/run_ablation.py"
find "$PROJECT/results" -type f -name predict.checkpoint.jsonl
```

Replace that project path if the output shows another directory. If the
evaluator is between configurations, its PID may briefly be absent; locate the
project from the checkpoint paths. Do not assume the interactive shell's
`/vllm-workspace` is the project directory.

## 5. Optional phone alerts for backup failures/recovery

If you use ntfy, save your HTTPS topic URL outside the experiment. The same
unpredictable topic you subscribed to for experiment alerts can be used.

```bash
python3 - <<'PY'
from getpass import getpass
from pathlib import Path
path = Path('/root/mars-backup/notify-url')
path.write_text(getpass('HTTPS ntfy topic URL: ').strip())
path.chmod(0o600)
PY
```

The current helper supports an unauthenticated HTTPS ntfy topic URL. For a
protected topic, omit this optional setting until token support is configured;
backup status/logs still work. Do not put an access token in a URL/query string.
Alerts report backup failure/recovery, never inference keys or experiment data.

## 6. Perform and verify the FIRST cloud backup

Run this once in the new RunPod terminal (with `PROJECT` set correctly):

```bash
nice -n 15 python3 /root/mars-backup/runpod_results_backup.py \
  --project "$PROJECT" \
  --config /root/mars-backup/mars-drive.conf \
  --state /root/mars-backup/state \
  --remote marsdrive:MARS-SQL-Backups/runpod_full_01 \
  --notify-file /root/mars-backup/notify-url \
  --once
```

Wait for `VERIFIED`. Then inspect:

```bash
cat /root/mars-backup/state/STATUS.json
rclone --config /root/mars-backup/mars-drive.conf cat marsdrive:MARS-SQL-Backups/runpod_full_01/LAST_SUCCESS.json
```

Also open Drive in your browser. Under `MARS-SQL-Backups/runpod_full_01` you will
see `slot-0.tar.gz`, its manifest, and `LAST_SUCCESS.json`. The second successful
backup creates `slot-1.tar.gz`. Do not rely on backup protection before this
first verified upload. On any error, leave the ablation running and inspect the
backup error separately.

## 7. Start automatic backups independent of Windows

The arguments are saved in a backup-only launch script outside the experiment.
Keep the same project/remote/state paths as the successful first backup.

```bash
printf '#!/usr/bin/env bash\nexec nice -n 15 python3 /root/mars-backup/runpod_results_backup.py --project %q --config /root/mars-backup/mars-drive.conf --state /root/mars-backup/state --remote marsdrive:MARS-SQL-Backups/runpod_full_01 --notify-file /root/mars-backup/notify-url --interval 900 >> /root/mars-backup/backup.log 2>&1\n' "$PROJECT" > /root/mars-backup/start-backup.sh
chmod 700 /root/mars-backup/start-backup.sh
tmux new-session -d -s mars_backup 'bash /root/mars-backup/start-backup.sh'
tmux list-sessions
tail -n 10 /root/mars-backup/backup.log
```

The helper starts immediately and targets a new backup every 900 seconds.
Each backup must finish before another begins; very large uploads extend that
interval. It retries failed cycles and never takes the ablation's locks. A
separate lock prevents two backup processes using the same state directory.
Your PC can now disconnect/shut down. Do not start a second backup process with
a different state directory targeting the SAME Drive folder.

Periodically check the UTC timestamp in Drive's `LAST_SUCCESS.json` and local
`STATUS.json`. A `running` tmux session alone is not evidence of a recent backup.

## What is guaranteed by each VERIFIED cycle?

* Only complete, checksummed checkpoint records are included. An incomplete
  trailing line is omitted from the COPY; the live checkpoint is untouched.
* Growing logs are copied to their initial byte length. Atomic final JSON files
  are captured as they exist during this cycle. The snapshot spans a time
  interval; it is not an instantaneous transaction across all eight runs.
* The snapshot is archived, SHA-256 identified, uploaded and checked using
  rclone's supported remote hashes. Only then is `LAST_SUCCESS.json` published.
* Two alternating recovery archives bound current backup size. The archive
  referenced by `LAST_SUCCESS.json` is not overwritten by the next attempt.
* `copy` is used, not `sync`/`move`. No experiment or unrelated Drive files are
  deleted. Old experiment results remain within each current cumulative snapshot;
  Drive can additionally keep file revisions under its own policy.

The estimated final current backup footprint is approximately twice the
compressed snapshot size, not a new full archive every 15 minutes. Staging needs
local free space for one uncompressed snapshot plus its archive. The snapshots
contain the six telemetry dimensions because they include the full journals,
final reports, SQL artifacts, logs and compiled-prompt provenance.

No process can back up work that has not been committed yet. Exposure is the
progress since the LAST VERIFIED upload, not a promised maximum of 15 minutes.
tmux survives SSH disconnects, not Pod shutdowns. If the Pod dies, the backups
already in Drive survive; the backup process cannot send an alert while dead.

## 8. After the ablation finishes

First request a clean stop of ONLY the backup process (not the experiment/server).
It finishes any current upload before exiting. Then do one final `--once` cycle
with the same command from step 6. This captures final metrics and completion
markers. The helper exits after each once-cycle.

```bash
touch /root/mars-backup/state/STOP
while tmux has-session -t mars_backup 2>/dev/null; do sleep 5; done
```

Run step 6 again after the backup session exits. Check the
new verified UTC time and question counts. Each completed configuration should
have 1,534 checkpoint records. Keep the Pod until that final backup is verified.

To restart the periodic backup later, remove only its stop marker and repeat
the tmux launch command in step 7:

```bash
python3 -c "from pathlib import Path; Path('/root/mars-backup/state/STOP').unlink(missing_ok=True)"
```

To download a recovery copy on Windows:

```powershell
$driveConfig = Join-Path $env:USERPROFILE '.config\rclone\mars-drive.conf'
$metadata = (rclone --config $driveConfig cat marsdrive:MARS-SQL-Backups/runpod_full_01/LAST_SUCCESS.json | ConvertFrom-Json)
$archiveName = $metadata.archive
rclone --config $driveConfig copyto "marsdrive:MARS-SQL-Backups/runpod_full_01/$archiveName" ".\$archiveName"
if ($LASTEXITCODE -ne 0) { throw 'Recovery download failed' }
if ((Get-FileHash -Algorithm SHA256 -LiteralPath ".\$archiveName").Hash.ToLowerInvariant() -ne $metadata.archive_sha256) { throw 'Recovery archive checksum mismatch' }
New-Item -ItemType Directory -Path .\recovered_mars_results
tar -xzf ".\$archiveName" -C .\recovered_mars_results
```

Restore into a NEW folder. Remote Linux checkpoints contain strict absolute-path,
dependency and source controls; downloading protects analysis/results but does
not automatically make them resumable on a different machine/path.
