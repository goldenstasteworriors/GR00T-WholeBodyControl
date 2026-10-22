#!/usr/bin/env bash
# One-time installation on the robot; invoke with sudo.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$EUID" == 0 ]] || { echo "Run this installation with sudo" >&2; exit 1; }
STATE="$REPO/.runtime/robot_collection_services"
mkdir -p "$STATE" /etc/systemd/system/sonic_d435i_camera.service.d
DROPIN=/etc/systemd/system/sonic_d435i_camera.service.d/30-sonic-rgb.conf
SUDOERS=/etc/sudoers.d/sonic-collection-services
for file in "$DROPIN" "$SUDOERS"; do
    if [[ -e "$file" ]]; then cp -p "$file" "$STATE/$(basename "$file").$(date +%Y%m%d_%H%M%S).bak"; fi
done
printf '[Service]\nExecStart=\nExecStart=/bin/bash %s/scripts/robot_collection_services.sh camera\n' "$REPO" > "$DROPIN"
printf '%s\n' 'unitree ALL=(root) NOPASSWD: /usr/bin/systemctl start sonic-dex1.service, /usr/bin/systemctl restart sonic_d435i_camera.service' > "$STATE/sudoers"
/usr/sbin/visudo -cf "$STATE/sudoers"
install -m 0440 "$STATE/sudoers" "$SUDOERS"
/usr/bin/systemctl daemon-reload
echo "Installed RGB camera command and permissions for the two collection services"

