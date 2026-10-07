#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p third_party

fetch() {
    local folder="$1" url="$2" commit="$3"
    if [[ ! -d "third_party/$folder/.git" ]]; then
        git clone --filter=blob:none "$url" "third_party/$folder"
    fi
    git -C "third_party/$folder" fetch --depth 1 origin "$commit"
    git -C "third_party/$folder" checkout --detach "$commit"
    test "$(git -C "third_party/$folder" rev-parse HEAD)" = "$commit"
}

fetch SuperPointPretrainedNetwork https://github.com/magicleap/SuperPointPretrainedNetwork.git 1fda796addba9b6f8e79d586a3699700a86b1cea
fetch Invertible-ISP https://github.com/yzxing87/Invertible-ISP.git 344dd333dd2a075f6a9e4ffc445dc387ca3014c4
fetch ELD https://github.com/Vandermode/ELD.git 53d5408dec3a33e5ca304bb49682d7b6b6dc79c2

printf '%s\n' \
  '52b6708629640ca883673b5d5c097c4ddad37d8048b33f09c8ca0d69db12c40e  third_party/SuperPointPretrainedNetwork/superpoint_v1.pth' \
  '483c92ea7f3f3e05e4622b0b7fc775aaf92545362773cb97c10970bd5dff1889  third_party/Invertible-ISP/pretrained/canon.pth' \
  '3f74dd0f60a44b889b8055a172517fe55ea47c14fdec9df671f39a0c32c833a8  third_party/ELD/camera_params/release/CanonEOS5D4_params.npy' | sha256sum --check
