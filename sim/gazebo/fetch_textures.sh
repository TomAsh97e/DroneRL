#!/usr/bin/env bash
# CC0 textures from ambientCG for the Gazebo worlds, kept out of git in sim/gazebo/media/textures.
# File names carry the asset, Ogre tells textures apart by the name alone.
set -euo pipefail

dest=$(cd "$(dirname "$0")" && pwd)/media/textures
assets=(WoodFloor051 PaintedPlaster017 Concrete034 PaintedMetal004 Planks037A Cardboard004 Bricks106A)

for asset in "${assets[@]}"; do
    [[ -f $dest/$asset/${asset}_Color.jpg ]] && continue
    mkdir -p "$dest/$asset"
    zip=$(mktemp)
    curl -fsSL "https://ambientcg.com/get?file=${asset}_1K-JPG.zip" -o "$zip"
    for map in Color NormalGL Roughness; do
        file=$dest/$asset/${asset}_$map.jpg
        unzip -p "$zip" "${asset}_1K-JPG_$map.jpg" > "$file"
        [[ -s $file ]] || { echo "no $map map in $asset" >&2; exit 1; }
    done
    rm "$zip"
    echo "$asset"
done
