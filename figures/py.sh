#!/usr/bin/env bash
# python with numpy/matplotlib/scipy from nixpkgs
exec nix shell --impure --expr 'with import <nixpkgs> {}; python3.withPackages (p: [p.numpy p.matplotlib p.scipy])' --command python3 "$@" 2> >(grep -v "search path entry" >&2)
