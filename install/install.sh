#!/usr/bin/env bash

set -e

echoexit() {
    local msg="$1"
    local code="$2"
    echo "$msg" >&2
    exit "$code"
}

downloader() {
    downloader=
    command -v curl > /dev/null && downloader='curl'
    [[ -z "$downloader" ]] && command -v wget > /dev/null && downloader="wget"
    [[ -z "$downloader" ]] && echoexit 'No `curl` or `wget` found locally. Please, install it using your package manager' 1
    echo "$downloader"
}

binary_name() {
    local bin="og"
    case "$OSTYPE" in
        linux-gnu*) bin="${bin}.linux" ;;
        darwin*) bin="${bin}.darwin" ;;
        msys* | cygwin*) bin="${bin}.win" ;;
        freebsd*) bin="${bin}.freebsd" ;;
        *) echoexit "Unknown operating system: $OSTYPE" 126 ;;
    esac

    case "$(uname -m)" in
        x86_64) echo "${bin}_amd64" ;;
        aarch64|arm64) echo "${bin}_arm64" ;;
        armv7l) echo "${bin}_arm32" ;;
        i686|i386) echo "${bin}_amd32" ;;
        riscv64) echo "${bin}_riscv64" ;;
        *) echoexit "Unknown architecture: $(uname -m)" 126;;
    esac
}

curl_download() {
    local bin="$1"
    curl -s https://api.github.com/repos/pasha1980/og/releases/latest \
        | grep "browser_download_url.*${bin}" \
        | cut -d : -f 2,3 \
        | tr -d \" \
        | xargs curl -sLo /tmp/og
}

wget_download() {
    local bin="$1"
    wget -qO - https://api.github.com/repos/pasha1980/og/releases/latest \
        | grep "browser_download_url.*${bin}" \
        | cut -d : -f 2,3 \
        | tr -d \" \
        | wget -qi - -O /tmp/og
}

print_adjust_path() {
    echo
    echo
    echo 'To make OG accessible set'
    echo '      export PATH="${HOME}/.local/bin:${PATH}"'
    echo 'to your init script (~/.bashrc or ~/.config/fish/config.fish)'
    echo
    echo
}

main() {
    local d=${ downloader ;}
    echo "Using $d to download OG" >&2
    # Temporary supports only Linux amd64
    # local bname=${ binary_name ;}
    local bname="og"
    echo "Binary name: $bname" >&2
    ${d}_download "$bname"

    chmod +x /tmp/og
    [[ ! -d ~/.local/bin ]] && mkdir -p ~/.local/bin
    mv /tmp/og ~/.local/bin/og

    echo 'OG installed successfuly to ~/.local/bin/og'
    grep "${HOME}/.local/bin" <<< "$PATH" > /dev/null || print_adjust_path
    export PATH="${HOME}/.local/bin:${PATH}"
    echo 'Try to run `og --help`'
}

main "$@"
