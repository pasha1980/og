package main

import "core:fmt"
import "core:os"

import "cmd"

main :: proc() {
    if len(os.args) < 2 {
        cmd.print_help()
        os.exit(0)
    }

    command := os.args[1]
    args := os.args[2:]

    code: int
    switch command {
    case "index":
        code = cmd.run_index(args)
    case "search":
        code = cmd.run_search(args)
    case "context":
        code = cmd.run_context(args)
    case "diff-index":
        code = cmd.run_diff_index(args)
    case "benchmark":
        code = cmd.run_benchmark(args)
    case "help", "-h", "--help":
        cmd.print_help()
        code = 0
    case:
        fmt.eprintln("Unknown command:", command)
        cmd.print_help()
        code = 1
    }

    os.exit(code)
}
