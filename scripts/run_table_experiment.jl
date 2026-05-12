#!/usr/bin/env julia

using Printf
using Statistics

const ROOT = normpath(joinpath(@__DIR__, ".."))
const OUTDIR = joinpath(ROOT, "results", "table_experiment")
const BENCH = joinpath(ROOT, "scripts", "run_qubo_benchmark.jl")

const T = 2000
const REPS = 8
const RESTARTS = 16
const ROUND_RESTARTS = 64
const SYSTEM = "random"
const SEED = 1

struct ExperimentCase
    label::String
    method::String
    rank::Union{Nothing,Int}
    reps::Int
end

const CASES = ExperimentCase[
    ExperimentCase("SDP+GW", "sdp_gw", nothing, REPS),
    ExperimentCase("Sign-Iteration", "sign_iteration", nothing, REPS),
    ExperimentCase("Box-PGA", "box_pgd", nothing, REPS),
    ExperimentCase("LR-SDP r=2", "low_rank", 2, REPS),
    ExperimentCase("LR-SDP r=4", "low_rank", 4, REPS),
    ExperimentCase("LR-SDP r=6", "low_rank", 6, REPS),
    ExperimentCase("LR-SDP r=8", "low_rank", 8, REPS),
]

function parse_args(args)
    parsed = Set{String}()
    for arg in args
        push!(parsed, arg)
    end
    return parsed
end

function csv_split(line::AbstractString)
    cells = String[]
    io = IOBuffer()
    inquote = false
    i = firstindex(line)
    while i <= lastindex(line)
        c = line[i]
        if c == '"'
            if inquote && i < lastindex(line) && line[nextind(line, i)] == '"'
                write(io, '"')
                i = nextind(line, i)
            else
                inquote = !inquote
            end
        elseif c == ',' && !inquote
            push!(cells, String(take!(io)))
        else
            write(io, c)
        end
        i = nextind(line, i)
    end
    push!(cells, String(take!(io)))
    return cells
end

function read_single_row_csv(path::AbstractString)
    lines = readlines(path)
    length(lines) >= 2 || error("Expected one data row in $(path)")
    header = csv_split(lines[1])
    values = csv_split(lines[2])
    return Dict(header[i] => get(values, i, "") for i in eachindex(header))
end

function csv_cell(x)
    s = string(x)
    if occursin(",", s) || occursin("\"", s) || occursin("\n", s)
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

function result_path(case::ExperimentCase, rep::Int)
    rank_part = case.rank === nothing ? "" : "_r$(case.rank)"
    return joinpath(OUTDIR, "$(case.method)$(rank_part)_rep$(rep).csv")
end

function run_case(case::ExperimentCase, rep::Int; force::Bool=false)
    path = result_path(case, rep)
    if isfile(path) && !force
        @printf("skip existing %-16s rep=%d -> %s\n", case.label, rep, path)
        return path
    end

    cmd = `julia --project=$ROOT $BENCH --system $SYSTEM --Ts $T --reps 1 --seed $(SEED + rep - 1) --methods $(case.method) --round-restarts $ROUND_RESTARTS --no-warmup --no-plots --output $path`
    if case.method == "sign_iteration"
        cmd = `$cmd --sign-restarts $RESTARTS`
    elseif case.method == "box_pgd"
        cmd = `$cmd --box-restarts $RESTARTS`
    elseif case.method == "low_rank"
        cmd = `$cmd --ranks $(case.rank) --low-rank-restarts $RESTARTS`
    end

    @printf("run %-16s rep=%d\n", case.label, rep)
    run(cmd)
    return path
end

function mean_std(xs)
    vals = Float64.(xs)
    if length(vals) == 1
        return vals[1], NaN
    end
    return mean(vals), std(vals)
end

function fmt_num(x; digits::Int=3)
    isnan(x) && return ""
    return @sprintf("%.*f", digits, x)
end

function fmt_mean_std(mu, sd; digits::Int=3)
    isnan(sd) && return fmt_num(mu; digits=digits)
    return "$(fmt_num(mu; digits=digits)) +/- $(fmt_num(sd; digits=digits))"
end

function write_summary(rows)
    summary_csv = joinpath(OUTDIR, "summary.csv")
    summary_md = joinpath(OUTDIR, "summary.md")
    columns = [
        "method",
        "max value",
        "quality",
        "time(s)",
        "peak memory (MB)",
        "runs",
    ]

    open(summary_csv, "w") do io
        println(io, join(columns, ","))
        for row in rows
            println(io, join(csv_cell(get(row, col, "")) for col in columns))
        end
    end

    open(summary_md, "w") do io
        println(io, "| method | max value | quality | time(s) | peak memory (MB) |")
        println(io, "|---|---:|---:|---:|---:|")
        for row in rows
            println(io, "| $(row["method"]) | $(row["max value"]) | $(row["quality"]) | $(row["time(s)"]) | $(row["peak memory (MB)"]) |")
        end
    end
    return summary_csv, summary_md
end

function main(args)
    flags = parse_args(args)
    force = "--force" in flags
    mkpath(OUTDIR)

    for case in CASES
        for rep in 1:case.reps
            run_case(case, rep; force=force)
        end
    end

    baseline_rows = [read_single_row_csv(result_path(CASES[1], rep))
                     for rep in 1:CASES[1].reps]
    baseline_values = [parse(Float64, row["binary_objective"])
                       for row in baseline_rows]

    rows = Dict{String,String}[]
    for case in CASES
        case_rows = [read_single_row_csv(result_path(case, rep)) for rep in 1:case.reps]
        values = [parse(Float64, row["binary_objective"]) for row in case_rows]
        qualities = [values[i] / baseline_values[i] for i in eachindex(values)]
        times = [parse(Float64, row["solve_seconds"]) for row in case_rows]
        memories = [parse(Float64, row["maxrss_mb"]) for row in case_rows]

        value_mu, value_sd = mean_std(values)
        quality_mu, quality_sd = mean_std(qualities)
        time_mu, time_sd = mean_std(times)
        memory_mu, memory_sd = mean_std(memories)

        push!(rows, Dict(
            "method" => case.label,
            "max value" => fmt_mean_std(value_mu, value_sd; digits=3),
            "quality" => fmt_mean_std(quality_mu, quality_sd; digits=4),
            "time(s)" => fmt_mean_std(time_mu, time_sd; digits=3),
            "peak memory (MB)" => fmt_mean_std(memory_mu, memory_sd; digits=1),
            "runs" => string(case.reps),
        ))
    end

    summary_csv, summary_md = write_summary(rows)
    @printf("wrote %s\n", summary_csv)
    @printf("wrote %s\n", summary_md)
end

main(ARGS)
