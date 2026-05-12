#!/usr/bin/env julia

using Printf
using Statistics

const ROOT = normpath(joinpath(@__DIR__, ".."))
const OUTDIR = joinpath(ROOT, "results", "t_scaling")
const RAWDIR = joinpath(OUTDIR, "raw")
const BENCH = joinpath(ROOT, "scripts", "run_qubo_benchmark.jl")
const TABLE_CACHE = joinpath(ROOT, "results", "table_experiment")

const TS = collect(200:200:2000)
const REPS = 8
const RESTARTS = 16
const ROUND_RESTARTS = 64
const SYSTEM = "random"
const SEED = 1

struct ExperimentCase
    label::String
    method::String
    rank::Union{Nothing,Int}
end

const CASES = ExperimentCase[
    ExperimentCase("SDP+GW", "sdp_gw", nothing),
    ExperimentCase("Sign-Iteration", "sign_iteration", nothing),
    ExperimentCase("Box-PGA", "box_pgd", nothing),
    ExperimentCase("LR-SDP r=2", "low_rank", 2),
    ExperimentCase("LR-SDP r=4", "low_rank", 4),
    ExperimentCase("LR-SDP r=6", "low_rank", 6),
    ExperimentCase("LR-SDP r=8", "low_rank", 8),
]

const COLORS = Dict(
    "SDP+GW" => "#1f77b4",
    "Sign-Iteration" => "#d62728",
    "Box-PGA" => "#2ca02c",
    "LR-SDP r=2" => "#9467bd",
    "LR-SDP r=4" => "#ff7f0e",
    "LR-SDP r=6" => "#17becf",
    "LR-SDP r=8" => "#8c564b",
)

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

function rank_part(case::ExperimentCase)
    return case.rank === nothing ? "" : "_r$(case.rank)"
end

function raw_path(case::ExperimentCase, T::Int, rep::Int)
    return joinpath(RAWDIR, "$(case.method)$(rank_part(case))_T$(T)_rep$(rep).csv")
end

function legacy_table_path(case::ExperimentCase, rep::Int)
    return joinpath(TABLE_CACHE, "$(case.method)$(rank_part(case))_rep$(rep).csv")
end

function run_case(case::ExperimentCase, T::Int, rep::Int)
    path = raw_path(case, T, rep)
    isfile(path) && return path

    legacy = legacy_table_path(case, rep)
    if T == 2000 && isfile(legacy)
        cp(legacy, path; force=true)
        @printf("reuse %-16s T=%4d rep=%d\n", case.label, T, rep)
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

    @printf("run %-16s T=%4d rep=%d\n", case.label, T, rep)
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

function xml_escape(s)
    return replace(string(s), "&" => "&amp;", "<" => "&lt;", ">" => "&gt;",
        "\"" => "&quot;")
end

function nice_num(x)
    ax = abs(Float64(x))
    if ax >= 1000
        return @sprintf("%.0f", x)
    elseif ax >= 100
        return @sprintf("%.1f", x)
    elseif ax >= 10
        return @sprintf("%.2f", x)
    else
        return @sprintf("%.3f", x)
    end
end

function make_path(points)
    isempty(points) && return ""
    io = IOBuffer()
    first_point = true
    for (x, y) in points
        print(io, first_point ? "M " : " L ")
        print(io, @sprintf("%.2f %.2f", x, y))
        first_point = false
    end
    return String(take!(io))
end

function write_shaded_svg(path, title, ylabel, rows, value_prefix)
    width = 1080
    height = 640
    left = 88
    right = 230
    top = 58
    bottom = 86
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs_all = TS

    y_lowers = Float64[]
    y_uppers = Float64[]
    for row in rows
        push!(y_lowers, Float64(row["$(value_prefix)_mean"]) -
            Float64(row["$(value_prefix)_std"]))
        push!(y_uppers, Float64(row["$(value_prefix)_mean"]) +
            Float64(row["$(value_prefix)_std"]))
    end
    y0 = min(0.0, minimum(y_lowers))
    y1 = maximum(y_uppers)
    y1 = y1 <= y0 ? y0 + 1.0 : y1
    pad = 0.06 * (y1 - y0)
    y0 -= pad
    y1 += pad

    sx(x) = left + (Float64(x) - minimum(xs_all)) /
        (maximum(xs_all) - minimum(xs_all)) * plot_w
    sy(y) = top + (y1 - Float64(y)) / (y1 - y0) * plot_h

    by_method = Dict(case.label => Dict{Int,Dict{String,Any}}() for case in CASES)
    for row in rows
        by_method[String(row["method"])][Int(row["T"])] = row
    end

    open(path, "w") do io
        println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$width" height="$height" viewBox="0 0 $width $height">""")
        println(io, """<rect width="100%" height="100%" fill="white"/>""")
        println(io, """<text x="$left" y="32" font-family="Arial" font-size="21" font-weight="700">$(xml_escape(title))</text>""")
        println(io, """<line x1="$left" y1="$(top + plot_h)" x2="$(left + plot_w)" y2="$(top + plot_h)" stroke="#333" stroke-width="1.2"/>""")
        println(io, """<line x1="$left" y1="$top" x2="$left" y2="$(top + plot_h)" stroke="#333" stroke-width="1.2"/>""")

        for tick in 0:5
            y = y0 + tick * (y1 - y0) / 5
            py = sy(y)
            println(io, """<line x1="$left" y1="$py" x2="$(left + plot_w)" y2="$py" stroke="#e8e8e8"/>""")
            println(io, """<text x="$(left - 10)" y="$(py + 4)" text-anchor="end" font-family="Arial" font-size="12">$(nice_num(y))</text>""")
        end

        for T in xs_all
            px = sx(T)
            println(io, """<line x1="$px" y1="$(top + plot_h)" x2="$px" y2="$(top + plot_h + 5)" stroke="#333"/>""")
            println(io, """<text x="$px" y="$(top + plot_h + 24)" text-anchor="middle" font-family="Arial" font-size="12">$T</text>""")
        end

        for case in CASES
            method_rows = [by_method[case.label][T] for T in xs_all
                           if haskey(by_method[case.label], T)]
            isempty(method_rows) && continue
            color = COLORS[case.label]
            upper = [(sx(row["T"]), sy(Float64(row["$(value_prefix)_mean"]) +
                Float64(row["$(value_prefix)_std"]))) for row in method_rows]
            lower = reverse([(sx(row["T"]), sy(Float64(row["$(value_prefix)_mean"]) -
                Float64(row["$(value_prefix)_std"]))) for row in method_rows])
            polygon = vcat(upper, lower)
            println(io, """<path d="$(make_path(polygon)) Z" fill="$color" opacity="0.16" stroke="none"/>""")

            line_points = [(sx(row["T"]), sy(row["$(value_prefix)_mean"]))
                           for row in method_rows]
            println(io, """<path d="$(make_path(line_points))" fill="none" stroke="$color" stroke-width="2.4"/>""")
            for (px, py) in line_points
                println(io, """<circle cx="$px" cy="$py" r="3.2" fill="$color"/>""")
            end
        end

        println(io, """<text x="$(left + plot_w / 2)" y="$(height - 25)" text-anchor="middle" font-family="Arial" font-size="15">T</text>""")
        println(io, """<text transform="translate(24,$(top + plot_h / 2)) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="15">$(xml_escape(ylabel))</text>""")

        for (idx, case) in enumerate(CASES)
            color = COLORS[case.label]
            y = top + 24 * (idx - 1)
            x = left + plot_w + 32
            println(io, """<rect x="$x" y="$(y - 10)" width="18" height="12" fill="$color" opacity="0.75"/>""")
            println(io, """<line x1="$x" y1="$(y - 4)" x2="$(x + 18)" y2="$(y - 4)" stroke="$color" stroke-width="2.4"/>""")
            println(io, """<text x="$(x + 28)" y="$(y + 1)" font-family="Arial" font-size="12">$(xml_escape(case.label))</text>""")
        end

        println(io, "</svg>")
    end
    return path
end

function write_summary(rows)
    summary_path = joinpath(OUTDIR, "summary.csv")
    columns = [
        "T", "method", "rank", "runs",
        "max_value_mean", "max_value_std",
        "time_seconds_mean", "time_seconds_std",
        "peak_memory_mb_mean", "peak_memory_mb_std",
    ]
    open(summary_path, "w") do io
        println(io, join(columns, ","))
        for row in rows
            println(io, join((csv_cell(get(row, col, "")) for col in columns), ","))
        end
    end
    return summary_path
end

function aggregate_rows()
    rows = Dict{String,Any}[]
    for T in TS
        for case in CASES
            raw_rows = [read_single_row_csv(raw_path(case, T, rep))
                        for rep in 1:REPS]
            values = [parse(Float64, row["binary_objective"]) for row in raw_rows]
            times = [parse(Float64, row["solve_seconds"]) for row in raw_rows]
            memories = [parse(Float64, row["maxrss_mb"]) for row in raw_rows]
            value_mu, value_sd = mean_std(values)
            time_mu, time_sd = mean_std(times)
            memory_mu, memory_sd = mean_std(memories)
            push!(rows, Dict{String,Any}(
                "T" => T,
                "method" => case.label,
                "rank" => case.rank === nothing ? "" : case.rank,
                "runs" => REPS,
                "max_value_mean" => value_mu,
                "max_value_std" => value_sd,
                "time_seconds_mean" => time_mu,
                "time_seconds_std" => time_sd,
                "peak_memory_mb_mean" => memory_mu,
                "peak_memory_mb_std" => memory_sd,
            ))
        end
    end
    return rows
end

function main()
    mkpath(RAWDIR)
    for T in TS
        for case in CASES
            for rep in 1:REPS
                run_case(case, T, rep)
            end
        end
    end

    rows = aggregate_rows()
    summary_path = write_summary(rows)
    write_shaded_svg(joinpath(OUTDIR, "max_value_vs_T.svg"),
        "Max Value vs T", "max value", rows, "max_value")
    write_shaded_svg(joinpath(OUTDIR, "time_vs_T.svg"),
        "Runtime vs T", "time (s)", rows, "time_seconds")
    write_shaded_svg(joinpath(OUTDIR, "peak_memory_vs_T.svg"),
        "Peak Memory vs T", "peak memory (MB)", rows, "peak_memory_mb")

    @printf("wrote %s\n", summary_path)
    @printf("wrote plots under %s\n", OUTDIR)
end

main()
