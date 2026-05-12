#!/usr/bin/env julia

using CS6787Final
using Printf
using Random
using Statistics

const COLUMNS = [
    "T", "d", "method", "method_label", "rank", "start_index", "seed",
    "status", "binary_objective", "continuous_objective", "relaxed_objective",
    "solve_seconds", "iterations", "best_iteration", "grad_norm", "converged",
    "round_restarts", "maxrss_mb",
]

function parse_args(args)
    parsed = Dict{String,String}()
    i = 1
    while i <= length(args)
        arg = args[i]
        if startswith(arg, "--")
            item = arg[3:end]
            if occursin("=", item)
                key, value = split(item, "="; limit=2)
                parsed[key] = value
                i += 1
            elseif item in ("no-sdp", "no-plots", "only-sdp", "append",
                    "plot-existing")
                parsed[item] = "true"
                i += 1
            else
                @assert i < length(args) "Missing value for --$(item)"
                parsed[item] = args[i + 1]
                i += 2
            end
        else
            error("Unexpected positional argument: $(arg)")
        end
    end
    return parsed
end

opt(parsed, key, default) = get(parsed, key, default)
opt_int(parsed, key, default) = parse(Int, opt(parsed, key, string(default)))
opt_float(parsed, key, default) = parse(Float64, opt(parsed, key, string(default)))

function parse_int_list(text::AbstractString)
    return parse.(Int, strip.(split(text, ",")))
end

adaptive_base_restarts(T::Integer) = max(1, floor(Int, 64 * Float64(T) / 2000))
adaptive_low_rank_restarts(T::Integer, rank::Integer) =
    max(1, floor(Int, 64 * Float64(rank) / 2 * Float64(T) / 2000))

function method_seed(base_seed::Integer, T::Integer, method_idx::Integer;
        rank=missing, rep::Integer=1)
    return base_seed + 1_000_000 * rep + 10_000 * T + 101 * method_idx +
        (rank === missing ? 0 : Int(rank))
end

function method_label(method::Symbol, rank)
    method == :sign_iteration && return "Sign-Iteration"
    method == :box_pgd && return "Box-PGA"
    method == :low_rank && return "LR-SDP r=$(rank)"
    method == :sdp_gw && return "SDP+GW"
    return String(method)
end

nt_get(nt::NamedTuple, key::Symbol, default) = haskey(nt, key) ? nt[key] : default

function csv_cell(x)
    if x === missing || x === nothing
        return ""
    end
    s = string(x)
    if occursin(",", s) || occursin("\"", s) || occursin("\n", s)
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

function write_header(path::AbstractString)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(COLUMNS, ","))
    end
end

function append_row(path::AbstractString, row::Dict{String,Any})
    open(path, "a") do io
        println(io, join((csv_cell(get(row, col, missing)) for col in COLUMNS), ","))
    end
end

function read_rows(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && return Dict{String,Any}[]
    columns = split(lines[1], ",")
    rows = Dict{String,Any}[]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        values = split(line, ",")
        row = Dict{String,Any}()
        for (idx, col) in enumerate(columns)
            raw = idx <= length(values) ? values[idx] : ""
            if col in ("T", "d", "rank", "start_index", "seed", "iterations",
                    "best_iteration", "round_restarts")
                row[col] = isempty(raw) ? missing : parse(Int, raw)
            elseif col in ("binary_objective", "continuous_objective",
                    "relaxed_objective", "solve_seconds", "grad_norm",
                    "maxrss_mb")
                row[col] = isempty(raw) ? missing : parse(Float64, raw)
            elseif col == "converged"
                row[col] = isempty(raw) ? missing : raw == "true"
            else
                row[col] = raw
            end
        end
        push!(rows, row)
    end
    return rows
end

function result_row(T, d, method, rank, start_index, seed, sol, timed)
    return Dict{String,Any}(
        "T" => T,
        "d" => d,
        "method" => String(method),
        "method_label" => method_label(method, rank),
        "rank" => rank,
        "start_index" => start_index,
        "seed" => seed,
        "status" => nt_get(sol, :status, "ok"),
        "binary_objective" => nt_get(sol, :binary_value,
            nt_get(sol, :rounded_value, missing)),
        "continuous_objective" => nt_get(sol, :continuous_value, missing),
        "relaxed_objective" => nt_get(sol, :relaxed_value, missing),
        "solve_seconds" => timed.time,
        "iterations" => nt_get(sol, :iterations, missing),
        "best_iteration" => nt_get(sol, :best_iteration, missing),
        "grad_norm" => nt_get(sol, :grad_norm, missing),
        "converged" => nt_get(sol, :converged, missing),
        "round_restarts" => nt_get(sol, :round_restarts, missing),
        "maxrss_mb" => Sys.maxrss() / 2.0^20,
    )
end

function failure_row(T, d, method, rank, start_index, seed, err, elapsed)
    return Dict{String,Any}(
        "T" => T,
        "d" => d,
        "method" => String(method),
        "method_label" => method_label(method, rank),
        "rank" => rank,
        "start_index" => start_index,
        "seed" => seed,
        "status" => "failed: $(typeof(err)): $(err)",
        "binary_objective" => missing,
        "continuous_objective" => missing,
        "relaxed_objective" => missing,
        "solve_seconds" => elapsed,
        "iterations" => missing,
        "best_iteration" => missing,
        "grad_norm" => missing,
        "converged" => false,
        "round_restarts" => missing,
        "maxrss_mb" => Sys.maxrss() / 2.0^20,
    )
end

function run_single_start(method::Symbol, W::AbstractMatrix, rank, rng,
        round_restarts::Integer, parsed)
    if method == :sign_iteration
        return sign_iteration(W; restarts=1,
            maxiter=opt_int(parsed, "sign-maxiter", 200), rng=rng)
    elseif method == :box_pgd
        return box_projected_gradient(W; restarts=1,
            maxiter=opt_int(parsed, "box-maxiter", 1000),
            tol=opt_float(parsed, "box-tol", 1e-5), rng=rng)
    elseif method == :low_rank
        return low_rank_sdp_qubo(W; rank=rank, restarts=1,
            maxiter=opt_int(parsed, "low-rank-maxiter", 1000),
            tol=opt_float(parsed, "low-rank-tol", 1e-5),
            round_restarts=round_restarts, rng=rng)
    elseif method == :sdp_gw
        return solve_sdp_gw_qubo(W; restarts=round_restarts, rng=rng)
    end
    error("Unknown method: $(method)")
end

finite_real(x) = x isa Real && isfinite(Float64(x))

function group_rows(rows)
    groups = Dict{String,Vector{Dict{String,Any}}}()
    for row in rows
        finite_real(row["binary_objective"]) || continue
        push!(get!(groups, string(row["method_label"]), Dict{String,Any}[]), row)
    end
    return groups
end

function label_sort_key(label)
    label == "SDP+GW" && return 1000
    label == "Sign-Iteration" && return 10
    label == "Box-PGA" && return 20
    m = match(r"LR-SDP r=(\d+)", label)
    m !== nothing && return 100 + parse(Int, m.captures[1])
    return 10_000
end

function nice_num(x)
    if abs(x) >= 1000 || (abs(x) > 0 && abs(x) < 0.01)
        return @sprintf("%.2e", x)
    end
    return @sprintf("%.3g", x)
end

xml_escape(s) = replace(string(s), "&" => "&amp;", "<" => "&lt;", ">" => "&gt;")

function write_objective_plot(path::AbstractString, rows)
    groups = group_rows(rows)
    labels = sort(collect(keys(groups)); by=label_sort_key)
    isempty(labels) && return nothing

    means = Float64[]
    sds = Float64[]
    ns = Int[]
    for label in labels
        vals = Float64[row["binary_objective"] for row in groups[label]]
        push!(means, mean(vals))
        push!(sds, length(vals) > 1 ? std(vals) : 0.0)
        push!(ns, length(vals))
    end

    y0 = minimum(means .- sds)
    y1 = maximum(means .+ sds)
    y0 == y1 && (y0 -= 1.0; y1 += 1.0)
    pad = 0.08 * (y1 - y0)
    y0 -= pad
    y1 += pad

    width, height = 980, 560
    left, right, top, bottom = 90, 40, 58, 108
    plot_w = width - left - right
    plot_h = height - top - bottom
    sx(i) = left + (i - 0.5) / length(labels) * plot_w
    sy(y) = top + (y1 - y) / (y1 - y0) * plot_h
    bar_w = min(72.0, 0.55 * plot_w / length(labels))

    mkpath(dirname(path))
    open(path, "w") do io
        println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$width" height="$height" viewBox="0 0 $width $height">""")
        println(io, """<rect width="100%" height="100%" fill="white"/>""")
        println(io, """<text x="$left" y="32" font-family="Arial" font-size="20" font-weight="700">T=2000 objective by method</text>""")
        println(io, """<text x="$left" y="52" font-family="Arial" font-size="12" fill="#555">Bars show mean x'Wx; error bars are ±1 standard deviation across initial points. SDP+GW is one run.</text>""")
        println(io, """<line x1="$left" y1="$(top + plot_h)" x2="$(left + plot_w)" y2="$(top + plot_h)" stroke="#333"/>""")
        println(io, """<line x1="$left" y1="$top" x2="$left" y2="$(top + plot_h)" stroke="#333"/>""")
        for t in 0:4
            y = y0 + t * (y1 - y0) / 4
            py = sy(y)
            println(io, """<line x1="$(left - 5)" y1="$py" x2="$left" y2="$py" stroke="#333"/>""")
            println(io, """<line x1="$left" y1="$py" x2="$(left + plot_w)" y2="$py" stroke="#e8e8e8"/>""")
            println(io, """<text x="$(left - 10)" y="$(py + 4)" text-anchor="end" font-family="Arial" font-size="12">$(nice_num(y))</text>""")
        end
        colors = ["#2f6f73", "#c44e36", "#5b7cba", "#7d5ba6", "#c78d2e", "#4f8f49", "#777777"]
        for (i, label) in enumerate(labels)
            x = sx(i)
            y = sy(means[i])
            h = top + plot_h - y
            color = colors[mod1(i, length(colors))]
            println(io, """<rect x="$(x - bar_w / 2)" y="$y" width="$bar_w" height="$h" fill="$color" opacity="0.88"/>""")
            if sds[i] > 0
                ylo = sy(means[i] - sds[i])
                yhi = sy(means[i] + sds[i])
                println(io, """<line x1="$x" y1="$yhi" x2="$x" y2="$ylo" stroke="#222" stroke-width="1.5"/>""")
                println(io, """<line x1="$(x - 10)" y1="$yhi" x2="$(x + 10)" y2="$yhi" stroke="#222" stroke-width="1.5"/>""")
                println(io, """<line x1="$(x - 10)" y1="$ylo" x2="$(x + 10)" y2="$ylo" stroke="#222" stroke-width="1.5"/>""")
            end
            println(io, """<text x="$x" y="$(top + plot_h + 24)" text-anchor="middle" font-family="Arial" font-size="12">$(xml_escape(label))</text>""")
            println(io, """<text x="$x" y="$(top + plot_h + 42)" text-anchor="middle" font-family="Arial" font-size="11" fill="#555">n=$(ns[i])</text>""")
        end
        println(io, """<text transform="translate(24,$(top + plot_h / 2)) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="14">binary objective x'Wx</text>""")
        println(io, "</svg>")
    end
    return path
end

function write_runtime_plot(path::AbstractString, rows)
    groups = group_rows(rows)
    labels = sort(collect(keys(groups)); by=label_sort_key)
    isempty(labels) && return nothing

    totals = Float64[]
    for label in labels
        vals = [Float64(row["solve_seconds"]) for row in groups[label] if
            finite_real(row["solve_seconds"])]
        push!(totals, sum(vals))
    end
    positive = [v for v in totals if v > 0]
    isempty(positive) && return nothing
    y0 = minimum(positive) / 1.5
    y1 = maximum(positive) * 1.5

    width, height = 980, 560
    left, right, top, bottom = 90, 40, 58, 108
    plot_w = width - left - right
    plot_h = height - top - bottom
    sx(i) = left + (i - 0.5) / length(labels) * plot_w
    sy(y) = top + (log10(y1) - log10(max(y, y0))) /
        (log10(y1) - log10(y0)) * plot_h
    bar_w = min(72.0, 0.55 * plot_w / length(labels))

    mkpath(dirname(path))
    open(path, "w") do io
        println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$width" height="$height" viewBox="0 0 $width $height">""")
        println(io, """<rect width="100%" height="100%" fill="white"/>""")
        println(io, """<text x="$left" y="32" font-family="Arial" font-size="20" font-weight="700">T=2000 runtime by method</text>""")
        println(io, """<text x="$left" y="52" font-family="Arial" font-size="12" fill="#555">Bars show total wall-clock solve time for the current restart budget; log-scale y-axis.</text>""")
        println(io, """<line x1="$left" y1="$(top + plot_h)" x2="$(left + plot_w)" y2="$(top + plot_h)" stroke="#333"/>""")
        println(io, """<line x1="$left" y1="$top" x2="$left" y2="$(top + plot_h)" stroke="#333"/>""")
        for t in 0:4
            ly = log10(y0) + t * (log10(y1) - log10(y0)) / 4
            y = 10.0^ly
            py = sy(y)
            println(io, """<line x1="$(left - 5)" y1="$py" x2="$left" y2="$py" stroke="#333"/>""")
            println(io, """<line x1="$left" y1="$py" x2="$(left + plot_w)" y2="$py" stroke="#e8e8e8"/>""")
            println(io, """<text x="$(left - 10)" y="$(py + 4)" text-anchor="end" font-family="Arial" font-size="12">$(nice_num(y))</text>""")
        end
        colors = ["#2f6f73", "#c44e36", "#5b7cba", "#7d5ba6", "#c78d2e", "#4f8f49", "#777777"]
        for (i, label) in enumerate(labels)
            x = sx(i)
            y = sy(totals[i])
            h = top + plot_h - y
            color = colors[mod1(i, length(colors))]
            println(io, """<rect x="$(x - bar_w / 2)" y="$y" width="$bar_w" height="$h" fill="$color" opacity="0.88"/>""")
            println(io, """<text x="$x" y="$(top + plot_h + 24)" text-anchor="middle" font-family="Arial" font-size="12">$(xml_escape(label))</text>""")
            println(io, """<text x="$x" y="$(top + plot_h + 42)" text-anchor="middle" font-family="Arial" font-size="11" fill="#555">$(nice_num(totals[i]))s</text>""")
        end
        println(io, """<text transform="translate(24,$(top + plot_h / 2)) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="14">seconds, log scale</text>""")
        println(io, "</svg>")
    end
    return path
end

function main(args)
    parsed = parse_args(args)
    T = opt_int(parsed, "T", 2000)
    seed = opt_int(parsed, "seed", 1)
    ranks = parse_int_list(opt(parsed, "ranks", "2,4,6,8"))
    output = opt(parsed, "output", joinpath(@__DIR__, "..", "results",
        "t2000_seeded_qubo.csv"))
    plot_dir = opt(parsed, "plot-dir", joinpath(dirname(output), "plots"))
    max_starts = opt_int(parsed, "max-starts", 0)
    round_restarts = opt_int(parsed, "round-restarts", adaptive_base_restarts(T))

    if haskey(parsed, "plot-existing")
        rows = read_rows(output)
        write_objective_plot(joinpath(plot_dir, "t2000_objective_errorbars.svg"), rows)
        write_runtime_plot(joinpath(plot_dir, "t2000_runtime.svg"), rows)
        @printf("wrote plots to %s from %s\n", plot_dir, output)
        return nothing
    end

    system = make_simple_system(rho=opt_float(parsed, "rho", 0.3))
    matrix_timed = @timed build_true_qubo_matrix(system, T)
    W = matrix_timed.value
    d = size(W, 1)
    @printf("built W for T=%d d=%d in %.3fs\n", T, d, matrix_timed.time)

    if !(haskey(parsed, "append") && isfile(output))
        write_header(output)
    end
    rows = Dict{String,Any}[]

    cases = Tuple{Symbol,Any,Int,Int}[]
    if haskey(parsed, "only-sdp")
        push!(cases, (:sdp_gw, missing, 4, 1))
    else
        push!(cases, (:sign_iteration, missing, 1, adaptive_base_restarts(T)))
        push!(cases, (:box_pgd, missing, 2, adaptive_base_restarts(T)))
        for rank in ranks
            push!(cases, (:low_rank, rank, 3, adaptive_low_rank_restarts(T, rank)))
        end
        if !haskey(parsed, "no-sdp")
            push!(cases, (:sdp_gw, missing, 4, 1))
        end
    end

    for (method, rank, method_idx, starts) in cases
        actual_starts = max_starts > 0 && method != :sdp_gw ?
            min(starts, max_starts) : starts
        base_seed = method_seed(seed, T, method_idx; rank=rank)
        @printf("case %-14s starts=%d base_seed=%d\n",
            method_label(method, rank), actual_starts, base_seed)
        for start_index in 1:actual_starts
            run_seed = base_seed + start_index - 1
            rng = Random.Xoshiro(run_seed)
            t0 = time()
            row = try
                timed = @timed run_single_start(method, W, rank, rng,
                    round_restarts, parsed)
                result_row(T, d, method, rank, start_index, run_seed, timed.value, timed)
            catch err
                failure_row(T, d, method, rank, start_index, run_seed, err,
                    time() - t0)
            end
            append_row(output, row)
            push!(rows, row)
            @printf("%-14s start=%4d seed=%d status=%s obj=%s sec=%s\n",
                row["method_label"], start_index, run_seed, row["status"],
                string(row["binary_objective"]), string(row["solve_seconds"]))
            GC.gc()
        end
    end

    if !haskey(parsed, "no-plots")
        write_objective_plot(joinpath(plot_dir, "t2000_objective_errorbars.svg"), rows)
        write_runtime_plot(joinpath(plot_dir, "t2000_runtime.svg"), rows)
        @printf("wrote plots to %s\n", plot_dir)
    end
    @printf("wrote rows to %s\n", output)
end

main(ARGS)
