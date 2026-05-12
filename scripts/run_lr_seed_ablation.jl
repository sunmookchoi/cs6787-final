#!/usr/bin/env julia

using CS6787Final
using LinearAlgebra
using Printf
using Random

const SEED_ABLATION_COLUMNS = [
    "system", "system_seed", "rho_A", "n", "p", "T", "d", "rank",
    "seed_count", "initial_seed", "seed_binary_objective",
    "seed_relaxed_objective", "best_binary_objective",
    "best_seed_relaxed_objective", "best_initial_seed", "best_seed_count",
    "status", "solve_seconds", "cumulative_solve_seconds",
    "allocated_bytes", "gc_seconds", "maxrss_bytes", "maxrss_mb",
    "matrix_build_seconds", "matrix_allocated_bytes", "round_restarts",
    "maxiter", "tol", "iterations", "best_iteration", "grad_norm",
    "converged",
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
            elseif item == "no-plot"
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
    text = strip(text)
    if occursin(":", text)
        parts = parse.(Int, strip.(split(text, ":")))
        if length(parts) == 2
            return collect(parts[1]:parts[2])
        elseif length(parts) == 3
            return collect(parts[1]:parts[2]:parts[3])
        end
        error("Range must be start:stop or start:step:stop")
    end
    return parse.(Int, strip.(split(text, ",")))
end

function parse_seed_counts(text::AbstractString)
    text = strip(text)
    if !occursin(":", text) && !occursin(",", text)
        return collect(1:parse(Int, text))
    end
    return parse_int_list(text)
end

nt_get(nt::NamedTuple, key::Symbol, default) = haskey(nt, key) ? nt[key] : default

function spectral_radius(A::AbstractMatrix)
    return maximum(abs.(eigvals(Matrix(A))))
end

function make_ablation_system(parsed, seed::Integer)
    system_name = lowercase(strip(opt(parsed, "system", "simple")))
    n = opt_int(parsed, "n", 3)
    p = opt_int(parsed, "p", 2)

    if system_name in ("simple", "current", "deterministic")
        rho = opt_float(parsed, "rho", 0.3)
        system = make_simple_system(rho=rho)
        return system, (
            system = "simple",
            system_seed = missing,
            rho_A = spectral_radius(system.A),
            n = size(system.A, 1),
            p = size(system.B, 2),
        )
    elseif system_name == "random"
        rho = opt_float(parsed, "rho", 0.5)
        system_seed = opt_int(parsed, "system-seed", 1)
        rng = Random.Xoshiro(system_seed)
        system = make_random_system(n, p; rho=rho, rng=rng)
        return system, (
            system = "random",
            system_seed = system_seed,
            rho_A = spectral_radius(system.A),
            n = n,
            p = p,
        )
    end

    error("--system must be simple/current/deterministic or random")
end

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

function write_csv(path::AbstractString, rows::Vector{Dict{String,Any}})
    dir = dirname(path)
    isempty(dir) || mkpath(dir)
    open(path, "w") do io
        println(io, join(SEED_ABLATION_COLUMNS, ","))
        for row in rows
            println(io, join((csv_cell(get(row, col, missing))
                for col in SEED_ABLATION_COLUMNS), ","))
        end
    end
    return path
end

xml_escape(s) = replace(string(s), "&" => "&amp;", "<" => "&lt;", ">" => "&gt;")

function nice_num(x::Real)
    xf = Float64(x)
    isfinite(xf) || return string(xf)
    ax = abs(xf)
    if ax >= 1e4 || (ax > 0 && ax < 1e-3)
        return @sprintf("%.2e", xf)
    elseif ax >= 100
        return @sprintf("%.0f", xf)
    elseif ax >= 10
        return @sprintf("%.1f", xf)
    end
    return @sprintf("%.3f", xf)
end

function write_seed_plot_svg(path::AbstractString, rows::Vector{Dict{String,Any}},
        rank::Integer)
    isempty(rows) && return nothing
    mkpath(dirname(path))

    Ts = sort(unique(Int(row["T"]) for row in rows))
    round_counts = sort(unique(Int(row["round_restarts"]) for row in rows))
    series = NamedTuple[]
    for T in Ts
        for round_restarts in round_counts
            t_rows = sort([row for row in rows if Int(row["T"]) == T &&
                Int(row["round_restarts"]) == round_restarts],
                by=row -> Int(row["seed_count"]))
            isempty(t_rows) && continue
            label = length(round_counts) == 1 ? "T=$(T)" :
                (length(Ts) == 1 ? "rounds=$(round_restarts)" :
                 "T=$(T), rounds=$(round_restarts)")
            push!(series, (
                label = label,
                xs = Float64[Float64(row["seed_count"]) for row in t_rows],
                ys = Float64[Float64(row["best_binary_objective"]) for row in t_rows],
            ))
        end
    end

    all_x = reduce(vcat, [s.xs for s in series])
    all_y = reduce(vcat, [s.ys for s in series])
    x0 = minimum(all_x)
    x1 = maximum(all_x)
    y0 = minimum(all_y)
    y1 = maximum(all_y)
    x0 == x1 && (x1 += 1)
    if y0 == y1
        pad = max(1.0, abs(y0) * 0.05)
        y0 -= pad
        y1 += pad
    else
        pad = 0.08 * (y1 - y0)
        y0 -= pad
        y1 += pad
    end

    width = 920
    height = 540
    left = 86
    right = 190
    top = 58
    bottom = 82
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = [
        "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
        "#17becf", "#8c564b", "#e377c2",
    ]
    sx(x) = left + (Float64(x) - x0) / (x1 - x0) * plot_w
    sy(y) = top + (y1 - Float64(y)) / (y1 - y0) * plot_h

    open(path, "w") do io
        println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$width" height="$height" viewBox="0 0 $width $height">""")
        println(io, """<rect width="100%" height="100%" fill="white"/>""")
        println(io, """<text x="$left" y="32" font-family="Arial" font-size="20" font-weight="700">LR-SDP r=$(rank) Initialization Seed Ablation</text>""")
        println(io, """<line x1="$left" y1="$(top + plot_h)" x2="$(left + plot_w)" y2="$(top + plot_h)" stroke="#333" stroke-width="1.2"/>""")
        println(io, """<line x1="$left" y1="$top" x2="$left" y2="$(top + plot_h)" stroke="#333" stroke-width="1.2"/>""")

        for t in 0:4
            y = y0 + t * (y1 - y0) / 4
            py = sy(y)
            println(io, """<line x1="$(left - 5)" y1="$py" x2="$left" y2="$py" stroke="#333"/>""")
            println(io, """<line x1="$left" y1="$py" x2="$(left + plot_w)" y2="$py" stroke="#e8e8e8"/>""")
            println(io, """<text x="$(left - 10)" y="$(py + 4)" text-anchor="end" font-family="Arial" font-size="12">$(nice_num(y))</text>""")
        end

        for t in 0:4
            x = x0 + t * (x1 - x0) / 4
            px = sx(x)
            println(io, """<line x1="$px" y1="$(top + plot_h)" x2="$px" y2="$(top + plot_h + 5)" stroke="#333"/>""")
            println(io, """<text x="$px" y="$(top + plot_h + 24)" text-anchor="middle" font-family="Arial" font-size="12">$(nice_num(x))</text>""")
        end

        for (idx, s) in enumerate(series)
            color = colors[mod1(idx, length(colors))]
            path_parts = String[]
            for (point_idx, (x, y)) in enumerate(zip(s.xs, s.ys))
                push!(path_parts, "$(point_idx == 1 ? "M" : "L") $(sx(x)) $(sy(y))")
            end
            println(io, """<path d="$(join(path_parts, " "))" fill="none" stroke="$color" stroke-width="2.3"/>""")
            for (x, y) in zip(s.xs, s.ys)
                println(io, """<circle cx="$(sx(x))" cy="$(sy(y))" r="2.8" fill="$color"/>""")
            end
        end

        println(io, """<text x="$(left + plot_w / 2)" y="$(height - 24)" text-anchor="middle" font-family="Arial" font-size="14">number of initial seeds K</text>""")
        println(io, """<text transform="translate(24,$(top + plot_h / 2)) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="14">max binary objective over first K seeds</text>""")

        for (idx, s) in enumerate(series)
            color = colors[mod1(idx, length(colors))]
            legend_y = top + 24 * (idx - 1)
            println(io, """<line x1="$(left + plot_w + 28)" y1="$legend_y" x2="$(left + plot_w + 48)" y2="$legend_y" stroke="$color" stroke-width="2.3"/>""")
            println(io, """<text x="$(left + plot_w + 58)" y="$(legend_y + 4)" font-family="Arial" font-size="12">$(xml_escape(s.label))</text>""")
        end

        println(io, "</svg>")
    end
    return path
end

function write_round_restart_plot_svg(path::AbstractString,
        rows::Vector{Dict{String,Any}}, rank::Integer)
    isempty(rows) && return nothing
    mkpath(dirname(path))

    Ts = sort(unique(Int(row["T"]) for row in rows))
    seed_counts = sort(unique(Int(row["seed_count"]) for row in rows))
    series = NamedTuple[]
    for T in Ts
        for seed_count in seed_counts
            t_rows = sort([row for row in rows if Int(row["T"]) == T &&
                Int(row["seed_count"]) == seed_count],
                by=row -> Int(row["round_restarts"]))
            isempty(t_rows) && continue
            label = length(Ts) == 1 ? "K=$(seed_count)" :
                "T=$(T), K=$(seed_count)"
            push!(series, (
                label = label,
                xs = Float64[Float64(row["round_restarts"]) for row in t_rows],
                ys = Float64[Float64(row["best_binary_objective"]) for row in t_rows],
            ))
        end
    end

    all_x = reduce(vcat, [s.xs for s in series])
    all_y = reduce(vcat, [s.ys for s in series])
    x0 = minimum(all_x)
    x1 = maximum(all_x)
    y0 = minimum(all_y)
    y1 = maximum(all_y)
    x0 == x1 && (x1 += 1)
    if y0 == y1
        pad = max(1.0, abs(y0) * 0.05)
        y0 -= pad
        y1 += pad
    else
        pad = 0.08 * (y1 - y0)
        y0 -= pad
        y1 += pad
    end

    width = 920
    height = 540
    left = 86
    right = 190
    top = 58
    bottom = 82
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = [
        "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
        "#17becf", "#8c564b", "#e377c2",
    ]
    sx(x) = left + (Float64(x) - x0) / (x1 - x0) * plot_w
    sy(y) = top + (y1 - Float64(y)) / (y1 - y0) * plot_h

    open(path, "w") do io
        println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$width" height="$height" viewBox="0 0 $width $height">""")
        println(io, """<rect width="100%" height="100%" fill="white"/>""")
        println(io, """<text x="$left" y="32" font-family="Arial" font-size="20" font-weight="700">LR-SDP r=$(rank) Rounding Restarts Ablation</text>""")
        println(io, """<line x1="$left" y1="$(top + plot_h)" x2="$(left + plot_w)" y2="$(top + plot_h)" stroke="#333" stroke-width="1.2"/>""")
        println(io, """<line x1="$left" y1="$top" x2="$left" y2="$(top + plot_h)" stroke="#333" stroke-width="1.2"/>""")

        for t in 0:4
            y = y0 + t * (y1 - y0) / 4
            py = sy(y)
            println(io, """<line x1="$(left - 5)" y1="$py" x2="$left" y2="$py" stroke="#333"/>""")
            println(io, """<line x1="$left" y1="$py" x2="$(left + plot_w)" y2="$py" stroke="#e8e8e8"/>""")
            println(io, """<text x="$(left - 10)" y="$(py + 4)" text-anchor="end" font-family="Arial" font-size="12">$(nice_num(y))</text>""")
        end

        for t in 0:4
            x = x0 + t * (x1 - x0) / 4
            px = sx(x)
            println(io, """<line x1="$px" y1="$(top + plot_h)" x2="$px" y2="$(top + plot_h + 5)" stroke="#333"/>""")
            println(io, """<text x="$px" y="$(top + plot_h + 24)" text-anchor="middle" font-family="Arial" font-size="12">$(nice_num(x))</text>""")
        end

        for (idx, s) in enumerate(series)
            color = colors[mod1(idx, length(colors))]
            path_parts = String[]
            for (point_idx, (x, y)) in enumerate(zip(s.xs, s.ys))
                push!(path_parts, "$(point_idx == 1 ? "M" : "L") $(sx(x)) $(sy(y))")
            end
            println(io, """<path d="$(join(path_parts, " "))" fill="none" stroke="$color" stroke-width="2.3"/>""")
            for (x, y) in zip(s.xs, s.ys)
                println(io, """<circle cx="$(sx(x))" cy="$(sy(y))" r="2.8" fill="$color"/>""")
            end
        end

        println(io, """<text x="$(left + plot_w / 2)" y="$(height - 24)" text-anchor="middle" font-family="Arial" font-size="14">round restarts</text>""")
        println(io, """<text transform="translate(24,$(top + plot_h / 2)) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="14">max binary objective over first K seeds</text>""")

        for (idx, s) in enumerate(series)
            color = colors[mod1(idx, length(colors))]
            legend_y = top + 24 * (idx - 1)
            println(io, """<line x1="$(left + plot_w + 28)" y1="$legend_y" x2="$(left + plot_w + 48)" y2="$legend_y" stroke="$color" stroke-width="2.3"/>""")
            println(io, """<text x="$(left + plot_w + 58)" y="$(legend_y + 4)" font-family="Arial" font-size="12">$(xml_escape(s.label))</text>""")
        end

        println(io, "</svg>")
    end
    return path
end

function run_seed_ablation(parsed)
    seed = opt_int(parsed, "seed", 1)
    Ts = parse_int_list(haskey(parsed, "Ts") ? parsed["Ts"] : opt(parsed, "T", "2000"))
    rank = opt_int(parsed, "rank", 4)
    ablation = lowercase(strip(opt(parsed, "ablation", "seed")))
    @assert ablation in ("seed", "round") "--ablation must be seed or round"
    seed_counts = if ablation == "round"
        sort(unique(parse_int_list(opt(parsed, "seed-counts",
            string(opt_int(parsed, "max-seeds", 64))))))
    elseif haskey(parsed, "seed-counts")
        sort(unique(parse_seed_counts(parsed["seed-counts"])))
    else
        collect(1:opt_int(parsed, "max-seeds", 64))
    end
    max_seeds = maximum(seed_counts)
    seed_count_set = Set(seed_counts)
    maxiter = opt_int(parsed, "maxiter", 1000)
    tol = opt_float(parsed, "tol", 1e-5)
    round_restart_counts = sort(unique(parse_int_list(opt(parsed,
        "round-restarts", "64"))))
    output = opt(parsed, "output", joinpath(@__DIR__, "..", "results",
        "lr_seed_ablation.csv"))
    plot_path = opt(parsed, "plot", replace(output, r"\.csv$" => ".svg"))
    round_plot_path = opt(parsed, "round-plot",
        replace(output, r"\.csv$" => "_round_restarts.svg"))

    @assert rank >= 1 "--rank must be positive"
    @assert !isempty(seed_counts) "--seed-counts must not be empty"
    @assert all(>(0), seed_counts) "--seed-counts values must be positive"
    @assert maxiter >= 1 "--maxiter must be positive"
    @assert !isempty(round_restart_counts) "--round-restarts must not be empty"
    @assert all(>(0), round_restart_counts) "--round-restarts values must be positive"

    system, system_info = make_ablation_system(parsed, seed)
    rows = Dict{String,Any}[]

    @printf("system=%s n=%d p=%d rho_A=%.4f seed=%s ablation=%s rank=%d max_seeds=%d checkpoints=%s round_restarts=%s\n",
        system_info.system, system_info.n, system_info.p,
        Float64(system_info.rho_A), string(system_info.system_seed),
        ablation, rank, max_seeds, join(seed_counts, ","),
        join(round_restart_counts, ","))

    for T in Ts
        matrix_timed = @timed build_true_qubo_matrix(system, T)
        W = matrix_timed.value
        d = size(W, 1)
        best_binary = Dict(rr => -Inf for rr in round_restart_counts)
        best_seed_relaxed = Dict(rr => -Inf for rr in round_restart_counts)
        best_initial_seed = Dict(rr => 0 for rr in round_restart_counts)
        best_seed_count = Dict(rr => 0 for rr in round_restart_counts)
        cumulative_factor_time = 0.0
        cumulative_round_time = Dict(rr => 0.0 for rr in round_restart_counts)

        for seed_count in 1:max_seeds
            initial_seed = seed + 1_000_000 * T + 10_000 * rank + seed_count
            rng = Random.Xoshiro(initial_seed)
            factor_timed = @timed low_rank_sdp_qubo(W;
                rank=rank,
                restarts=1,
                maxiter=maxiter,
                tol=tol,
                round_restarts=1,
                rng=rng)
            sol = factor_timed.value
            seed_relaxed = Float64(sol.relaxed_value)
            cumulative_factor_time += factor_timed.time

            for round_restarts in round_restart_counts
                round_seed = seed + 2_000_000 * T + 10_000 * rank + seed_count
                round_timed = @timed hyperplane_round_qubo(W, sol.Y;
                    restarts=round_restarts,
                    rng=Random.Xoshiro(round_seed))
                seed_binary = Float64(round_timed.value[2])
                cumulative_round_time[round_restarts] += round_timed.time

                if seed_binary > best_binary[round_restarts]
                    best_binary[round_restarts] = seed_binary
                    best_seed_relaxed[round_restarts] = seed_relaxed
                    best_initial_seed[round_restarts] = initial_seed
                    best_seed_count[round_restarts] = seed_count
                end

                if seed_count in seed_count_set
                    rss = Sys.maxrss()
                    push!(rows, Dict{String,Any}(
                        "system" => system_info.system,
                        "system_seed" => system_info.system_seed,
                        "rho_A" => system_info.rho_A,
                        "n" => system_info.n,
                        "p" => system_info.p,
                        "T" => T,
                        "d" => d,
                        "rank" => rank,
                        "seed_count" => seed_count,
                        "initial_seed" => initial_seed,
                        "seed_binary_objective" => seed_binary,
                        "seed_relaxed_objective" => seed_relaxed,
                        "best_binary_objective" => best_binary[round_restarts],
                        "best_seed_relaxed_objective" =>
                            best_seed_relaxed[round_restarts],
                        "best_initial_seed" => best_initial_seed[round_restarts],
                        "best_seed_count" => best_seed_count[round_restarts],
                        "status" => nt_get(sol, :status, "ok"),
                        "solve_seconds" => factor_timed.time + round_timed.time,
                        "cumulative_solve_seconds" => cumulative_factor_time +
                            cumulative_round_time[round_restarts],
                        "allocated_bytes" => factor_timed.bytes + round_timed.bytes,
                        "gc_seconds" => factor_timed.gctime + round_timed.gctime,
                        "maxrss_bytes" => rss,
                        "maxrss_mb" => rss / 2.0^20,
                        "matrix_build_seconds" => matrix_timed.time,
                        "matrix_allocated_bytes" => matrix_timed.bytes,
                        "round_restarts" => round_restarts,
                        "maxiter" => maxiter,
                        "tol" => tol,
                        "iterations" => nt_get(sol, :iterations, missing),
                        "best_iteration" => nt_get(sol, :best_iteration, missing),
                        "grad_norm" => nt_get(sol, :grad_norm, missing),
                        "converged" => nt_get(sol, :converged, missing),
                    ))

                    @printf("LR-SDP r=%d T=%4d seed_count=%3d/%d rounds=%3d obj=% .6e best=% .6e best_seed_count=%d sec=%.3f\n",
                        rank, T, seed_count, max_seeds, round_restarts,
                        seed_binary, best_binary[round_restarts],
                        best_seed_count[round_restarts],
                        factor_timed.time + round_timed.time)
                end
            end
            GC.gc()
        end
    end

    write_csv(output, rows)
    @printf("wrote %d rows to %s\n", length(rows), output)
    if !haskey(parsed, "no-plot")
        if ablation == "round"
            write_round_restart_plot_svg(plot_path, rows, rank)
            @printf("wrote round restart plot to %s\n", plot_path)
        else
            write_seed_plot_svg(plot_path, rows, rank)
            @printf("wrote plot to %s\n", plot_path)
        end
        if ablation == "seed" && length(round_restart_counts) > 1
            write_round_restart_plot_svg(round_plot_path, rows, rank)
            @printf("wrote round restart plot to %s\n", round_plot_path)
        end
    end
    return rows
end

run_seed_ablation(parse_args(ARGS))
