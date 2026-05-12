#!/usr/bin/env julia

using CS6787Final
using LinearAlgebra
using Printf
using Random
using Statistics

const QUBO_COLUMNS = [
    "system", "system_seed", "rho_A", "n", "p", "rep", "T", "d",
    "action_dim", "method", "method_label", "status", "rank",
    "binary_objective", "normalized_binary_quality",
    "quality_ref_binary_objective", "continuous_objective",
    "best_continuous_objective", "relaxed_objective", "solve_seconds",
    "allocated_bytes", "gc_seconds", "maxrss_bytes", "maxrss_mb",
    "matrix_build_seconds", "matrix_allocated_bytes", "restarts",
    "round_restarts", "iterations", "best_iteration", "best_restart",
    "grad_norm", "converged",
]

const ANYTIME_COLUMNS = [
    "system", "system_seed", "rho_A", "n", "p", "rep", "T", "d",
    "method", "method_label", "rank", "trace_index", "wall_seconds",
    "best_binary_objective",
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
            elseif item in ("no-plots", "no-warmup", "no-anytime")
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

function adaptive_base_restarts(T::Integer)
    return max(1, floor(Int, 64 * Float64(T) / 2000))
end

function adaptive_low_rank_restarts(T::Integer, rank::Integer)
    return max(1, floor(Int, 64 * Float64(rank) / 2 * Float64(T) / 2000))
end

function opt_or_adaptive_base(parsed, key::AbstractString, T::Integer)
    return haskey(parsed, key) ? parse(Int, parsed[key]) : adaptive_base_restarts(T)
end

function sdp_rounding_count(parsed, T::Integer)
    if haskey(parsed, "sdp-round-restarts")
        return parse(Int, parsed["sdp-round-restarts"])
    elseif haskey(parsed, "round-restarts")
        return parse(Int, parsed["round-restarts"])
    end
    return adaptive_base_restarts(T)
end

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

function parse_methods(text::AbstractString)
    aliases = Dict(
        "sign" => :sign_iteration,
        "sign_iteration" => :sign_iteration,
        "box" => :box_pgd,
        "box_pgd" => :box_pgd,
        "projected_gradient" => :box_pgd,
        "low_rank" => :low_rank,
        "lr_sdp" => :low_rank,
        "sdp" => :sdp_gw,
        "sdp_gw" => :sdp_gw,
    )
    methods = Symbol[]
    for raw in split(lowercase(text), ",")
        key = strip(raw)
        haskey(aliases, key) || error("Unknown method $(key)")
        push!(methods, aliases[key])
    end
    return methods
end

nt_get(nt::NamedTuple, key::Symbol, default) = haskey(nt, key) ? nt[key] : default

function spectral_radius(A::AbstractMatrix)
    return maximum(abs.(eigvals(Matrix(A))))
end

function make_benchmark_system(parsed, seed::Integer)
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

function method_label(method::Symbol, rank)
    if method == :low_rank
        return "LR-SDP r=$(rank)"
    elseif method == :sdp_gw
        return "SDP+GW"
    elseif method == :sign_iteration
        return "Sign-Iteration"
    elseif method == :box_pgd
        return "Box-PGA"
    end
    return String(method)
end

function solve_method(method::Symbol, W::AbstractMatrix, parsed, rank, rng,
        T::Integer;
        trace::Bool=false)
    trace_interval = opt_int(parsed, "anytime-trace-interval", 10)
    if method == :sign_iteration
        return sign_iteration(W;
            restarts=opt_or_adaptive_base(parsed, "sign-restarts", T),
            maxiter=opt_int(parsed, "sign-maxiter", 200),
            rng=rng,
            trace=trace,
            trace_interval=trace_interval)
    elseif method == :box_pgd
        return box_projected_gradient(W;
            restarts=opt_or_adaptive_base(parsed, "box-restarts", T),
            maxiter=opt_int(parsed, "box-maxiter", 1000),
            tol=opt_float(parsed, "box-tol", 1e-5),
            rng=rng,
            trace=trace,
            trace_interval=trace_interval)
    elseif method == :low_rank
        lr_restarts = haskey(parsed, "low-rank-restarts") ?
            parse(Int, parsed["low-rank-restarts"]) :
            adaptive_low_rank_restarts(T, Int(rank))
        return low_rank_sdp_qubo(W;
            rank=rank,
            restarts=lr_restarts,
            maxiter=opt_int(parsed, "low-rank-maxiter", 1000),
            tol=opt_float(parsed, "low-rank-tol", 1e-5),
            round_restarts=sdp_rounding_count(parsed, T),
            rng=rng,
            trace=trace,
            trace_interval=trace_interval,
            trace_round_restarts=opt_int(parsed, "anytime-round-restarts", 1))
    elseif method == :sdp_gw
        return solve_sdp_gw_qubo(W;
            restarts=sdp_rounding_count(parsed, T),
            rng=rng)
    end
    error("Unknown method: $(method)")
end

function success_row(system_info, rep, T, action_dim, method, rank, sol, timed,
        matrix_time, matrix_bytes)
    rounded = nt_get(sol, :binary_value, nt_get(sol, :rounded_value, missing))
    rss = Sys.maxrss()
    return Dict{String,Any}(
        "system" => system_info.system,
        "system_seed" => system_info.system_seed,
        "rho_A" => system_info.rho_A,
        "n" => system_info.n,
        "p" => system_info.p,
        "rep" => rep,
        "T" => T,
        "d" => length(sol.x),
        "action_dim" => action_dim,
        "method" => String(method),
        "method_label" => method_label(method, rank),
        "status" => nt_get(sol, :status, "ok"),
        "rank" => rank === missing ? missing : rank,
        "binary_objective" => rounded,
        "normalized_binary_quality" => missing,
        "quality_ref_binary_objective" => missing,
        "continuous_objective" => nt_get(sol, :continuous_value, missing),
        "best_continuous_objective" => nt_get(sol, :best_continuous_value, missing),
        "relaxed_objective" => nt_get(sol, :relaxed_value, missing),
        "solve_seconds" => timed.time,
        "allocated_bytes" => timed.bytes,
        "gc_seconds" => timed.gctime,
        "maxrss_bytes" => rss,
        "maxrss_mb" => rss / 2.0^20,
        "matrix_build_seconds" => matrix_time,
        "matrix_allocated_bytes" => matrix_bytes,
        "restarts" => nt_get(sol, :restarts, missing),
        "round_restarts" => nt_get(sol, :round_restarts, missing),
        "iterations" => nt_get(sol, :iterations, missing),
        "best_iteration" => nt_get(sol, :best_iteration, missing),
        "best_restart" => nt_get(sol, :best_restart, missing),
        "grad_norm" => nt_get(sol, :grad_norm, missing),
        "converged" => nt_get(sol, :converged, missing),
    )
end

function failure_row(system_info, rep, T, d, action_dim, method, rank, err,
        matrix_time, matrix_bytes)
    rss = Sys.maxrss()
    return Dict{String,Any}(
        "system" => system_info.system,
        "system_seed" => system_info.system_seed,
        "rho_A" => system_info.rho_A,
        "n" => system_info.n,
        "p" => system_info.p,
        "rep" => rep,
        "T" => T,
        "d" => d,
        "action_dim" => action_dim,
        "method" => String(method),
        "method_label" => method_label(method, rank),
        "status" => "failed: $(typeof(err)): $(err)",
        "rank" => rank === missing ? missing : rank,
        "binary_objective" => missing,
        "normalized_binary_quality" => missing,
        "quality_ref_binary_objective" => missing,
        "continuous_objective" => missing,
        "best_continuous_objective" => missing,
        "relaxed_objective" => missing,
        "solve_seconds" => missing,
        "allocated_bytes" => missing,
        "gc_seconds" => missing,
        "maxrss_bytes" => rss,
        "maxrss_mb" => rss / 2.0^20,
        "matrix_build_seconds" => matrix_time,
        "matrix_allocated_bytes" => matrix_bytes,
        "restarts" => missing,
        "round_restarts" => missing,
        "iterations" => missing,
        "best_iteration" => missing,
        "best_restart" => missing,
        "grad_norm" => missing,
        "converged" => false,
    )
end

function work_count_label(row)
    method = row["method"]
    starts = row["restarts"]
    rounds = row["round_restarts"]
    if method == "sdp_gw"
        return "rounds=$(starts)"
    elseif method == "low_rank"
        return "starts=$(starts) rounds=$(rounds)"
    end
    return "starts=$(starts)"
end

function append_anytime_rows!(rows, system_info, rep, T, d, method, rank, sol)
    times = nt_get(sol, :trace_time, Float64[])
    values = nt_get(sol, :trace_value, Float64[])
    for (idx, (t, val)) in enumerate(zip(times, values))
        push!(rows, Dict{String,Any}(
            "system" => system_info.system,
            "system_seed" => system_info.system_seed,
            "rho_A" => system_info.rho_A,
            "n" => system_info.n,
            "p" => system_info.p,
            "rep" => rep,
            "T" => T,
            "d" => d,
            "method" => String(method),
            "method_label" => method_label(method, rank),
            "rank" => rank === missing ? missing : rank,
            "trace_index" => idx,
            "wall_seconds" => t,
            "best_binary_objective" => val,
        ))
    end
    return rows
end

function add_normalized_quality!(rows)
    groups = Dict{Tuple{Any,Any,Any},Vector{Dict{String,Any}}}()
    for row in rows
        key = (row["system"], row["rep"], row["T"])
        push!(get!(groups, key, Dict{String,Any}[]), row)
    end

    for group_rows in values(groups)
        vals = Float64[]
        for row in group_rows
            val = row["binary_objective"]
            if val isa Real && isfinite(Float64(val))
                push!(vals, Float64(val))
            end
        end
        isempty(vals) && continue
        ref = maximum(vals)
        for row in group_rows
            row["quality_ref_binary_objective"] = ref
            val = row["binary_objective"]
            row["normalized_binary_quality"] =
                (val isa Real && ref > 0) ? Float64(val) / ref : missing
        end
    end
    return rows
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
    columns = copy(QUBO_COLUMNS)
    for row in rows
        for key in keys(row)
            if !(key in columns)
                push!(columns, key)
            end
        end
    end
    open(path, "w") do io
        println(io, join(columns, ","))
        for row in rows
            println(io, join((csv_cell(get(row, col, missing)) for col in columns), ","))
        end
    end
    return path
end

function write_anytime_csv(path::AbstractString, rows::Vector{Dict{String,Any}})
    dir = dirname(path)
    isempty(dir) || mkpath(dir)
    open(path, "w") do io
        println(io, join(ANYTIME_COLUMNS, ","))
        for row in rows
            println(io, join((csv_cell(get(row, col, missing)) for col in ANYTIME_COLUMNS), ","))
        end
    end
    return path
end

xml_escape(s) = replace(string(s), "&" => "&amp;", "<" => "&lt;", ">" => "&gt;")

function finite_real(x)
    return x isa Real && isfinite(Float64(x))
end

function summarize_series(rows, xkey, ykey; filter_row=_ -> true,
        label_row=row -> row["method_label"])
    grouped = Dict{String,Dict{Float64,Vector{Float64}}}()
    for row in rows
        filter_row(row) || continue
        finite_real(row[xkey]) || continue
        finite_real(row[ykey]) || continue
        label = string(label_row(row))
        x = Float64(row[xkey])
        y = Float64(row[ykey])
        by_x = get!(grouped, label, Dict{Float64,Vector{Float64}}())
        push!(get!(by_x, x, Float64[]), y)
    end

    series = NamedTuple[]
    for label in sort(collect(keys(grouped)); by=label_sort_key)
        by_x = grouped[label]
        xs = sort(collect(keys(by_x)))
        ys = [mean(by_x[x]) for x in xs]
        push!(series, (label=label, xs=xs, ys=ys))
    end
    return series
end

function label_sort_key(label)
    text = string(label)
    text == "SDP+GW" && return 10
    text == "Sign-Iteration" && return 20
    text == "Box-PGA" && return 30
    m = match(r"LR-SDP r=(\d+)", text)
    if m !== nothing
        return 100 + parse(Int, m.captures[1])
    end
    m = match(r"T=(\d+)", text)
    if m !== nothing
        return parse(Int, m.captures[1])
    end
    return 10_000
end

function nice_num(x)
    if abs(x) >= 1000 || (abs(x) > 0 && abs(x) < 0.01)
        return @sprintf("%.1e", x)
    end
    return @sprintf("%.3g", x)
end

function write_line_svg(path, title, xlabel, ylabel, series; logy=false,
        ymin=nothing, ymax=nothing)
    isempty(series) && return nothing
    mkpath(dirname(path))

    all_x = reduce(vcat, [s.xs for s in series])
    all_y = reduce(vcat, [s.ys for s in series])
    if logy
        all_y = [y for y in all_y if y > 0]
    end
    if isempty(all_x) || isempty(all_y)
        return nothing
    end

    x0 = minimum(all_x)
    x1 = maximum(all_x)
    if x0 == x1
        x0 -= 1
        x1 += 1
    end

    raw_y0 = isnothing(ymin) ? minimum(all_y) : ymin
    raw_y1 = isnothing(ymax) ? maximum(all_y) : ymax
    if raw_y0 == raw_y1
        raw_y0 -= max(1.0, abs(raw_y0) * 0.1)
        raw_y1 += max(1.0, abs(raw_y1) * 0.1)
    end
    if !logy
        pad = 0.06 * (raw_y1 - raw_y0)
        raw_y0 -= pad
        raw_y1 += pad
    else
        raw_y0 = max(raw_y0, minimum(all_y))
    end

    yscale(y) = logy ? log10(max(y, raw_y0)) : y
    y0 = yscale(raw_y0)
    y1 = yscale(raw_y1)
    if y0 == y1
        y0 -= 1
        y1 += 1
    end

    width = 900
    height = 560
    left = 82
    right = 210
    top = 54
    bottom = 76
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = [
        "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
        "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
    ]

    sx(x) = left + (x - x0) / (x1 - x0) * plot_w
    sy(y) = top + (y1 - yscale(y)) / (y1 - y0) * plot_h

    open(path, "w") do io
        println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$width" height="$height" viewBox="0 0 $width $height">""")
        println(io, """<rect width="100%" height="100%" fill="white"/>""")
        println(io, """<text x="$(left)" y="30" font-family="Arial" font-size="20" font-weight="700">$(xml_escape(title))</text>""")
        println(io, """<line x1="$left" y1="$(top + plot_h)" x2="$(left + plot_w)" y2="$(top + plot_h)" stroke="#333" stroke-width="1.2"/>""")
        println(io, """<line x1="$left" y1="$top" x2="$left" y2="$(top + plot_h)" stroke="#333" stroke-width="1.2"/>""")

        for t in 0:4
            x = x0 + t * (x1 - x0) / 4
            px = sx(x)
            println(io, """<line x1="$px" y1="$(top + plot_h)" x2="$px" y2="$(top + plot_h + 5)" stroke="#333"/>""")
            println(io, """<text x="$px" y="$(top + plot_h + 24)" text-anchor="middle" font-family="Arial" font-size="12">$(nice_num(x))</text>""")
        end

        for t in 0:4
            yy = y0 + t * (y1 - y0) / 4
            raw = logy ? 10.0^yy : yy
            py = top + (y1 - yy) / (y1 - y0) * plot_h
            println(io, """<line x1="$(left - 5)" y1="$py" x2="$left" y2="$py" stroke="#333"/>""")
            println(io, """<line x1="$left" y1="$py" x2="$(left + plot_w)" y2="$py" stroke="#e8e8e8"/>""")
            println(io, """<text x="$(left - 10)" y="$(py + 4)" text-anchor="end" font-family="Arial" font-size="12">$(nice_num(raw))</text>""")
        end

        println(io, """<text x="$(left + plot_w / 2)" y="$(height - 22)" text-anchor="middle" font-family="Arial" font-size="14">$(xml_escape(xlabel))</text>""")
        println(io, """<text transform="translate(22,$(top + plot_h / 2)) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="14">$(xml_escape(ylabel))</text>""")

        for (idx, s) in enumerate(series)
            color = colors[mod1(idx, length(colors))]
            points = String[]
            for (x, y) in zip(s.xs, s.ys)
                logy && y <= 0 && continue
                push!(points, @sprintf("%.2f,%.2f", sx(x), sy(y)))
            end
            isempty(points) && continue
            println(io, """<polyline fill="none" stroke="$color" stroke-width="2.2" points="$(join(points, " "))"/>""")
            for (x, y) in zip(s.xs, s.ys)
                logy && y <= 0 && continue
                println(io, """<circle cx="$(sx(x))" cy="$(sy(y))" r="3.2" fill="$color"/>""")
            end
            legend_y = top + 22 * (idx - 1)
            println(io, """<line x1="$(left + plot_w + 28)" y1="$legend_y" x2="$(left + plot_w + 48)" y2="$legend_y" stroke="$color" stroke-width="2.2"/>""")
            println(io, """<text x="$(left + plot_w + 56)" y="$(legend_y + 4)" font-family="Arial" font-size="12">$(xml_escape(s.label))</text>""")
        end

        println(io, "</svg>")
    end
    return path
end

function write_grouped_bar_svg(path, title, xlabel, ylabel, series)
    isempty(series) && return nothing
    mkpath(dirname(path))

    x_values = sort(unique(reduce(vcat, [s.xs for s in series])))
    isempty(x_values) && return nothing

    by_label = Dict(s.label => Dict(zip(s.xs, s.ys)) for s in series)
    all_y = Float64[]
    for s in series
        append!(all_y, s.ys)
    end
    isempty(all_y) && return nothing

    y0 = 0.0
    y1 = maximum(all_y)
    y1 <= 0 && (y1 = 1.0)
    y1 *= 1.08

    width = 980
    height = 560
    left = 82
    right = 220
    top = 54
    bottom = 86
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = [
        "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
        "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
    ]

    group_w = plot_w / length(x_values)
    bar_gap = 2.0
    bar_w = max(2.0, 0.78 * group_w / length(series) - bar_gap)
    sy(y) = top + (y1 - y) / (y1 - y0) * plot_h

    open(path, "w") do io
        println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$width" height="$height" viewBox="0 0 $width $height">""")
        println(io, """<rect width="100%" height="100%" fill="white"/>""")
        println(io, """<text x="$(left)" y="30" font-family="Arial" font-size="20" font-weight="700">$(xml_escape(title))</text>""")
        println(io, """<line x1="$left" y1="$(top + plot_h)" x2="$(left + plot_w)" y2="$(top + plot_h)" stroke="#333" stroke-width="1.2"/>""")
        println(io, """<line x1="$left" y1="$top" x2="$left" y2="$(top + plot_h)" stroke="#333" stroke-width="1.2"/>""")

        for t in 0:4
            y = y0 + t * (y1 - y0) / 4
            py = sy(y)
            println(io, """<line x1="$(left - 5)" y1="$py" x2="$left" y2="$py" stroke="#333"/>""")
            println(io, """<line x1="$left" y1="$py" x2="$(left + plot_w)" y2="$py" stroke="#e8e8e8"/>""")
            println(io, """<text x="$(left - 10)" y="$(py + 4)" text-anchor="end" font-family="Arial" font-size="12">$(nice_num(y))</text>""")
        end

        for (x_idx, x) in enumerate(x_values)
            group_left = left + (x_idx - 1) * group_w
            group_center = group_left + group_w / 2
            println(io, """<text x="$group_center" y="$(top + plot_h + 24)" text-anchor="middle" font-family="Arial" font-size="12">$(nice_num(x))</text>""")

            bars_total_w = length(series) * bar_w + (length(series) - 1) * bar_gap
            start_x = group_center - bars_total_w / 2
            for (series_idx, s) in enumerate(series)
                y = get(by_label[s.label], x, missing)
                y === missing && continue
                color = colors[mod1(series_idx, length(colors))]
                bx = start_x + (series_idx - 1) * (bar_w + bar_gap)
                by = sy(y)
                bh = top + plot_h - by
                println(io, """<rect x="$bx" y="$by" width="$bar_w" height="$bh" fill="$color"/>""")
            end
        end

        println(io, """<text x="$(left + plot_w / 2)" y="$(height - 24)" text-anchor="middle" font-family="Arial" font-size="14">$(xml_escape(xlabel))</text>""")
        println(io, """<text transform="translate(22,$(top + plot_h / 2)) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="14">$(xml_escape(ylabel))</text>""")

        for (idx, s) in enumerate(series)
            color = colors[mod1(idx, length(colors))]
            legend_y = top + 22 * (idx - 1)
            println(io, """<rect x="$(left + plot_w + 28)" y="$(legend_y - 10)" width="18" height="12" fill="$color"/>""")
            println(io, """<text x="$(left + plot_w + 56)" y="$(legend_y + 1)" font-family="Arial" font-size="12">$(xml_escape(s.label))</text>""")
        end

        println(io, "</svg>")
    end
    return path
end

function write_plots(plot_dir, rows, rank_Ts)
    mkpath(plot_dir)
    quality = summarize_series(rows, "T", "binary_objective")
    write_line_svg(joinpath(plot_dir, "figure1_quality_vs_T.svg"),
        "Final Binary Objective vs T", "Commit horizon T", "x'Wx",
        quality)

    runtime = summarize_series(rows, "T", "solve_seconds")
    write_line_svg(joinpath(plot_dir, "figure2_runtime_vs_T.svg"),
        "Runtime vs T", "T", "seconds (log scale)", runtime; logy=true)

    peak = summarize_series(rows, "T", "maxrss_mb")
    write_grouped_bar_svg(joinpath(plot_dir, "figure3_peak_memory_vs_T.svg"),
        "Peak Memory Usage vs T", "T", "peak memory (MB)", peak)

    rank_set = Set(Float64.(rank_Ts))
    rank_rows = [row for row in rows if row["method"] == "low_rank" &&
        finite_real(row["T"]) && Float64(row["T"]) in rank_set]
    rank_quality = summarize_series(rank_rows, "rank", "binary_objective";
        label_row=row -> "T=$(Int(row["T"]))")
    write_line_svg(joinpath(plot_dir, "figure4_rank_quality.svg"),
        "LR-SDP Rank Ablation", "rank r", "x'Wx", rank_quality)

    rank_runtime = summarize_series(rank_rows, "rank", "solve_seconds";
        label_row=row -> "T=$(Int(row["T"]))")
    write_line_svg(joinpath(plot_dir, "figure4_rank_runtime.svg"),
        "LR-SDP Runtime vs Rank", "rank r", "seconds", rank_runtime)
    return plot_dir
end

function run_anytime_benchmark(parsed, system, system_info, seed, rep, Ts)
    ranks = parse_int_list(opt(parsed, "anytime-ranks", "4,8"))
    rows = Dict{String,Any}[]

    cases = Tuple{Symbol,Any}[
        (:sign_iteration, missing),
        (:box_pgd, missing),
    ]
    append!(cases, [(:low_rank, rank) for rank in ranks])

    for T in Ts
        matrix_timed = @timed build_true_qubo_matrix(system, T)
        W = matrix_timed.value
        d = size(W, 1)

        for (case_idx, (method, rank)) in enumerate(cases)
            rng_seed = seed + 9_000_000 * rep + 10_000 * T + 1_003 * case_idx +
                (rank === missing ? 0 : rank)
            rng = Random.Xoshiro(rng_seed)
            try
                sol = solve_method(method, W, parsed, rank, rng, T; trace=true)
                append_anytime_rows!(rows, system_info, rep, T, d, method, rank, sol)
                @printf("anytime %-14s rep=%d T=%4d points=%d final=% .6e\n",
                    method_label(method, rank), rep, T,
                    length(nt_get(sol, :trace_value, Float64[])),
                    Float64(nt_get(sol, :binary_value, nt_get(sol, :rounded_value, NaN))))
            catch err
                @warn "anytime case failed" rep T method rank err
            end
            GC.gc()
        end
    end
    return rows
end

function write_anytime_plots(plot_dir, rows)
    isempty(rows) && return nothing
    grouped = Dict{Int,Vector{Dict{String,Any}}}()
    for row in rows
        T = Int(row["T"])
        push!(get!(grouped, T, Dict{String,Any}[]), row)
    end

    for T in sort(collect(keys(grouped)))
        series = summarize_series(grouped[T], "wall_seconds", "best_binary_objective")
        write_line_svg(joinpath(plot_dir, "figure5_anytime_T$(T).svg"),
            "Anytime Performance at T=$(T)", "wall-clock seconds",
            "best binary objective so far", series)
    end
    return plot_dir
end

function warmup_methods(parsed, methods, ranks, system, seed)
    haskey(parsed, "no-warmup") && return nothing
    W = build_true_qubo_matrix(system, 4)
    for method in methods
        method == :sdp_gw && continue
        method_ranks = method == :low_rank ? [minimum(ranks)] : [missing]
        for rank in method_ranks
            rng = Random.Xoshiro(seed + 77 + (rank === missing ? 0 : rank))
            try
                solve_method(method, W, parsed, rank, rng, 4)
            catch err
                @warn "warmup failed" method rank err
            end
        end
    end
    GC.gc()
    return nothing
end

function main(args)
    parsed = parse_args(args)
    Ts = parse_int_list(opt(parsed, "Ts", "200:200:4000"))
    ranks = parse_int_list(opt(parsed, "ranks", "2,4,6,8"))
    rank_Ts = parse_int_list(opt(parsed, "rank-Ts", "1000,2000,3000,4000"))
    methods = parse_methods(opt(parsed, "methods",
        "sign_iteration,box_pgd,low_rank,sdp_gw"))
    sdp_max_T = opt_int(parsed, "sdp-max-T", 2000)
    reps = opt_int(parsed, "reps", 1)
    seed = opt_int(parsed, "seed", 1)
    output = opt(parsed, "output", joinpath(@__DIR__, "..", "results",
        "qubo_benchmark.csv"))
    output_stem = splitext(basename(output))[1]
    plot_dir = opt(parsed, "plot-dir", joinpath(dirname(output),
        "$(output_stem)_plots"))
    anytime_output = opt(parsed, "anytime-output",
        joinpath(dirname(output), "$(output_stem)_anytime_trace.csv"))

    system, system_info = make_benchmark_system(parsed, seed)
    _, action_dim = size(system.B)
    rows = Dict{String,Any}[]

    @printf("system=%s n=%d p=%d rho_A=%.4f seed=%s\n",
        system_info.system, system_info.n, system_info.p,
        Float64(system_info.rho_A), string(system_info.system_seed))

    warmup_methods(parsed, methods, ranks, system, seed)

    for rep in 1:reps
        for T in Ts
            matrix_timed = @timed build_true_qubo_matrix(system, T)
            W = matrix_timed.value
            matrix_time = matrix_timed.time
            matrix_bytes = matrix_timed.bytes
            d = size(W, 1)

            for method in methods
                if method == :sdp_gw && T > sdp_max_T
                    @printf("skip sdp_gw T=%d above --sdp-max-T=%d\n", T, sdp_max_T)
                    continue
                end

                method_ranks = method == :low_rank ? ranks : [missing]
                for rank in method_ranks
                    rng_seed = seed + 1_000_000 * rep + 10_000 * T +
                        101 * findfirst(==(method), methods) +
                        (rank === missing ? 0 : rank)
                    rng = Random.Xoshiro(rng_seed)
                    try
                        timed = @timed solve_method(method, W, parsed, rank, rng, T)
                        push!(rows, success_row(system_info, rep, T, action_dim,
                            method, rank, timed.value, timed, matrix_time,
                            matrix_bytes))
                        row = rows[end]
                        @printf("%-14s T=%4d d=%4d %s obj=% .6e sec=%.3f rss=%s\n",
                            row["method_label"], T, d,
                            work_count_label(row),
                            Float64(row["binary_objective"]),
                            Float64(row["solve_seconds"]),
                            string(row["maxrss_bytes"]))
                    catch err
                        push!(rows, failure_row(system_info, rep, T, d, action_dim,
                            method, rank, err, matrix_time, matrix_bytes))
                        @warn "benchmark case failed" rep T method rank err
                    end
                    GC.gc()
                end
            end
        end
    end

    add_normalized_quality!(rows)
    write_csv(output, rows)
    @printf("wrote %d rows to %s\n", length(rows), output)

    if !haskey(parsed, "no-plots")
        write_plots(plot_dir, rows, rank_Ts)
        if !haskey(parsed, "no-anytime")
            anytime_Ts = haskey(parsed, "anytime-T") ?
                parse_int_list(parsed["anytime-T"]) : Ts
            anytime_rows = run_anytime_benchmark(parsed, system, system_info,
                seed, 1, anytime_Ts)
            write_anytime_csv(anytime_output, anytime_rows)
            write_anytime_plots(plot_dir, anytime_rows)
            @printf("wrote anytime trace to %s\n", anytime_output)
        end
        @printf("wrote plots to %s\n", plot_dir)
    end
end

main(ARGS)
