#!/usr/bin/env julia

using Printf
using Random
using Statistics

const ROOT = normpath(joinpath(@__DIR__, ".."))
const DEFAULT_INPUT_DIR = joinpath(ROOT, "results", "ablation_studies")
const DEFAULT_OUTPUT_DIR = joinpath(ROOT, "results", "ablation_studies")

const GROUP_METHODS = ["Sign-Iteration", "Box-PGA", "LR-SDP r=2", "LR-SDP r=4"]
const ROUNDING_METHODS = ["LR-SDP r=2", "LR-SDP r=4"]
const COLORS = Dict(
    "Sign-Iteration" => "#4E79A7",
    "Box-PGA" => "#F28E2B",
    "LR-SDP r=2" => "#59A14F",
    "LR-SDP r=4" => "#B07AA1",
    "SDP+GW" => "#8C564B",
)

function print_usage()
    println("""
Usage:
  julia --project=. scripts/plot_ablation_studies.jl [options]

Options:
  --input-dir DIR             Directory containing solver_results.csv and rounding_trace.csv.
                              Default: $DEFAULT_INPUT_DIR
  --output-dir DIR            Directory for summary CSV and SVG outputs.
                              Default: $DEFAULT_OUTPUT_DIR
  --T INT                     Planning horizon to plot. Default: 2000
  --group-count INT           Number of random groups per group size. Default: 8
  --group-max-size INT        Maximum group size for group-size ablation. Default: 32
  --rounding-group-size INT   Seed group size for rounding ablation. Default: 8
  --sampling-seed INT         RNG seed for reproducible group sampling. Default: 20260512
""")
end

function parse_args(args)
    opts = Dict{String,Any}(
        "input-dir" => DEFAULT_INPUT_DIR,
        "output-dir" => DEFAULT_OUTPUT_DIR,
        "T" => 2000,
        "group-count" => 8,
        "group-max-size" => 32,
        "rounding-group-size" => 8,
        "sampling-seed" => 20260512,
    )
    int_keys = Set(["T", "group-count", "group-max-size", "rounding-group-size", "sampling-seed"])

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("-h", "--help")
            print_usage()
            exit(0)
        elseif startswith(arg, "--")
            key = arg[3:end]
            haskey(opts, key) || error("Unknown option --$key")
            i < length(args) || error("Missing value for --$key")
            raw = args[i + 1]
            opts[key] = key in int_keys ? parse(Int, raw) : raw
            i += 2
        else
            error("Unexpected argument: $arg")
        end
    end
    return opts
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

function read_csv(path::AbstractString)
    isfile(path) || error("Missing CSV: $path")
    lines = readlines(path)
    isempty(lines) && return Dict{String,String}[]
    header = csv_split(lines[1])
    rows = Dict{String,String}[]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        values = csv_split(line)
        row = Dict{String,String}()
        for (i, key) in enumerate(header)
            row[key] = get(values, i, "")
        end
        push!(rows, row)
    end
    return rows
end

function csv_cell(x)
    s = string(x)
    if occursin(",", s) || occursin("\"", s) || occursin("\n", s)
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

function write_csv(path::AbstractString, columns, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(columns, ","))
        for row in rows
            println(io, join((csv_cell(get(row, col, "")) for col in columns), ","))
        end
    end
    return path
end

as_int(row, key) = parse(Int, row[key])
as_float(row, key) = parse(Float64, row[key])

function solver_rows(solver, T::Int, label::AbstractString)
    return [row for row in solver
            if get(row, "method_label", "") == label &&
               get(row, "T", "") != "" &&
               as_int(row, "T") == T &&
               !startswith(get(row, "status", ""), "failed:")]
end

function rounding_rows(rounding, T::Int, label::AbstractString)
    return [row for row in rounding
            if get(row, "method_label", "") == label &&
               get(row, "T", "") != "" &&
               as_int(row, "T") == T]
end

function sdp_reference(solver, T::Int)
    rows = solver_rows(solver, T, "SDP+GW")
    isempty(rows) && error("No SDP+GW solver row found for T=$T")
    return maximum(as_float(row, "binary_objective") for row in rows)
end

function common_seed_indices(solver, T::Int, labels)
    seed_sets = Set{Int}[]
    for label in labels
        rows = solver_rows(solver, T, label)
        isempty(rows) && error("No rows found for $label at T=$T")
        push!(seed_sets, Set(as_int(row, "seed_index") for row in rows
                             if get(row, "seed_index", "") != ""))
    end
    common = intersect(seed_sets...)
    isempty(common) && error("No common seed indices across requested methods")
    return sort!(collect(common))
end

function values_by_seed(rows, value_col::AbstractString)
    lookup = Dict{Int,Float64}()
    for row in rows
        get(row, "seed_index", "") == "" && continue
        value_text = get(row, value_col, "")
        value_text == "" && continue
        value = tryparse(Float64, value_text)
        value === nothing && continue
        isfinite(value) || continue
        lookup[as_int(row, "seed_index")] = value
    end
    return lookup
end

function random_groups(seed_indices, group_count::Int, group_size::Int, rng)
    needed = group_count * group_size
    needed <= length(seed_indices) ||
        error("Need $needed seeds for $group_count groups of size $group_size, but only $(length(seed_indices)) are available.")
    sampled = shuffle(rng, collect(seed_indices))[1:needed]
    return [sampled[((g - 1) * group_size + 1):(g * group_size)] for g in 1:group_count]
end

function group_max_stats(rows, groups, value_col::AbstractString)
    lookup = values_by_seed(rows, value_col)
    maxima = Float64[]
    for group in groups
        missing = [seed for seed in group if !haskey(lookup, seed)]
        isempty(missing) || error("Missing $value_col for seed indices $(join(missing, " "))")
        push!(maxima, maximum(lookup[seed] for seed in group))
    end
    return mean(maxima), length(maxima) > 1 ? std(maxima) : 0.0, maxima
end

function xml_escape(s)
    return replace(string(s), "&" => "&amp;", "<" => "&lt;", ">" => "&gt;",
        "\"" => "&quot;")
end

function pct_num(x)
    return @sprintf("%.0f%%", 100 * Float64(x))
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

function y_limits(series)
    vals = Float64[]
    for s in series
        for i in eachindex(s.ys)
            std_i = s.band ? s.stds[i] : 0.0
            push!(vals, s.ys[i] - std_i)
            push!(vals, s.ys[i] + std_i)
        end
    end
    isempty(vals) && return 0.0, 1.0
    y0 = max(0.0, floor((minimum(vals) - 0.01) * 100) / 100)
    y1 = min(1.02, ceil((maximum(vals) + 0.01) * 100) / 100)
    if y1 - y0 < 0.04
        center = 0.5 * (y0 + y1)
        y0 = max(0.0, center - 0.02)
        y1 = min(1.02, center + 0.02)
    end
    return y0, y1
end

function write_quality_svg(path, title, xlabel, ylabel, series; xticks=nothing)
    width = 1080
    height = 640
    left = 92
    right = 58
    top = 66
    bottom = 86
    plot_w = width - left - right
    plot_h = height - top - bottom

    xs_all = Float64[]
    for s in series
        append!(xs_all, s.xs)
    end
    isempty(xs_all) && error("No data to plot for $title")
    x0 = minimum(xs_all)
    x1 = maximum(xs_all)
    x1 == x0 && (x1 = x0 + 1.0)
    y0, y1 = y_limits(series)

    sx(x) = left + (Float64(x) - x0) / (x1 - x0) * plot_w
    sy(y) = top + (y1 - Float64(y)) / (y1 - y0) * plot_h
    ytick_values = collect(range(y0, y1; length=6))
    xtick_values = xticks === nothing ? collect(range(x0, x1; length=6)) : xticks

    open(path, "w") do io
        println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$width" height="$height" viewBox="0 0 $width $height">""")
        println(io, """<rect width="100%" height="100%" fill="white"/>""")
        println(io, """<text x="$(left + plot_w / 2)" y="36" text-anchor="middle" font-family="Arial" font-size="24" font-weight="700">$(xml_escape(title))</text>""")

        for y in ytick_values
            py = sy(y)
            println(io, """<line x1="$left" y1="$py" x2="$(left + plot_w)" y2="$py" stroke="#e7e7e7"/>""")
            println(io, """<text x="$(left - 12)" y="$(py + 5)" text-anchor="end" font-family="Arial" font-size="13">$(pct_num(y))</text>""")
        end
        for x in xtick_values
            px = sx(x)
            println(io, """<line x1="$px" y1="$top" x2="$px" y2="$(top + plot_h)" stroke="#f0f0f0"/>""")
            println(io, """<line x1="$px" y1="$(top + plot_h)" x2="$px" y2="$(top + plot_h + 6)" stroke="#333"/>""")
            label = abs(x - round(x)) < 1e-9 ? @sprintf("%.0f", x) : @sprintf("%.1f", x)
            println(io, """<text x="$px" y="$(top + plot_h + 27)" text-anchor="middle" font-family="Arial" font-size="13">$label</text>""")
        end

        println(io, """<line x1="$left" y1="$(top + plot_h)" x2="$(left + plot_w)" y2="$(top + plot_h)" stroke="#333" stroke-width="1.2"/>""")
        println(io, """<line x1="$left" y1="$top" x2="$left" y2="$(top + plot_h)" stroke="#333" stroke-width="1.2"/>""")

        for s in series
            if s.band
                upper = [(sx(s.xs[i]), sy(s.ys[i] + s.stds[i])) for i in eachindex(s.xs)]
                lower = reverse([(sx(s.xs[i]), sy(s.ys[i] - s.stds[i])) for i in eachindex(s.xs)])
                println(io, """<path d="$(make_path(vcat(upper, lower))) Z" fill="$(s.color)" opacity="0.14" stroke="none"/>""")
            end
            line_points = [(sx(s.xs[i]), sy(s.ys[i])) for i in eachindex(s.xs)]
            println(io, """<path d="$(make_path(line_points))" fill="none" stroke="$(s.color)" stroke-opacity="0.8" stroke-width="3.0" stroke-linejoin="round" stroke-linecap="round"/>""")
        end

        println(io, """<text x="$(left + plot_w / 2)" y="$(height - 26)" text-anchor="middle" font-family="Arial" font-size="16">$(xml_escape(xlabel))</text>""")
        println(io, """<text transform="translate(27,$(top + plot_h / 2)) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="16">$(xml_escape(ylabel))</text>""")

        legend_w = 190
        legend_h = 22 + 28 * length(series)
        legend_x = left + plot_w - legend_w - 16
        legend_y = top + 18
        println(io, """<rect x="$legend_x" y="$legend_y" width="$legend_w" height="$legend_h" rx="8" fill="white" opacity="0.88" stroke="#dddddd"/>""")
        for (idx, s) in enumerate(series)
            y = legend_y + 28 * idx
            println(io, """<line x1="$(legend_x + 15)" y1="$y" x2="$(legend_x + 50)" y2="$y" stroke="$(s.color)" stroke-opacity="0.8" stroke-width="3.0" stroke-linecap="round"/>""")
            println(io, """<text x="$(legend_x + 62)" y="$(y + 5)" font-family="Arial" font-size="13">$(xml_escape(s.label))</text>""")
        end

        println(io, "</svg>")
    end
    return path
end

function make_group_size_ablation(solver, output_dir, T::Int, group_count::Int,
                                  max_group_size::Int, sampling_seed::Int)
    ref = sdp_reference(solver, T)
    seeds = common_seed_indices(solver, T, GROUP_METHODS)
    rng = Random.Xoshiro(sampling_seed)
    group_sizes = collect(1:max_group_size)
    groups_by_size = Dict(size => random_groups(seeds, group_count, size, rng)
                          for size in group_sizes)

    records = Dict{String,Any}[]
    plot_series = Any[]
    for label in GROUP_METHODS
        rows = solver_rows(solver, T, label)
        means = Float64[]
        stds = Float64[]
        for size in group_sizes
            groups = groups_by_size[size]
            mean_obj, std_obj, _ = group_max_stats(rows, groups, "binary_objective")
            quality = mean_obj / ref
            quality_std = std_obj / ref
            push!(means, quality)
            push!(stds, quality_std)
            push!(records, Dict{String,Any}(
                "T" => T,
                "method_label" => label,
                "group_size" => size,
                "groups_used" => group_count,
                "sampling_seed" => sampling_seed,
                "sampled_seed_indices" => join(vcat(groups...), " "),
                "mean_group_max_objective" => mean_obj,
                "std_group_max_objective" => std_obj,
                "quality_vs_sdp_gw" => quality,
                "quality_std_vs_sdp_gw" => quality_std,
            ))
        end
        push!(plot_series, (
            label=label,
            xs=Float64.(group_sizes),
            ys=means,
            stds=stds,
            color=COLORS[label],
            band=true,
        ))
    end

    summary_path = joinpath(output_dir, "group_size_ablation.csv")
    write_csv(summary_path, [
        "T", "method_label", "group_size", "groups_used", "sampling_seed",
        "sampled_seed_indices", "mean_group_max_objective", "std_group_max_objective",
        "quality_vs_sdp_gw", "quality_std_vs_sdp_gw",
    ], records)

    svg_path = joinpath(output_dir, "group_size_ablation.svg")
    xticks = [x for x in [1, 4, 8, 16, 24, 32] if x <= max_group_size]
    write_quality_svg(svg_path, "T=$T Group-Size Ablation", "Seed group size",
        "Quality", plot_series; xticks=Float64.(xticks))
    return summary_path, svg_path
end

function rounding_groups(solver, T::Int, labels, group_size::Int, sampling_seed::Int)
    seeds = common_seed_indices(solver, T, labels)
    group_count = div(length(seeds), group_size)
    group_count >= 1 || error("Not enough seeds for rounding group size $group_size")
    rng = Random.Xoshiro(sampling_seed)
    return random_groups(seeds, group_count, group_size, rng)
end

function make_rounding_ablation(solver, rounding, output_dir, T::Int,
                                group_size::Int, sampling_seed::Int)
    ref = sdp_reference(solver, T)
    candidate_rounds = Int[]
    for label in vcat(ROUNDING_METHODS, ["SDP+GW"])
        append!(candidate_rounds, [as_int(row, "rounding_index")
                                   for row in rounding_rows(rounding, T, label)
                                   if get(row, "rounding_index", "") != ""])
    end
    isempty(candidate_rounds) && error("No rounding rows found for T=$T")
    rounds = collect(1:maximum(candidate_rounds))
    groups = rounding_groups(solver, T, ROUNDING_METHODS, group_size, sampling_seed)

    records = Dict{String,Any}[]
    plot_series = Any[]
    for label in ROUNDING_METHODS
        rows = rounding_rows(rounding, T, label)
        means = Float64[]
        stds = Float64[]
        for k in rounds
            rows_k = [row for row in rows if as_int(row, "rounding_index") == k]
            mean_obj, std_obj, _ = group_max_stats(rows_k, groups, "best_objective")
            quality = mean_obj / ref
            quality_std = std_obj / ref
            push!(means, quality)
            push!(stds, quality_std)
            push!(records, Dict{String,Any}(
                "T" => T,
                "method_label" => label,
                "group_size" => group_size,
                "groups_used" => length(groups),
                "roundings" => k,
                "sampling_seed" => sampling_seed,
                "mean_group_max_objective" => mean_obj,
                "std_group_max_objective" => std_obj,
                "quality_vs_sdp_gw" => quality,
                "quality_std_vs_sdp_gw" => quality_std,
            ))
        end
        push!(plot_series, (
            label=label,
            xs=Float64.(rounds),
            ys=means,
            stds=stds,
            color=COLORS[label],
            band=true,
        ))
    end

    sdp_by_round = Dict{Int,Float64}()
    for row in rounding_rows(rounding, T, "SDP+GW")
        get(row, "rounding_index", "") == "" && continue
        value = get(row, "best_objective", "") == "" ? NaN : as_float(row, "best_objective")
        isfinite(value) || continue
        sdp_by_round[as_int(row, "rounding_index")] = value
    end
    sdp_quality = Float64[]
    for k in rounds
        haskey(sdp_by_round, k) || error("Missing SDP+GW rounding row for k=$k")
        quality = sdp_by_round[k] / ref
        push!(sdp_quality, quality)
        push!(records, Dict{String,Any}(
            "T" => T,
            "method_label" => "SDP+GW",
            "group_size" => "",
            "groups_used" => 1,
            "roundings" => k,
            "sampling_seed" => sampling_seed,
            "mean_group_max_objective" => sdp_by_round[k],
            "std_group_max_objective" => 0.0,
            "quality_vs_sdp_gw" => quality,
            "quality_std_vs_sdp_gw" => 0.0,
        ))
    end
    push!(plot_series, (
        label="SDP+GW",
        xs=Float64.(rounds),
        ys=sdp_quality,
        stds=zeros(length(rounds)),
        color=COLORS["SDP+GW"],
        band=false,
    ))

    summary_path = joinpath(output_dir, "rounding_ablation_group$(group_size).csv")
    write_csv(summary_path, [
        "T", "method_label", "group_size", "groups_used", "roundings", "sampling_seed",
        "mean_group_max_objective", "std_group_max_objective",
        "quality_vs_sdp_gw", "quality_std_vs_sdp_gw",
    ], records)

    svg_path = joinpath(output_dir, "rounding_ablation_group$(group_size).svg")
    xticks = unique(vcat([1], collect(10:10:maximum(rounds)), [maximum(rounds)]))
    write_quality_svg(svg_path, "T=$T GW Rounding Ablation", "Number of GW roundings",
        "Quality", plot_series; xticks=Float64.(xticks))
    return summary_path, svg_path
end

function main()
    opts = parse_args(ARGS)
    input_dir = String(opts["input-dir"])
    output_dir = String(opts["output-dir"])
    mkpath(output_dir)

    solver = read_csv(joinpath(input_dir, "solver_results.csv"))
    rounding = read_csv(joinpath(input_dir, "rounding_trace.csv"))

    @printf("Loaded solver rows: %d\n", length(solver))
    @printf("Loaded rounding rows: %d\n", length(rounding))

    outputs = String[]
    append!(outputs, make_group_size_ablation(solver, output_dir, opts["T"],
        opts["group-count"], opts["group-max-size"], opts["sampling-seed"]))
    append!(outputs, make_rounding_ablation(solver, rounding, output_dir, opts["T"],
        opts["rounding-group-size"], opts["sampling-seed"]))

    println("Wrote:")
    for path in outputs
        println("  $path")
    end
end

main()
