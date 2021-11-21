push!(LOAD_PATH,"../")

using Pkg; Pkg.add("Documenter")

using Documenter, DomDecOT

makedocs(
    modules = [DomDecOT],
    sitename="DomDecOT.jl",
    format = Documenter.HTML(
        # Use clean URLs, unless built as a "local" build
        prettyurls = !("local" in ARGS),
        canonical = "https://ismedina.github.io/DomDecOT.jl/",
        assets = ["assets/favicon.ico"],
        analytics = "UA-136089579-2",
        highlights = ["yaml"],
        ansicolor = true,
    ),
    clean=true,
    repo="github.com/ismedina/DomDecOT.jl.git",
    expandfirst= [],
    authors="Ismael Medina, Mauro Bonafini and Bernhard Schmitzer",
    pages = [
        "Home" => "index.md",
        "Manual" => Any[
            "Library" => "library.md",
            "Internals" => "internals.md"
        ]
    ]
)

deploydocs(
    repo = "github.com/ismedina/DomDecOT.jl.git"
)
