push!(LOAD_PATH,"../")

using Documenter, DomDecOT

makedocs(
    modules = [DomDecOT],
    sitename="DomDecOT.jl",
    authors="Ismael Medina, Mauro Bonafini and Bernhard Schmitzer",
    pages = [
        "Home" => "index.md",
        "Manual" => Any[
            "Library" => "library.md"
        ]
    ]
)

deploydocs(
    repo = "github.com/ismedina/DomDecOT.jl.git",
    target = "build",
    push_preview = true,
)
