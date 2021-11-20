function Base.show(io::IO, M::DomDecPlan)
    println(io, "DomDecPlan with cellsize ", M.cellsize, " and marginals")
    print(io, "    mu: ")
    show(io, M.mu)
    print(io, "\n    nu: ")
    show(io, M.nu)
end