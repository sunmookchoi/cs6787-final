struct BanditSystem{TA<:AbstractMatrix,TB<:AbstractMatrix,TC<:AbstractMatrix}
    A::TA
    B::TB
    C::TC
end

function make_simple_system(; rho::Real=0.3)
    A = Diagonal(Float64[rho, 0.5 * rho, 0.4 * rho])
    B = [1.0 0.0;
         0.0 1.0;
         0.5 0.4]
    C = [1.0 0.0 0.0;
         0.0 1.0 0.3]
    return BanditSystem(A, B, C)
end
