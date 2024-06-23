include("./graph/nodes.jl")
import Base: tanh

dense(x::GraphNode, w::GraphNode, b::GraphNode) = BroadcastedOperator(dense, x, w, b)
forward(::BroadcastedOperator{typeof(dense)}, x, w, b) = w * x + b
backward(::BroadcastedOperator{typeof(dense)}, x, w, b, g) = tuple(w' * g, g * x', g)

tanh(x::GraphNode) = BroadcastedOperator(tanh, x)
forward(::BroadcastedOperator{typeof(tanh)}, x) = return tanh.(x)
backward(::BroadcastedOperator{typeof(tanh)}, x, g) = return tuple(g .* (1 .- tanh.(x).^2))

identity(x::GraphNode) = BroadcastedOperator(identity, x)
forward(::BroadcastedOperator{typeof(identity)}, x) = x
backward(::BroadcastedOperator{typeof(identity)}, x, g) = tuple(g)

function rnn(x::GraphNode, w::GraphNode, w_rec::GraphNode, b::GraphNode, h_prev::GraphNode)
    return BroadcastedOperator(rnn, x, w, w_rec, b, h_prev)
end

function forward(::BroadcastedOperator{typeof(rnn)}, x, w, w_rec, b, h_prev)
    return (w * x .+ w_rec * h_prev .+ b)
end

mutable struct Gradients
    grad_x::AbstractArray{Float64}
    grad_w::AbstractArray{Float64}
    grad_w_rec::AbstractArray{Float64}
    grad_b::AbstractArray{Float64}
    grad_h_prev::AbstractArray{Float64}
end

function init_gradients(x, w, w_rec)
    h_prev = zeros(64)

    grad_h = zeros(Float32, 64)
    grad_x = zeros(Float32, size(w.output, 2), size(grad_h, 2))
    grad_w = zeros(Float32, size(h_prev, 1), size(x.output, 1))
    grad_w_rec = zeros(Float64, size(h_prev, 1), size(h_prev, 1))

    grad_b = zeros(Float64, size(h_prev, 1), 1)
    grad_h_prev = zeros(Float64, size(w_rec.output, 2), size(grad_h, 2))
    
    return Gradients(grad_x, grad_w, grad_w_rec, grad_b, grad_h_prev)
end

function reset_gradients!(grads::Gradients)
    grads.grad_x .= 0
    grads.grad_w .= 0
    grads.grad_w_rec .= 0
    grads.grad_b .= 0
    grads.grad_h_prev .= 0
end

# Optimized backward function
import LinearAlgebra: mul!

backward(op::BroadcastedOperator{typeof(rnn)}, x, w, w_rec, b, h_prev, g) = let
    h = forward(op, x, w, w_rec, b, h_prev)  # OPTIMIZATION
    global grads
    grad_h = g .* (1 .- h.^2)

    mul!(grads.grad_x, w', grad_h)
    mul!(grads.grad_w, grad_h, x')

    grads.grad_w_rec = grad_h * h_prev'
    grads.grad_b = sum(grad_h, dims=2)
    grads.grad_h_prev = w_rec' * grad_h
 
    return tuple(grads.grad_x, grads.grad_w, grads.grad_w_rec, grads.grad_b, grads.grad_h_prev)
end
