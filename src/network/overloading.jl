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
    grad_h::Vector{Float64}
    grad_x::Vector{Float64}
    grad_w::Matrix{Float64}
    grad_w_rec::Matrix{Float64}
    grad_b::Vector{Float64}
    grad_h_prev::Vector{Float64}
end

function init_gradients(input_size, output_size)
    grad_h = zeros(output_size)
    grad_x = zeros(input_size)
    grad_w = zeros(output_size, input_size)
    grad_w_rec = zeros(output_size, output_size)
    grad_b = zeros(output_size)
    grad_h_prev = zeros(output_size)
    
    return Gradients(grad_h, grad_x, grad_w, grad_w_rec, grad_b, grad_h_prev)
end

# Optimized backward function
import LinearAlgebra: mul!
backward(op::BroadcastedOperator{typeof(rnn)}, x, w, w_rec, b, h_prev, g) = let
    h = forward(op, x, w, w_rec, b, h_prev)  # OPTIMIZATION
    global grads
    grads.grad_h = g .* (1 .- h.^2)

    mul!(grads.grad_x, w', grads.grad_h)
    mul!(grads.grad_w, grads.grad_h, x')
    mul!(grads.grad_h_prev, w_rec', grads.grad_h)
    grads.grad_w_rec = grads.grad_h * h_prev'
    grads.grad_b = sum(grads.grad_h, dims=2)
 
    return tuple(grads.grad_x, grads.grad_w, grads.grad_w_rec, grads.grad_b, grads.grad_h_prev)
end
