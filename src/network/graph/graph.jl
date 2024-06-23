function build_graph(x_t1, x_t2, x_t3, x_t4, train_y, rnn_weights, rnn_recurrent_weights, rnn_bias, dense_weights, dense_bias, arch)
    h_prev = Constant(zeros(64))  
   
    l1 = rnn(x_t1, rnn_weights, rnn_recurrent_weights, rnn_bias, h_prev) |> tanh
    l2 = rnn(x_t2, rnn_weights, rnn_recurrent_weights, rnn_bias, l1) |> tanh
    l3 = rnn(x_t3, rnn_weights, rnn_recurrent_weights, rnn_bias, l2) |> tanh
    l4 = rnn(x_t4, rnn_weights, rnn_recurrent_weights, rnn_bias, l3) |> tanh

    l5 = dense(l4, dense_weights, dense_bias) |> identity
    e = cross_entropy_loss(l5, train_y)

    return topological_sort(e), l5
end

function update_weights!(graph::Vector, lr::Float64, batch_size::Int64)
    for node in graph
        if isa(node, Variable) && hasproperty(node, :batch_gradient)
			node.batch_gradient ./= batch_size
            node.output .-= lr * node.batch_gradient 
            fill!(node.batch_gradient, 0) ## OPTIMIZATION
        end
    end
end