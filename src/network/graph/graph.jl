function build_graph(x_t1, x_t2, x_t3, x_t4, train_y, rnn_weights, rnn_recurrent_weights, rnn_bias, dense_weights, dense_bias)
    h_prev = Constant(zeros(64))  
   
    h = h_prev
    for x_t in [x_t1, x_t2, x_t3, x_t4]
        h = rnn(x_t, rnn_weights, rnn_recurrent_weights, rnn_bias, h) |> tanh
    end

    y_hat = dense(h, dense_weights, dense_bias) |> identity
    e = cross_entropy_loss(y_hat, train_y)

    return topological_sort(e), y_hat
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