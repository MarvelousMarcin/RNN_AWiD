mutable struct RNN_CUST
    epochs::Int64
    batch_size::Int64
    learning_rate::Float64
    rnn_weights
    rnn_recurrent_weights
    rnn_bias
    dense_weights
    dense_bias
end


# Constructor for the RNN struct
function RNN_CUST(epochs, batch_size, learning_rate)
    rnn_weights, rnn_recurrent_weights, rnn_bias, dense_weights, dense_bias = init_weights()
    RNN_CUST(epochs, batch_size, learning_rate, rnn_weights, rnn_recurrent_weights, rnn_bias, dense_weights, dense_bias)
end