import tensorflow

def create_ffn(hidden_units, dropout_rate, regularizer_weight, name=None):
    fnn_layers = []

    for units in hidden_units:
        fnn_layers.append(tensorflow.keras.layers.Dense(units,
                                                        activation=tensorflow.nn.relu,
                                                        activity_regularizer=tensorflow.keras.regularizers.l2(regularizer_weight),
                                                        bias_regularizer=tensorflow.keras.regularizers.l2(regularizer_weight)))
        fnn_layers.append(tensorflow.keras.layers.Dropout(dropout_rate))

    return tensorflow.keras.Sequential(fnn_layers, name=name)


class GraphConvLayer(tensorflow.keras.layers.Layer):
    def __init__(
        self,
        hidden_units,
        dropout_rate,
        aggregation_type,
        combination_type,
        regularizer_weight,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.aggregation_type = aggregation_type
        self.combination_type = combination_type

        self.update_fn = create_ffn(hidden_units, dropout_rate, regularizer_weight)

    def prepare(self, node_repesentations, weights=None):
        messages = node_repesentations * tensorflow.expand_dims(weights, -1)
        return messages

    def aggregate(self, node_indices, neighbour_messages, node_repesentations):
        num_nodes = node_repesentations.shape[0]
        if self.aggregation_type == "sum":
            aggregated_message = tensorflow.math.unsorted_segment_sum(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        elif self.aggregation_type == "mean":
            aggregated_message = tensorflow.math.unsorted_segment_mean(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        elif self.aggregation_type == "max":
            aggregated_message = tensorflow.math.unsorted_segment_max(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        else:
            raise ValueError(
                f"Invalid aggregation type: {self.aggregation_type}.")

        return aggregated_message

    def update(self, node_repesentations, aggregated_messages):
        if self.combination_type == "concat":
            h = tensorflow.concat([node_repesentations, aggregated_messages], axis=1)
        elif self.combination_type == "add":
            h = node_repesentations + aggregated_messages
        elif self.combination_type == "none":
            h = aggregated_messages
        else:
            raise ValueError(
                f"Invalid combination type: {self.combination_type}.")

        node_embeddings = self.update_fn(h)
        return node_embeddings

    def call(self, inputs):
        node_repesentations, edges, edge_weights = inputs
        node_indices, neighbour_indices = edges[0], edges[1]
        neighbour_repesentations = tensorflow.gather(node_repesentations, neighbour_indices)

        neighbour_messages = self.prepare(neighbour_repesentations, edge_weights)
        aggregated_messages = self.aggregate(node_indices, neighbour_messages, node_repesentations)
        return self.update(node_repesentations, aggregated_messages)


class GNNNodeClassifier(tensorflow.keras.Model):
    def __init__(
        self,
        graph_info,
        class_size,
        hidden_units,
        dropout_rate,
        aggregation_type,
        combination_type,
        regularizer_weight,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.class_size = class_size

        self.node_features = graph_info[0]
        self.edges = graph_info[1]
        self.edge_weights = graph_info[2]

        self.conv1 = GraphConvLayer(hidden_units, dropout_rate, aggregation_type, combination_type, regularizer_weight)
        self.conv2 = GraphConvLayer([class_size], dropout_rate, aggregation_type, combination_type, regularizer_weight)
        self.compute_logits = tensorflow.keras.layers.Softmax()

    def call(self, input_node_indices):
        x1 = self.conv1((self.node_features, self.edges, self.edge_weights))
        x2 = self.conv2((x1, self.edges, self.edge_weights))
        node_embeddings = tensorflow.gather(x2, tensorflow.cast(input_node_indices, tensorflow.dtypes.int64))
        return self.compute_logits(node_embeddings)

    def one_hot_kl_divergence(self, y_true, y_pred):
        y_true_one_hot = tensorflow.reshape(tensorflow.one_hot(tensorflow.cast(y_true, tensorflow.dtypes.int64), self.class_size), (-1, self.class_size))
        kl = tensorflow.keras.losses.KLDivergence()
        return kl(y_true_one_hot, y_pred)