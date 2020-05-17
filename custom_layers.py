# https://github.com/bfelbo/DeepMoji/blob/master/deepmoji/attlayer.py
class AttentionWeightedAverage(Layer):
    def __init__(self, return_attention=False, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('uniform')
        self.return_attention = return_attention
        super(AttentionWeightedAverage, self).__init__(** kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]

        assert len(input_shape) == 3
        self.W = self.add_weight(shape=(input_shape[2], 1),
                                 name='{}_W'.format(self.name),
                                 initializer=self.init)
        self.trainable_weights = [self.W]
        super(AttentionWeightedAverage, self).build(input_shape)

    def call(self, x, mask=None):
        logits, x_shape = K.dot(x, self.W), K.shape(x)
        logits = K.reshape(logits, (x_shape[0], x_shape[1]))
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))

        if mask is not None:
            ai = ai * K.cast(mask, K.floatx())
        
        ai_sum = K.sum(ai, axis=1, keepdims=True)
        att_weights = ai / (ai_sum + K.epsilon())
        result = K.sum(x * K.expand_dims(att_weights), axis=1)
        if self.return_attention: return [result, att_weights]

        return result

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len),
                    (input_shape[0], input_shape[1])]
 
        return (input_shape[0], output_len)

    def compute_mask(self, input, inp=None):
        is_list = isinstance(inp, list)

        if not is_list:
            return None
        else:
            return None * len(input_mask)
